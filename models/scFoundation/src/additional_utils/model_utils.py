
import torch, torch.nn.functional as F
from collections import OrderedDict

# -------------------- utils --------------------
def _strip_prefix(k):
    for p in ("state_dict.", "model.", "module."):
        if k.startswith(p): return k[len(p):]
    return k

def load_branch_autointersect(model, ckpt_path, branch="gene", device="cuda"):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    src = ckpt[branch]["state_dict"]
    src = OrderedDict((_strip_prefix(k), v) for k, v in src.items())

    dst = model.state_dict()
    loadable = {k: v for k, v in src.items() if k in dst and v.shape == dst[k].shape}

    missing = [k for k in dst.keys() if k not in loadable]
    skipped = [k for k in src.keys() if k not in loadable]

    print(f"[autointersect] loadable: {len(loadable)} tensors "
          f"(skipped_from_ckpt: {len(skipped)}, missing_in_model: {len(missing)})")

    model.load_state_dict(loadable, strict=False)
    return model.to(device).train()


def build_finetune_batch(x, y, pad_token_id, seq_len):
    B, N = x.shape; dev = x.device
    pad = (x == pad_token_id)
    pos = torch.arange(N, device=dev).repeat(B, 1); pos[pad] = seq_len
    dec = x.clone(); dpad = (dec == pad_token_id)
    dpos = torch.arange(N, device=dev).repeat(B, 1); dpos[dpad] = seq_len
    return {
        "x": x,
        "padding_label": pad,
        "encoder_pos_ids": pos,
        "encoder_labels": ~pad,
        "decoder_data": dec,
        "mask_gene_name": False,
        "mask_labels": torch.zeros(B, N, dtype=torch.bool, device=dev),
        "decoder_pos_ids": dpos,
        "decoder_pad_labels": dpad,
        "target": y,
    }

def build_optimizer(model, lr_head=1e-3, lr_enc=1e-4, wd=0.01):
    head_pref = ("decoder.", "decoder_embed.", "to_final.")
    enc, head = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        (head if n.startswith(head_pref) else enc).append(p)
    return torch.optim.AdamW([{"params": enc, "lr": lr_enc},
                              {"params": head, "lr": lr_head}], weight_decay=wd)

def freeze_encoder(model, freeze_token_emb=False, freeze_pos_emb=False):
    for n, p in model.named_parameters():
        if n.startswith("encoder."): p.requires_grad = False
        if freeze_token_emb and n.startswith(("token_emb.","encoder.token_emb.")): p.requires_grad = False
        if freeze_pos_emb   and n.startswith(("pos_emb.","encoder.pos_emb.")):     p.requires_grad = False

def check_nan_loss(loss, y_pred, y_true, observed):
    """loss가 NaN/Inf일 때 간단한 원인 출력"""
    if torch.isnan(loss) or torch.isinf(loss):
        print("\n[⚠️ NaN/Inf detected]")
        print(f"  observed.sum() = {observed.sum().item()} / {observed.numel()}")
        print(f"  y_pred: nan={torch.isnan(y_pred).sum().item()}, inf={torch.isinf(y_pred).sum().item()}, "
              f"max={y_pred.abs().max().item():.3e}")
        print(f"  y_true: nan={torch.isnan(y_true).sum().item()}, inf={torch.isinf(y_true).sum().item()}, "
              f"max={y_true.abs().max().item():.3e}")
        if observed.sum() == 0:
            print("  ⚠️ observed mask가 전부 False입니다. (loss 계산 대상 없음)")
        return True
    return False

def set_heads(model_config, enc_heads=None, dec_heads=None):
    # ----- encoder -----
    if enc_heads is not None:
        enc_dim = model_config["encoder"]["hidden_dim"]
        assert enc_dim % enc_heads == 0, f"encoder: {enc_dim} % {enc_heads} != 0"
        model_config["encoder"]["heads"] = enc_heads
        # head_dim 갱신 (768 // 24 = 32 같은 식)
        # model_config["encoder"]["dim_head"] = enc_dim // enc_heads

    # ----- decoder -----
    if dec_heads is not None:
        dec_dim = model_config["decoder"]["hidden_dim"]
        assert dec_dim % dec_heads == 0, f"decoder: {dec_dim} % {dec_heads} != 0"
        model_config["decoder"]["heads"] = dec_heads



def infer_cond_in_dim(cache):
    """
    embedding_cache 구조를 대충 이렇게 가정:
      - {"TP53": {"Y220C": arr, "R248Q": arr, ...}, ...}
      - arr는 (L, D) 또는 (D,) 또는 리스트(list[list[float]]) 등일 수 있음
    """
    # 1) 아무 gene 하나 뽑기
    any_gene_emb = random.choice(list(cache.values()))

    # 2) gene 아래가 dict면 (variant dict)에서 하나 더 뽑기
    if isinstance(any_gene_emb, dict):
        any_var_emb = random.choice(list(any_gene_emb.values()))
    else:
        any_var_emb = any_gene_emb

    # 3) numpy array인 경우 → 마지막 차원 사용
    if isinstance(any_var_emb, np.ndarray):
        if any_var_emb.ndim >= 2:
            return any_var_emb.shape[-1]  # (L, D) or (1, D) → D
        else:
            return any_var_emb.shape[0]   # (D,)인 경우

    # 4) 파이썬 리스트인 경우
    if isinstance(any_var_emb, list):
        # (L, D) 형태의 list[list[float]]라고 가정
        if len(any_var_emb) == 0:
            raise ValueError("embedding_cache 안에 빈 리스트가 있습니다.")
        first = any_var_emb[0]
        if isinstance(first, (list, tuple, np.ndarray)):
            return len(first)  # (L, D)에서 D
        else:
            return len(any_var_emb)      # (D,)인 경우

    raise TypeError(f"알 수 없는 임베딩 타입: {type(any_var_emb)}")