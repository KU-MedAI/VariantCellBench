import os
import torch
import sys
import json



if '/home/tech/variantseq/seunghun' not in sys.path:
    sys.path.append('/home/tech/variantseq/seunghun')
from config import *
from network import *
from result import *
from variant import *
from utils import *
from bsh import *



config = {
    'model':'scFoundation',
    'data_name':'kim2023_hct116_[benchmark][1_3-fold]',
    'adata_name':'perturb_processed_metadata',
    'gex_layer':'counts',
    'embedding_model': 'esm2_t33_650M_UR50D', # 'esm2_t33_650M_UR50D' / 'esm_msa1_t12_100M_UR50S' / 'ProtT5'
    'variant_representation': 'ALT',    # 'ALT' / 'DIFF'
    'compression':'position_embedding', # 'full_sequence_average' / 'position_embedding'
    
    'split':'exist',
    'data_seed': 1,
    'epochs': 1,
    'batch_size': 8,
    # 'hidden_dim': 1280,                       # 자동 매칭 해놓음
    # 'pad_variant_emb': 1280,                # variant embedding의 차원 수를 고정해야하는 경우 사용
    # 'learning_rate': 1e-3,
    # 'weight_decay': 5e-4,                     # 5e-4
    'trainer': 'Ban',
    'use_dummy_embedding': False,
    'visible_emb': False                    # Embedding 꺼내먹기 False or True
}


# # --- read override json if provided -- #
# import os, json
# _override_path = os.environ.get("CONFIG_OVERRIDE_PATH")
# if _override_path and os.path.exists(_override_path):
#     with open(_override_path) as f:
#         config.update(json.load(f))
# # ------------------------------------- #


# 데이터 불러오기
variantseq = CustomConditionData(GEARS_DATA_PATH)
variantseq.load(config)



def _pick_num_heads(d_model: int, max_head_dim: int = 64) -> int:
    """d_model을 나누는 값 중 head_dim <= max_head_dim인 가장 큰 head 수 반환"""
    cands = [h for h in range(1, d_model + 1) if d_model % h == 0 and (d_model // h) <= max_head_dim]
    return max(cands) if cands else 1

def _validate_heads(name: str, d_model: int, heads: int):
    assert d_model % heads == 0, f"[{name}] d_model({d_model}) % heads({heads}) != 0"



# scFoundation model 선언부

import torch
from torch import nn
import pandas as pd
from pretrainmodels.performer import PerformerModule
from pretrainmodels.transformer import pytorchTransformerModule
from pretrainmodels.mae_autobin import AutoDiscretizationEmbedding2
torch.backends.cudnn.benchmark = False  # 입력 길이 달라지는 상황에서 과도한 워크스페이스 방지
class MaeAutobin(nn.Module):
    def __init__(self, *,
        num_tokens, max_seq_len, embed_dim, decoder_embed_dim,
        bin_alpha=1.0, bin_num=100, pad_token_id=103, mask_token_id=102,
        cond_in_dim=1280,                              # 변이 임베딩 차원
        gene_index_tsv="/home/tech/variantseq/foundation/scFoundation/model/OS_scRNA_gene_index.19264.tsv",
        embedding_cache=None,                           # {(gene, variant): np.ndarray or Tensor}
        model_config,
        config
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.num_tokens  = num_tokens
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id

        # (1) 임베딩
        self.token_emb = AutoDiscretizationEmbedding2(
            embed_dim, max_seq_len, bin_num=bin_num, bin_alpha=bin_alpha,
            pad_token_id=pad_token_id, mask_token_id=mask_token_id
        )
        self.pos_emb = nn.Embedding(max_seq_len + 1, embed_dim)

        # (2) 변이 임베딩 축소 MLP
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_in_dim, embed_dim, bias=True),
        )

        # (3) gene index 로드 < - 이거 기준으로 variant 매핑할겁니다.
        # scFoundation 기준 유전자 리스트
        gi = pd.read_csv(gene_index_tsv, sep="\t")
        self.gene2idx = {str(g): int(i) for g, i in zip(gi["gene_name"], gi["index"])}

        self.embedding_cache = embedding_cache or {}


        # ----- (4) Encoder/Decoder/Head (기존 구성 유지하되 헤드/차원 일관성 체크) -----
        enc_cfg = model_config['encoder']
        dec_cfg = model_config['decoder']

        enc_dim   = enc_cfg['hidden_dim']
        enc_heads = enc_cfg.get('heads')
        if enc_heads is None:
            enc_heads = _pick_num_heads(enc_dim, max_head_dim=64)
        _validate_heads("encoder", enc_dim, enc_heads)

        dec_dim   = dec_cfg['hidden_dim']
        dec_heads = dec_cfg.get('heads')
        if dec_heads is None:
            dec_heads = _pick_num_heads(dec_dim, max_head_dim=64)
        _validate_heads("decoder", dec_dim, dec_heads)

        # Performer용 dim_head 일관화: 제공 안되면 자동 유도
        dec_dim_head = dec_cfg.get('dim_head')
        if dec_dim_head is None:
            # 가장 흔한 선택: dec_dim // dec_heads
            dec_dim_head = dec_dim // dec_heads
        # Performer 구현이 (heads * dim_head == dim) 전제를 가진다면 체크
        assert dec_heads * dec_dim_head == dec_dim, \
            f"[decoder] heads({dec_heads}) * dim_head({dec_dim_head}) != hidden_dim({dec_dim})"
        

        # (4) Encoder/Decoder/Head (기존 구성 유지)
        # Encoder
        self.encoder = pytorchTransformerModule(
            max_seq_len,
            dim=enc_dim,
            depth=enc_cfg['depth'],
            heads=enc_heads,
        )

        # Decoder (Performer)
        self.decoder = PerformerModule(
            max_seq_len,
            dim=dec_dim,
            depth=dec_cfg['depth'],
            heads=dec_heads,
            dim_head=dec_dim_head,
            ff_dropout=dec_cfg.get('ff_dropout', 0.0),
            attn_dropout=dec_cfg.get('attn_dropout', 0.0),
        )
        
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.norm = nn.LayerNorm(decoder_embed_dim)
        self.to_final = nn.Linear(decoder_embed_dim, 1)
        self.config = config

    # === 헬퍼들 ===
    def _get_cached_embedding(self, gene, var_key):
        arr = self.embedding_cache.get((gene, var_key), None)
        if arr is None: return None
        t = torch.as_tensor(arr) if not torch.is_tensor(arr) else arr
        if t.dim() == 2: t = t.unsqueeze(0)      # (L, Dv) -> (1, L, Dv)
        return t

    @staticmethod
    def _pad_to_length(t: torch.Tensor, target_L: int):
        B, L, D = t.shape
        if L >= target_L: return t
        return torch.cat([t, t.new_zeros((B, target_L - L, D))], dim=1)

    # === forward ===
    def forward(self, x, padding_label, encoder_position_gene_ids, encoder_labels,
                decoder_data, mask_gene_name, mask_labels, decoder_position_gene_ids,
                decoder_data_padding_labels, data, output_attentions=False, **kwargs):
        variant_representation = self.config.get("variant_representation", 'ALT') # 'ALT' / 'DIFF'
        compression = self.config.get("compression", 'position_embedding') # 'full_sequence_average' / 'position_embedding'
        B, Kenc = x.shape[0], x.shape[1]
        device, f32 = x.device, torch.float32

        # 1) 토큰+포지션 임베딩
        x = self.token_emb(torch.unsqueeze(x, 2), output_weight=0)      # [B, Kenc, E]
        if output_attentions: x.requires_grad_()
        pos = self.pos_emb(encoder_position_gene_ids)                    # [B, Kenc, E]
        x = x + pos

        # 2) 변이 주입 (batch["variant"] = List[List["GENE~ALT"]])
        cond_batch = kwargs.get("variant", None) or kwargs.get("condition", None)
        if cond_batch is not None:
            for b in range(B):
                specs = cond_batch[b]
                if not specs: continue
                # 유전자별 누적 벡터
                acc = {}
                for spec in specs:
                    if not isinstance(spec, str) or "~" not in spec: continue
                    gene_, var_ = spec.split("~", 1)
                    # 유전자 index 가져옵니다. (ex. TP53 -> n)
                    gidx = int(self.gene2idx.get(gene_, -1))
                    if gidx < 0: continue

                    # ALT와 REF 시퀀스 임베딩을 캐시파일에서 로드해옵니다. 캐시파일은 데이터 로드 시 불러옵니다.
                        # 전체 서열 임베딩을 사용하는 경우. REF와 ALT를 각각 가져옴
                    if compression == 'full_sequence_average':
                        ref = get_cached_embedding(gene_, "REF", data.embedding_cache)
                        alt = get_cached_embedding(gene_, var_, data.embedding_cache)
                    # 변이 발생 위치의 임베딩을 사용하는 경우.
                        # 설정한 variant_representation에 따라 ALT or DIFF 임베딩을 가져옴
                    elif compression == 'position_embedding':
                        ref = get_cached_embedding(gene_, var_, data.embedding_cache, variant_representation)
                        alt = get_cached_embedding(gene_, var_, data.embedding_cache, variant_representation)
                    else:
                        pass
                    
                    if ref is None or alt is None: continue
                    # 불필요한 변환 방지
                    if not isinstance(ref, torch.Tensor):
                        ref = torch.tensor(ref, dtype=torch.float32, device=device)
                        alt = torch.tensor(alt, dtype=torch.float32, device=device)

                    else:
                        ref = ref.to(dtype=torch.float32, device=device)
                        alt = alt.to(dtype=torch.float32, device=device)


                    if compression == 'full_sequence_average':
                        L = max(ref.shape[1], alt.shape[1])
                        ref = self._pad_to_length(torch.tensor(ref), L)
                        alt = self._pad_to_length(torch.tensor(alt), L)
                        if variant_representation == 'DIFF':
                            delta = (ref - alt).mean(dim=1).squeeze(0)           # [Dv]
                        elif variant_representation == 'ALT':
                            delta = alt.mean(dim=1).squeeze(0)
                    elif compression == 'position_embedding':
                        delta = alt

                    vecE  = self.cond_mlp(delta).to(x.dtype)             # [E]
                    acc[gidx] = acc.get(gidx, 0) + vecE

                if acc:
                    pos_ids_b = encoder_position_gene_ids[b]             # [Kenc]
                    for gidx, v in acc.items():
                        m = (pos_ids_b == gidx)
                        if m.any():
                            x[b, m] = x[b, m] + v

        # 3) Encoder
        x = self.encoder(x, padding_mask=padding_label)

        # 4) Decoder 입력 준비 및 주입
        dec = self.token_emb(torch.unsqueeze(decoder_data, 2))
        posd = self.pos_emb(decoder_position_gene_ids)
        if mask_gene_name:
            raise NotImplementedError("mask_gene_name")
        batch_idx, gen_idx = (encoder_labels == True).nonzero(as_tuple=True)
        dec[batch_idx, gen_idx] = x[~padding_label].to(dec.dtype)
        dec = dec + posd
        dec = self.decoder_embed(dec)

        # 5) Decoder
        x = self.decoder(dec, padding_mask=decoder_data_padding_labels)
        x = self.norm(x)
        x = self.to_final(x)
        return x.squeeze(2)                     # [B, N]

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

# -------------------- train/eval steps --------------------
import torch
import torch.nn.functional as F
from torch.amp import autocast as amp_autocast          # 최신 API
from torch.cuda.amp import GradScaler                   # CUDA AMP 스케일러

def _to_device_batch(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}

def train_step(
    model,
    batch,
    optimizer,
    *,
    device: str | torch.device | None = None,
    use_amp: bool = False,
    scaler: GradScaler | None = None,
    max_grad_norm: float = 1.0,
    check_grad_finite: bool = False,
    amp_dtype: torch.dtype | None = torch.bfloat16,   # A100/H100 권장
) -> float:
    """
    통합형 학습 스텝 (AMP 토글)
      - use_amp=False: FP32
      - use_amp=True : autocast + GradScaler (scaler 필수)
    반환: loss.item()
    """

    y = model(
        x=batch["x"],
        padding_label=batch["padding_label"],
        encoder_position_gene_ids=batch["encoder_pos_ids"],
        encoder_labels=batch["encoder_labels"],
        decoder_data=batch["decoder_data"],
        mask_gene_name=batch["mask_gene_name"],
        mask_labels=batch["mask_labels"],
        decoder_position_gene_ids=batch["decoder_pos_ids"],
        decoder_data_padding_labels=batch["decoder_pad_labels"],
        variant=batch.get("variant", None),
    )

    # 관측 마스크(True=유효 위치)
    obs = ~batch["decoder_pad_labels"]
    if obs.dtype != torch.bool:
        obs = obs.bool()
    if obs.device != y.device:
        obs = obs.to(y.device)

    per_elem = F.mse_loss(y, batch["target"], reduction="none")
    obs_f = obs.to(per_elem.dtype)
    denom = obs_f.sum().clamp_min(1)
    loss = (per_elem * obs_f).sum() / denom

    if not torch.isfinite(loss):
        raise RuntimeError("NaN/Inf loss detected.")

    return loss


from torch.amp import autocast

def eval_step(model, batch, *, device, use_amp=True, amp_dtype=torch.bfloat16):

    # (필요시) batch to(device) 생략
    with torch.inference_mode():
        ctx = autocast(device_type="cuda", enabled=(use_amp and device=="cuda"), dtype=amp_dtype)
        with ctx:
            y = model(
                x=batch["x"],
                padding_label=batch["padding_label"],
                encoder_position_gene_ids=batch["encoder_pos_ids"],
                encoder_labels=batch["encoder_labels"],
                decoder_data=batch["decoder_data"],
                mask_gene_name=batch["mask_gene_name"],
                mask_labels=batch["mask_labels"],
                decoder_position_gene_ids=batch["decoder_pos_ids"],
                decoder_data_padding_labels=batch["decoder_pad_labels"],
                variant=batch.get("variant", None),
            )
        obs = ~batch["decoder_pad_labels"]
        loss_per_elem = F.mse_loss(y, batch["target"], reduction="none")
        if obs.shape == loss_per_elem.shape:
            loss_per_elem = loss_per_elem[obs]
        obs_f = obs.to(loss_per_elem.dtype)
        # 반환은 (스칼라 손실, 관측 수)
        loss = loss_per_elem.mean().item()
        n_obs = int(obs.sum().item()) if obs.ndim>0 else loss_per_elem.numel()

    return loss, n_obs

def _micro_iter(batch, micro_bs):
    keys = ["x","padding_label","encoder_pos_ids","encoder_labels",
            "decoder_data","mask_gene_name","mask_labels",
            "decoder_pos_ids","decoder_pad_labels","target","variant"]
    N = batch["target"].shape[0]
    for s in range(0, N, micro_bs):
        e = min(s+micro_bs, N)
        mb = {}
        for k in keys:
            v = batch.get(k, None)
            if torch.is_tensor(v) and v.dim() >= 1 and v.size(0) == N:
                mb[k] = v[s:e]
            else:
                mb[k] = v
        yield mb

def validate(model, valid_loader, device="cuda", *, needs_build=False, gom=None,
             model_config=None, weighted=True, use_amp=True, amp_dtype=torch.bfloat16,
             micro_bs=None):
    # model.to(device)

    tot_loss, tot_obs, n_batches = 0.0, 0, 0
    # with torch.inference_mode():
    with torch.no_grad():
        for b in valid_loader:

            if needs_build:
                x = b.x.to(device)
                y = b.y.to(device)
                batch = build_finetune_batch(x, y, model_config["pad_token_id"], model_config["seq_len"])
                if isinstance(b, dict) and "variant" in b:
                    batch["variant"] = b["variant"]
                elif hasattr(b, "variant"):
                    batch["variant"] = b.variant
            else:
                batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                            for k, v in b.items()}

            # 마이크로배치 평가
            if micro_bs is None:
                loss_item, n_obs = eval_step(model, batch, device=device, use_amp=use_amp, amp_dtype=amp_dtype)
                if weighted:
                    tot_loss += loss_item * n_obs; tot_obs += n_obs
                else:
                    tot_loss += loss_item; n_batches += 1
            else:
                # 누적 평균
                mb_loss_sum, mb_obs_sum, mb_cnt = 0.0, 0, 0
                for mb in _micro_iter(batch, micro_bs):
                    loss_item, n_obs = eval_step(model, mb, device=device, use_amp=use_amp, amp_dtype=amp_dtype)
                    if weighted:
                        mb_loss_sum += loss_item * n_obs; mb_obs_sum += n_obs
                    else:
                        mb_loss_sum += loss_item; mb_cnt += 1
                if weighted:
                    tot_loss += mb_loss_sum; tot_obs += mb_obs_sum
                else:
                    tot_loss += (mb_loss_sum / max(mb_cnt, 1)); n_batches += 1

        return (tot_loss / max(tot_obs, 1)) if weighted else (tot_loss / max(n_batches, 1))





# #### Variant-seq gene order -> scFoundation gene order
# 데이터의 유전자 순서를 매핑하는 유틸 (핵심 베이스 코드)
# to_model_order(): perturb-seq을 모델 유전자 순서에 맞게 정렬
# to_data_order(): 정렬된 유전자를 원본 perturb-seq에 맞게 복원

from dataclasses import dataclass
from typing import Sequence, List, Optional, Tuple
import torch

# -------------------------
# 기존 GeneOrderMap 그대로 사용
# -------------------------
@dataclass
class GeneOrderMap:
    data_genes: List[str]
    model_genes: List[str]
    to_model_index: torch.LongTensor
    to_model_missing_mask: torch.BoolTensor
    to_data_index: torch.LongTensor
    to_data_missing_mask: torch.BoolTensor
    missing_in_data: List[str]
    missing_in_model: List[str]

def build_gene_order_map(
    data_genes: Sequence[str],
    model_genes: Sequence[str],
    device: Optional[torch.device] = None,
) -> GeneOrderMap:
    data_genes = list(data_genes)
    model_genes = list(model_genes)

    data_idx = {}
    for i, g in enumerate(data_genes):
        if g not in data_idx:
            data_idx[g] = i
    model_idx = {g: i for i, g in enumerate(model_genes)}

    to_model_idx, to_model_missing = [], []
    for g in model_genes:
        if g in data_idx:
            to_model_idx.append(data_idx[g]); to_model_missing.append(False)
        else:
            to_model_idx.append(0);          to_model_missing.append(True)

    to_data_idx, to_data_missing = [], []
    for g in data_genes:
        if g in model_idx:
            to_data_idx.append(model_idx[g]); to_data_missing.append(False)
        else:
            to_data_idx.append(0);            to_data_missing.append(True)

    to_model_index = torch.tensor(to_model_idx, dtype=torch.long, device=device)
    to_model_missing_mask = torch.tensor(to_model_missing, dtype=torch.bool, device=device)
    to_data_index = torch.tensor(to_data_idx, dtype=torch.long, device=device)
    to_data_missing_mask = torch.tensor(to_data_missing, dtype=torch.bool, device=device)

    missing_in_data = [g for g, m in zip(model_genes, to_model_missing) if m]
    missing_in_model = [g for g, m in zip(data_genes, to_data_missing) if m]

    return GeneOrderMap(
        data_genes=data_genes, model_genes=model_genes,
        to_model_index=to_model_index, to_model_missing_mask=to_model_missing_mask,
        to_data_index=to_data_index, to_data_missing_mask=to_data_missing_mask,
        missing_in_data=missing_in_data, missing_in_model=missing_in_model,
    )

# -------------------------
# 레이아웃 어댑터
# -------------------------
def _infer_B_from_total(total: int, G: int) -> int:
    assert total % G == 0, f"Total({total}) not divisible by G({G})"
    return total // G

def _to_BGF(x: torch.Tensor, layout: str, G: int) -> Tuple[torch.Tensor, int, int]:
    """
    입력 텐서를 (B, G, F) 표준형으로 변환.
    지원 레이아웃:
      - 'BG'        : (B, G)
      - 'BGF'       : (B, G, F)
      - 'B*G'       : (B*G,)
      - '1xB*G'     : (1, B*G)
      - 'GB'        : (G, B)
      - 'G'         : (G,)
      - 'GF'        : (G, F)     <-- 추가
    반환: (x_bgf, B, F)
    """
    if layout == "BG":
        B = x.size(0); assert x.size(1) == G
        return x.unsqueeze(-1), B, 1

    if layout == "BGF":
        B = x.size(0); assert x.size(1) == G
        F = x.size(2)
        return x, B, F

    if layout == "B*G":
        total = x.numel()
        B = _infer_B_from_total(total, G)
        return x.view(B, G, 1), B, 1

    if layout == "1xB*G":
        assert x.dim() == 2 and x.size(0) == 1
        total = x.size(1)
        B = _infer_B_from_total(total, G)
        return x.view(1, B*G).view(B, G, 1), B, 1

    if layout == "GB":
        assert x.size(0) == G
        B = x.size(1)
        return x.permute(1, 0).contiguous().unsqueeze(-1), B, 1

    if layout == "G":
        assert x.numel() == G
        return x.view(1, G, 1), 1, 1

    if layout == "GF":                      # <-- 추가
        assert x.size(0) == G
        F = x.size(1)
        return x.unsqueeze(0), 1, F

    raise ValueError(f"Unsupported input layout: {layout}")


def _from_BGF(x_bgf: torch.Tensor, layout: str, B: int, G: int, F: int) -> torch.Tensor:
    """
    (B, G, F) → 원하는 레이아웃으로 복구
    """
    assert x_bgf.shape == (B, G, F)

    if layout == "BG":
        assert F == 1
        return x_bgf.squeeze(-1)

    if layout == "BGF":
        return x_bgf

    if layout == "B*G":
        assert F == 1
        return x_bgf.view(B*G)

    if layout == "1xB*G":
        assert F == 1
        return x_bgf.view(1, B*G)

    if layout == "GB":
        assert F == 1
        return x_bgf.squeeze(-1).permute(1, 0).contiguous()

    if layout == "G":
        assert F == 1 and B == 1
        return x_bgf.view(G)

    if layout == "GF":                      # <-- 추가
        assert B == 1
        return x_bgf.squeeze(0)

    raise ValueError(f"Unsupported output layout: {layout}")


# -------------------------
# 재정렬 본체
# -------------------------
@torch.no_grad()
def to_model_order(
    x: torch.Tensor,
    gom: GeneOrderMap,
    in_layout: str,
    out_layout: str,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """
    데이터 유전자 순서(G_data) → 모델 유전자 순서(G_model)
    어떤 레이아웃이 와도 동작: in_layout/out_layout로 명시
    """
    # 데이터/모델 유전자 길이(Gd, Gm) 파악
    Gd = len(gom.data_genes)
    Gm = len(gom.model_genes)

    # 내부 표준형 (B, Gd, F) 로 변환.
    # B(배치 크기), F(특징 차원)
    x_bgf, B, F = _to_BGF(x, in_layout, Gd)         # (B, Gd, F)
    out = x_bgf.new_full((B, Gm, F), fill_value)    # (B, Gm, F)
    # 모델 유전자 길이 Gm로 만들고 fill_value로 초기화. 데이터에 없는 유전자는 fill_value값으로 패딩
    # 원 텐서의 속성(device/dtype) 을 자동 상속
    # out = torch.full((B,Gm,F), fill_value, dtype=x_bgf.dtype, device=x_bgf.device)


    # 모델에 필요한 유전자 중 실제 데이터에 존재하는 것만 골라서 복사
    present = ~gom.to_model_missing_mask            # (Gm,)
    if present.any():
        src_idx = gom.to_model_index[present]       # indices into Gd
        out[:, present, :] = x_bgf[:, src_idx, :]

    return _from_BGF(out, out_layout, B, Gm, F)

@torch.no_grad()
def to_data_order(
    x: torch.Tensor,
    gom: GeneOrderMap,
    in_layout: str,
    out_layout: str,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """
    모델 유전자 순서(G_model) → 데이터 유전자 순서(G_data)
    """
    Gd = len(gom.data_genes)
    Gm = len(gom.model_genes)

    x_bgf, B, F = _to_BGF(x, in_layout, Gm)         # (B, Gm, F)
    out = x_bgf.new_full((B, Gd, F), fill_value)    # (B, Gd, F)

    present = ~gom.to_data_missing_mask             # (Gd,)
    if present.any():
        src_idx = gom.to_data_index[present]        # indices into Gm
        out[:, present, :] = x_bgf[:, src_idx, :]

    return _from_BGF(out, out_layout, B, Gd, F)

@torch.no_grad()
def reorder_batch_x(
    x: torch.Tensor,
    gom: GeneOrderMap,
    fill_value: float = 0.0
) -> torch.Tensor:
    """
    PyG 입력 batch.x 변환: (B*Gd, 1) -> (B, Gm)
    """
    return to_model_order(
        x.squeeze(-1),   # (B*Gd,)
        gom,
        in_layout="B*G",
        out_layout="BG",
        fill_value=fill_value,
    )


@torch.no_grad()
def reorder_batch_y(
    y: torch.Tensor,
    gom: GeneOrderMap,
    fill_value: float = 0.0
) -> torch.Tensor:
    """
    PyG 출력 batch.y 변환: (B, Gd) -> (B, Gm)
    """
    return to_model_order(
        y,   # (B, Gd)
        gom,
        in_layout="BG",
        out_layout="BG",
        fill_value=fill_value,
    )

import torch
import numpy as np

def to_long_1d(x, device=None) -> torch.Tensor:
    """
    list/tuple/np.ndarray/torch.Tensor 모두 받아서 1D LongTensor로 반환.
    None이면 None 그대로 반환.
    """
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.to(dtype=torch.long, device=device).view(-1)
    # list/tuple/np.ndarray 처리
    return torch.as_tensor(x, dtype=torch.long, device=device).view(-1)


def build_data2model_pos(gom: GeneOrderMap) -> torch.LongTensor:
    """
    data 유전자 순서의 각 위치(0..Gd-1)에 대응하는 model 위치를 담은 벡터를 만든다.
    - 존재하지 않는(모델에 없는) data 위치는 -1로 표기.
    """
    Gd = len(gom.data_genes)
    d2m = torch.full((Gd,), -1, dtype=torch.long)  # 기본 -1
    # gom.to_model_index[k] = data에서의 위치, k = model 위치
    for k in range(len(gom.model_genes)):
        if not gom.to_model_missing_mask[k]:
            d_pos = int(gom.to_model_index[k])
            d2m[d_pos] = k
    return d2m


def map_index_from_data_to_model(idx_tensor: torch.Tensor,
                                 d2m: torch.LongTensor,
                                 drop_invalid: bool = True,
                                 unique: bool = False,
                                 sort: bool = False) -> torch.Tensor:
    """
    data 순서의 인덱스 텐서를 model 순서 인덱스로 변환.
    - idx_tensor: 1D long tensor (예: de_idx, de_non_dropout_idx, hvg_total_idx 등)
    """
    m = d2m[idx_tensor]  # 같은 길이, 일부는 -1
    if drop_invalid:
        m = m[m >= 0]
    if unique:
        m = torch.unique_consecutive(m) if sort is False else torch.unique(m)
    if sort:
        m, _ = torch.sort(m)
    return m

from torch.utils.data import Dataset
from copy import deepcopy

class ReorderedVariantSeqDataset(Dataset):
    """
    base_dataset의 각 sample(Data)을 모델 유전자 순서(Gm)로 미리 재정렬해 보관.
    x, y는 물론, de_idx / de_non_dropout_idx / hvg_total_idx 같은 'data 순서 인덱스'도
    model 순서 인덱스로 변환해 둔다.
    """
    def __init__(self, base_dataset, gom: GeneOrderMap, fill_value: float = 0.0, clone: bool = True):
        self.gom = gom
        self.fill_value = fill_value
        self.model_G = len(gom.model_genes)
        self.data2model_pos = build_data2model_pos(gom)  # (Gd,)-> model pos or -1

        self.data_list = []
        for i in range(len(base_dataset)):
            d = base_dataset[i]
            d = deepcopy(d) if clone else d

            # --- x: (Gd,1) 또는 (Gd,)
            x = d.x if isinstance(d.x, torch.Tensor) else torch.as_tensor(d.x)
            if x.dim() == 2 and x.size(1) == 1:
                x_gm = to_model_order(x.squeeze(-1), gom, "B*G", "BG", fill_value)#.unsqueeze(-1)  # (Gm,1)
            else:
                x_gm = to_model_order(x.view(-1), gom, "B*G", "BG", fill_value)#.unsqueeze(-1)     # (Gm,1)
            d.x = x_gm

            # --- y: (1,Gd) -> (1,Gm)
            if hasattr(d, "y") and d.y is not None:
                y = d.y if isinstance(d.y, torch.Tensor) else torch.as_tensor(d.y)
                assert y.dim() == 2 and y.size(0) == 1, f"y shape unexpected: {tuple(y.shape)}"
                d.y = to_model_order(y, gom, "BG", "BG", fill_value)  # (1, Gm)

            # --- 인덱스 필드들: list -> 1D long tensor 로 변환 후 매핑
            if hasattr(d, "de_idx") and d.de_idx is not None:
                de_idx_1d = to_long_1d(d.de_idx)  # <<< 변경
                d.de_idx = map_index_from_data_to_model(de_idx_1d, self.data2model_pos,
                                                        drop_invalid=True, unique=False, sort=False)

            if hasattr(d, "de_non_dropout_idx") and d.de_non_dropout_idx is not None:
                de_non_1d = to_long_1d(d.de_non_dropout_idx)  # <<< 변경
                d.de_non_dropout_idx = map_index_from_data_to_model(de_non_1d, self.data2model_pos,
                                                                    drop_invalid=True, unique=False, sort=False)

            if hasattr(d, "hvg_total_idx") and d.hvg_total_idx is not None:
                hvg_1d = to_long_1d(d.hvg_total_idx)  # <<< 변경
                mapped = map_index_from_data_to_model(hvg_1d, self.data2model_pos,
                                                      drop_invalid=True, unique=False, sort=False)
                d.hvg_total_idx = mapped if mapped.numel() == self.model_G \
                                  else torch.arange(self.model_G, dtype=torch.long)

            d.num_nodes = self.model_G
            self.data_list.append(d)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
