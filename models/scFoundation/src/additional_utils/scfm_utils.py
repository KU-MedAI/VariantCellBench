# training pipeline
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


import os, re, json, time, signal, math, torch
from torch import nn
from torch.cuda import amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import tqdm

# -------------------------
# 유틸: 시드/이름/체크포인트
# -------------------------
def seed_all(seed: int = 42):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

SAFE_DELIMS = "_+_"
def sanitize_name(s: str) -> str:
    # Windows 금지문자 <>:"/\|?* 제거, 공백 -> '-'
    s = re.sub(r'[<>:"/\\|?*]+', '', s)
    s = s.replace(' ', '-')
    return s

def make_project_name(config, timestamp):
    # 따옴표 오류 방지: 키에는 꼭 "더블쿼트" 사용!
    parts = [
        str(config["model"]),
        str(config["data_name"]),
        str(config["embedding_model"]),
        str(config["variant_representation"]),
        str(config["compression"]),
        str(timestamp),
    ]
    return sanitize_name(SAFE_DELIMS.join(parts))

def save_checkpoint(path, model, optimizer, epoch, global_step, best_val=None, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        # "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_val": best_val,
        "extra": extra or {},
    }
    torch.save(state, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cuda"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer and ckpt.get("optimizer"): optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"): scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt

# 안전 종료 시 마지막 저장
def _handle_sigterm(*args):
    save_checkpoint(last_ckpt, model, optimizer, scheduler, epoch, global_step, best_val)
    print("[WARN] Received SIGTERM: saved last checkpoint and exiting.")
    raise SystemExit

def _to_dev_batch(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    # padding mask는 bool 권장(True=무시)
    for mk in ("padding_label", "decoder_pad_labels"):
        if mk in out and torch.is_tensor(out[mk]) and out[mk].dtype is not torch.bool:
            out[mk] = out[mk].bool()
    return out

def forward_eval(model, batch_, *, device="cuda", use_amp=True, amp_dtype=torch.bfloat16):
    """검증/추론 단계: inference_mode로 메모리 절약 + AMP"""
    b = _to_dev_batch(batch_, device)
    with torch.inference_mode():
        ctx = amp_autocast(device_type="cuda", enabled=(use_amp and str(device).startswith("cuda")), dtype=amp_dtype)
        with ctx:
            y_pred = model(
                x=b["x"],
                padding_label=b["padding_label"],
                encoder_position_gene_ids=b["encoder_pos_ids"],
                encoder_labels=b["encoder_labels"],
                decoder_data=b["decoder_data"],
                mask_gene_name=b["mask_gene_name"],
                mask_labels=b["mask_labels"],
                decoder_position_gene_ids=b["decoder_pos_ids"],
                decoder_data_padding_labels=b["decoder_pad_labels"],
                variant=b.get("variant", None),
            )
    return y_pred

def _normalize_variant_labels(variant_field, B):
    """
    batch["variant"]가 샘플별 리스트/튜플/텐서/단일 문자열/None 등
    어떤 형태로 와도 길이 B의 리스트[str]로 정규화.
    """
    if variant_field is None:
        return ["ctrl"] * B
    # torch 텐서면 CPU로 가져와 파이썬 객체로
    if torch.is_tensor(variant_field):
        try:
            variant_field = variant_field.cpu().tolist()
        except Exception:
            variant_field = [str(variant_field)]  # fallback
    # 단일 문자열이면 브로드캐스트
    if isinstance(variant_field, str):
        return [variant_field] * B
    # 리스트/튜플이면 각 원소를 문자열화
    if isinstance(variant_field, (list, tuple)):
        out = []
        for v in variant_field:
            if v is None:
                out.append("NA")
            elif isinstance(v, (list, tuple)):  # ["TP53~Y220C"] 같은 중첩 구조 방지
                out.append("|".join(map(str, v)))
            else:
                out.append(str(v))
        # 길이가 다르면 보정
        if len(out) != B:
            if len(out) == 1:
                out = out * B
            else:
                out = (out + ["NA"] * B)[:B]
        return out
    # 기타 타입은 문자열로 브로드캐스트
    return [str(variant_field)] * B

def build_config(cell_line, fold):
    return {
        "model": "scFoundation",
        "data_name": f"kim2023_{cell_line.lower()}_[benchmark][{fold}_3-fold]",
        "adata_name": "perturb_processed_metadata",
        "gex_layer": "counts",
        "embedding_model": "esm_msa1_t12_100M_UR50S",
        "variant_representation": "DIFF",
        "compression": "position_embedding",
        "mutation_type": "aminoMSA",
        "split": "exist",
        "epochs": 1,
        "batch_size": 2,
        "trainer": "Ban",
        "use_dummy_embedding": False,
        "visible_emb": False,
    }