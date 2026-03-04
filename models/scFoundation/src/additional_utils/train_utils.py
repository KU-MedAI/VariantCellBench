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
