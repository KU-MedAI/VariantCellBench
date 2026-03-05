# ============================================================
# 0. Imports
# ============================================================

import os
import sys
import json
import pickle
import random
from datetime import datetime
from copy import deepcopy

import torch
import numpy as np
import pandas as pd
import anndata as ad

from torch_geometric.loader import DataLoader

# ------------------------------------------------------------
# Project root path
# ------------------------------------------------------------
PROJECT_ROOT = "/home/tech/variantseq/seunghun"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

SRC_PATH = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)


# ------------------------------------------------------------
# Project modules
# ------------------------------------------------------------
from models.GEARS.src.config import *
from models.GEARS.src.data import *
from models.GEARS.src.emb_data import *
from models.GEARS.src.network import *
from models.GEARS.src.variant import *
from models.GEARS.src.utils import *


# ------------------------------------------------------------
# scFoundation additional utils
# ------------------------------------------------------------
from additional_utils.model import *
from additional_utils.model_utils import *
from additional_utils.gene_order import *
from additional_utils.scfm_utils import *
from pretrainmodels.load import load_model_frommmf

config = {
    'model':'scFoundation',
    'data_name':'kim2023_hct116_[benchmark][1_3-fold]',
    'adata_name':'perturb_processed_metadata',
    'gex_layer':'counts',
    'embedding_model': 'esm2_t33_650M_UR50D', # 'esm2_t33_650M_UR50D' / 'esm_msa1_t12_100M_UR50S' / 'ProtT5'
    'variant_representation': 'ALT',          # 'ALT' / 'DIFF'
    'compression':'position_embedding',       # 'full_sequence_average' / 'position_embedding'
    
    'split':'exist',
    'data_seed': 1,
    'epochs': 1,
    'batch_size': 8,
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



# Load data
variantseq = CustomConditionData(GEARS_DATA_PATH)
variantseq.load(config)

# Gene order in my AnnData object
data_genes = variantseq.adata.var_names.tolist()

# Gene order used during model pretraining (loaded from checkpoint/config)
scfoundation_gene_list = pd.read_csv('/home/tech/variantseq/foundation/scFoundation/model/OS_scRNA_gene_index.19264.tsv', header=0, delimiter='\t')
model_genes = scfoundation_gene_list['gene_name'].to_list()  # 예: List[str]

# Create GeneOrderMap object
gom = build_gene_order_map(data_genes, model_genes, device='cpu')






# ================================
# 1) Check whether cached datasets exist
# ================================
root_dir = f"/NFS_DATA/samsung/database/scFoundation/{config['data_name']}"
os.makedirs(root_dir, exist_ok=True)

train_cache = os.path.join(root_dir, "train.pt")
valid_cache = os.path.join(root_dir, "valid.pt")
test_cache  = os.path.join(root_dir, "test.pt")

if all(os.path.exists(x) for x in [train_cache, valid_cache, test_cache]):
    print("[INFO] Found existing cached datasets. Loading…")

    train_ds = load_dataset(train_cache)
    valid_ds = load_dataset(valid_cache)
    test_ds  = load_dataset(test_cache)

else:
    print("[INFO] No cached dataset. Creating new datasets…")

    # original dataset
    base_train = variantseq.dataloader['train_loader'].dataset
    base_valid = variantseq.dataloader['val_loader'].dataset
    base_test  = variantseq.dataloader['test_loader'].dataset

    # Reordered dataset
    train_ds = ReorderedVariantSeqDataset(base_train, gom, fill_value=0.0, clone=True)
    valid_ds = ReorderedVariantSeqDataset(base_valid, gom, fill_value=0.0, clone=True)
    test_ds  = ReorderedVariantSeqDataset(base_test,  gom, fill_value=0.0, clone=True)

    # save dataset
    save_dataset(train_cache, train_ds)
    save_dataset(valid_cache, valid_ds)
    save_dataset(test_cache, test_ds)


# ================================
# 2) Build DataLoader
# ================================
from torch_geometric.loader import DataLoader

train_loader = DataLoader(
    train_ds,
    batch_size=4,
    shuffle=False,
    num_workers=20,
    pin_memory=False,
    persistent_workers=False,
)

valid_loader = DataLoader(
    valid_ds,
    batch_size=4,
    shuffle=False,
    num_workers=20,
    pin_memory=False,
    persistent_workers=False,
)

test_loader = DataLoader(
    test_ds,
    batch_size=4,
    shuffle=False,
    num_workers=20,
    pin_memory=False,
    persistent_workers=False,
)

print("[DONE] Dataset load ready.")



from load import load_model_frommmf
ckpt_path = '/NFS_DATA/samsung/foundation/scFoundation/models.ckpt'
model, model_config = load_model_frommmf(ckpt_path)

# Reduce model depth
model_config["encoder"]["depth"] = 6   # original depth: 12
model_config["decoder"]["depth"] = 3   # original depth: 6

print("encoder depth:", model_config["encoder"]["depth"])
print("decoder depth:", model_config["decoder"]["depth"])



model = MaeAutobin(
    num_tokens=model_config["num_tokens"],
    max_seq_len=model_config["seq_len"],
    embed_dim=model_config["encoder"]["hidden_dim"],
    decoder_embed_dim=model_config["decoder"]["hidden_dim"],
    bin_num=model_config["bin_num"],
    bin_alpha=model_config["bin_alpha"],
    pad_token_id=model_config["pad_token_id"],
    mask_token_id=model_config["mask_token_id"],
    cond_in_dim = infer_cond_in_dim(variantseq.embedding_cache),
    gene_index_tsv="/home/tech/variantseq/foundation/scFoundation/model/OS_scRNA_gene_index.19264.tsv",
    embedding_cache=variantseq.embedding_cache,
    model_config = model_config,
    config = config
)

model_genes = scfoundation_gene_list['gene_name'].to_list()
load_branch_autointersect(model, ckpt_path, branch="gene", device=device)



from datetime import datetime
seed_all(42)

timestamp = get_timestamp()  # 기존 함수 사용
project_name = make_project_name(config, timestamp)  # ← 따옴표 버그 해결/위생 처리
OUTPUT_DIR = "/NFS_DATA/samsung/variant-scFoundation/log"
checkpoint_dir = os.path.join(OUTPUT_DIR, project_name)
os.makedirs(checkpoint_dir, exist_ok=True)

with open(f"{checkpoint_dir}/config.pkl", "wb") as f:
    import pickle; pickle.dump(config, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(f"{checkpoint_dir}/config.json", "w") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
with open(f"{checkpoint_dir}/model_config.json", "w") as f:
    json.dump(model_config, f, indent=2, ensure_ascii=False)


# ----------------------- #
# Early stopping / best checkpoint
# ----------------------- #
best_val = float("inf")
# patience, bad_epochs = 5, 0
last_ckpt = os.path.join(checkpoint_dir, "last.pt")
best_ckpt = os.path.join(checkpoint_dir, "best.pt")
global_step = 0
stop_training = False

# Save last checkpoint on safe termination
signal.signal(signal.SIGTERM, _handle_sigterm)

from torch import amp
import tqdm as tqdm

optimizer = build_optimizer(model, lr_head=1e-3, lr_enc=1e-4, wd=0.01)
scaler = amp.GradScaler(device="cuda", enabled=True)
epochs = config['epochs']
LOG_EVERY = 50


from torch.amp import autocast as amp_autocast   # PyTorch 2.x 권장
from torch.cuda.amp import GradScaler
import tqdm as tqdm
import torch

# ---- Hyperparameters / flags ----
max_grad_norm      = 1.0
check_grad_finite  = False
use_amp            = (torch.cuda.is_available())  # CUDA면 AMP 사용
device             = "cuda:0" if torch.cuda.is_available() else "cpu"

# ---- Safe initialization ----
global_step   = 0
best_val      = float("inf")
# bad_epochs    = 0
stop_training = False
epochs        = config["epochs"]
project_name  = project_name
scaler        = amp.GradScaler(device="cuda", enabled=True)



try:
    for epoch in range(1, epochs + 1):
        try:
            slack_msg(f"[Epoch {epoch}/{config['epochs']}] {project_name}")
        except:
            pass
        model.train()

        train_loss_sum = 0.0
        valid_loss_sum = 0.0
        num_steps      = len(train_loader)

        pbar = tqdm.tqdm(
            enumerate(train_loader, 1),
            total=num_steps,
            dynamic_ncols=True
        )

        for step, batch in pbar:
            # === Prepare batch ===
            batch_ = build_finetune_batch(
                x=batch.x, y=batch.y,
                pad_token_id=model_config["pad_token_id"],
                seq_len=model_config["seq_len"]
            )

            # Safely pass variant information from PyG Data or dict
            if isinstance(batch, dict) and "variant" in batch:
                batch_["variant"] = batch["variant"]
            elif hasattr(batch, "variant"):
                batch_["variant"] = batch.variant

            # Move tensors to device
            for k, v in list(batch_.items()):
                if torch.is_tensor(v):
                    batch_[k] = v.to(device, non_blocking=True)

            # === Forward & Loss ===
            optimizer.zero_grad(set_to_none=True)
            with amp_autocast(device_type=device, enabled=use_amp, dtype=torch.bfloat16 if use_amp else None):
                y_pred = model(
                    x=batch_["x"],
                    padding_label=batch_["padding_label"],
                    encoder_position_gene_ids=batch_["encoder_pos_ids"],
                    encoder_labels=batch_["encoder_labels"],
                    decoder_data=batch_["decoder_data"],
                    mask_gene_name=batch_["mask_gene_name"],
                    mask_labels=batch_["mask_labels"],
                    decoder_position_gene_ids=batch_["decoder_pos_ids"],
                    decoder_data_padding_labels=batch_["decoder_pad_labels"],
                    variant=batch_.get("variant", None),
                    data=variantseq
                )

                # Compute MSE only on observed positions
                observed = ~batch_["decoder_pad_labels"]
                loss = torch.nn.functional.mse_loss(y_pred[observed], batch_["target"][observed])

            # === Backward & Step ===
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # clip 전에 필수

                if check_grad_finite:
                    for p in model.parameters():
                        if p.grad is not None and not torch.isfinite(p.grad).all():
                            raise RuntimeError("NaN/Inf gradient detected under AMP.")

                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if check_grad_finite:
                    for p in model.parameters():
                        if p.grad is not None and not torch.isfinite(p.grad).all():
                            raise RuntimeError("NaN/Inf gradient detected.")
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=max_grad_norm
                )
                optimizer.step()

            train_loss_sum += loss.item()
            global_step    += 1

            if (global_step % LOG_EVERY == 0) or (step == 1):
                pbar.set_description(f"[Epoch {epoch}/{epochs}] step {step} loss={loss.item():.4f}")
            # break

        # === validation ===
        avg_train_loss = train_loss_sum / max(1, num_steps)

        model.eval()
        with torch.no_grad():
            tot_loss, tot_obs, n_batches = 0.0, 0, 0
            for batch in valid_loader:
                num_steps = len(valid_loader)
                batch_ = build_finetune_batch(batch.x, batch.y, model_config["pad_token_id"], model_config["seq_len"])
                if isinstance(batch, dict) and "variant" in batch:
                    batch_["variant"] = batch["variant"]
                elif hasattr(batch, "variant"):
                    batch_["variant"] = batch.variant
                
                # Move tensors to device
                for k, v in list(batch_.items()):
                    if torch.is_tensor(v):
                        batch_[k] = v.to(device, non_blocking=True)

                with torch.inference_mode():
                    ctx = amp_autocast(device_type="cuda", enabled=(use_amp and str(device).startswith("cuda")), dtype=torch.bfloat16)
                    with ctx:
                        y_pred = model(
                            x=batch_["x"],
                            padding_label=batch_["padding_label"],
                            encoder_position_gene_ids=batch_["encoder_pos_ids"],
                            encoder_labels=batch_["encoder_labels"],
                            decoder_data=batch_["decoder_data"],
                            mask_gene_name=batch_["mask_gene_name"],
                            mask_labels=batch_["mask_labels"],
                            decoder_position_gene_ids=batch_["decoder_pos_ids"],
                            decoder_data_padding_labels=batch_["decoder_pad_labels"],
                            variant=batch_.get("variant", None),
                            data=variantseq
                        )

                observed = ~batch_["decoder_pad_labels"]
                loss = torch.nn.functional.mse_loss(y_pred[observed], batch_["target"][observed])

                valid_loss_sum += loss.item()
                global_step    += 1

            avg_valid_loss = valid_loss_sum / max(1, num_steps)

        try:
            slack_msg(f"[Epoch {epoch}/{epochs}] train_loss={avg_train_loss:.4f} val_loss={avg_valid_loss:.4f}")
        except:
            pass

        # log_dict = {
        #     "epoch": epoch,
        #     "train_loss": avg_train_loss,
        #     "val_loss": avg_valid_loss,
        # }

        # === Checkpoint & Early Stopping ===
        save_checkpoint(last_ckpt, model, optimizer, epoch, global_step, best_val)

        if avg_valid_loss < best_val:
            best_val = avg_valid_loss
            save_checkpoint(best_ckpt, model, optimizer, epoch, global_step, best_val)
            best_state_dict = {
                k: v.cpu() for k, v in model.state_dict().items()
            }


except KeyboardInterrupt:
    print("[INTERRUPTED] Saving last checkpoint…")
    save_checkpoint(last_ckpt, model, optimizer, epoch, global_step, best_val)