import os
import torch 
import torch.nn as nn
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# torch.set_num_threads(8)           
# torch.set_num_interop_threads(2)

from zipfile import ZipFile
from collections import defaultdict
import numpy as np
import pandas as pd
import networkx as nx
import scanpy as sc
import pickle

from torch_geometric.data import Data
from torch_geometric.data import DataLoader

import zarr
import numpy as np
import requests
import mygene
import json
import torch
import sys

# Add the ../src directory (located in the parent directory) to the Python path
sys.path.append(os.path.abspath("../src"))

# Import project modules from src
from config import *
from data import *
from emb_data import *
from network import *
from result import *
from variant import *
from utils import *


config = {
    'model':'GEARS',                                      # Name of the model being trained
    'data_name':'kim2023_hct116_[benchmark][1_3-fold]',   # Name of the training dataset
    'adata_name':'perturb_processed_metadata',            # Name of the data file
    'split':'exist',                                      # Data split condition. Use 'exist' if using pre-generated splits
    'embedding_model': 'esm2_t33_650M_UR50D',             # 'esm2_t33_650M_UR50D' / 'esm_msa1_t12_100M_UR50S' / 'ProtT5-XXL-U50' / 'Ankh3-Large' / 'xTrimoPGLM-10B-MLM'
    'mutation_type': 'amino',                             # Not critical; mutation type specification
    'gears_mode':'variant',                               # 'variant' / 'gene'
    'embedding_merge_method': 'no_pert',                  # Method to merge embeddings: 'cat', 'no_pert', 'element', or 'bilinear'
    'variant_representation': 'ALT',                      # Variant representation type: 'ALT' or 'DIFF'
    'compression': 'position_embedding',                  # Embedding compression method: 'position_embedding' or 'full_sequence_average'
    'pert_graph' : False,                                 # Apply perturbation graph GNN if True {default: False}
    'epochs': 20,                                         # Number of training epochs
    'batch_size': 32,                                     # Batch size
    # 'hidden_dim': 64,                                   # Model hidden dimension size
    'learning_rate': 1e-3,                                # Learning rate
    'weight_decay': 5e-4,                                 # Weight decay
    'uncertainty': False,                                 # Unused option; whether to apply uncertainty loss
    'n_gene_layer': 1,                                    # Number of gene GNN layers
    'n_condition_layer': 1,                               # Number of perturbation GNN layers
    'compress_by': 'hidden-wise',                         # Compression strategy: 'hidden-wise' or 'sequence-wise'

    'direction_method': 'tanh',                           # Direction calculation method: 'sign', 'tanh', or 'hybrid'
    # The following options are applied when loss_version != 1
    'direction_lambda': 5,                                #
    'direction_alpha': 0.5,                               #

    'loss_version': 1,                                    #
    'top_n_hvg': 100,                                     #
    'hvg_weight': 0,                                      #

    'trainer': 'Ban',                                     # Person responsible for model training
    'use_dummy_embedding': False,                         # Zero embedding for ablation experiments
    'visible_emb': False                                  # Whether to output embeddings (False / True)
}

# # Either you can set different configs
# # --- read override json if provided -- #
# import os, json
# _override_path = os.environ.get("CONFIG_OVERRIDE_PATH")
# if _override_path and os.path.exists(_override_path):
#     with open(_override_path) as f:
#         config.update(json.load(f))
# # ------------------------------------- #

Data = CustomConditionData(GEARS_DATA_PATH)
Data.load(config)



# Initialize dataset object using the GEARS data path
Data = CustomConditionData(GEARS_DATA_PATH)

# Load dataset according to the configuration settings
Data.load(config)


import os, re, json, time, signal, math, torch
from torch import nn
from torch.cuda import amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import tqdm


def seed_all(seed: int = 42):
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.
    Also enables performance optimizations such as TF32 and cuDNN benchmarking.
    """
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    # Enable faster GPU operations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


SAFE_DELIMS = "_+_"

def sanitize_name(s: str) -> str:
    """
    Sanitize a string for safe use as a directory or file name.

    - Remove characters forbidden in Windows file systems: <>:"/\\|?*
    - Replace spaces with '-'
    """
    s = re.sub(r'[<>:"/\\|?*]+', '', s)
    s = s.replace(' ', '-')
    return s


def make_project_name(config, timestamp):
    """
    Generate a unique project name based on key configuration parameters.

    The project name is composed of:
    model + dataset name + embedding model + variant representation
    + compression method + timestamp
    """
    parts = [
        str(config["model"]),
        str(config["data_name"]),
        str(config["embedding_model"]),
        str(config["variant_representation"]),
        str(config["compression"]),
        str(timestamp),
    ]
    return sanitize_name(SAFE_DELIMS.join(parts))


from datetime import datetime

# Set global random seed
seed_all(42)

# Generate timestamp for the current run
timestamp = get_timestamp()

# Create a unique project name for this experiment
project_name = make_project_name(config, timestamp)

# Create directory for saving checkpoints and outputs
checkpoint_dir = os.path.join(OUTPUT_DIR, project_name)
os.makedirs(checkpoint_dir, exist_ok=True)


import pickle

# Save configuration in binary format for exact reproducibility
with open(f"{checkpoint_dir}/config.pkl", "wb") as f:
    pickle.dump(config, f, protocol=pickle.HIGHEST_PROTOCOL)

# Save configuration in JSON format for readability
with open(f"{checkpoint_dir}/config.json", "w") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print(project_name)


device = 'cuda'
model = GEARS_2(Data, config=config).to(device)


update_visible = False
# model.dataloader.dataloader
# Data.adata


from datetime import datetime
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from tqdm import tqdm

# Initialize logging variables
loss_log = []
save_every = 5

# Directory for saving checkpoints and logs
checkpoint_dir = os.path.join(OUTPUT_DIR, project_name)
log_path = os.path.join(checkpoint_dir, f"loss_log_{timestamp}.csv")
os.makedirs(checkpoint_dir, exist_ok=True)

# Optimizer and learning-rate scheduler
optimizer = optim.Adam(
    model.parameters(),
    lr=config['learning_rate'],
    weight_decay=config['weight_decay']
)

# Reduce learning rate by half every epoch
scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

# Monitoring configuration for best model selection
monitor_metric = 'val_loss'
monitor_mode   = 'min'
min_delta      = float(config.get('min_delta', 0.0))

# Utility class that tracks the best checkpoint
best_saver = BestSaver(
    monitor=monitor_metric,
    mode=monitor_mode,
    min_delta=min_delta
)

# Training and validation dataloaders
train_loader = model.dataloader.dataloader['train_loader']
val_loader   = model.dataloader.dataloader['val_loader']


# ===============================
# Training loop
# ===============================
for epoch in tqdm(range(config['epochs'])):

    print(f"[Epoch {epoch+1}/{config['epochs']}] {project_name}")

    model.train()

    # Accumulators for epoch-level loss statistics
    epoch_loss = 0
    epoch_autofocus_loss = 0
    epoch_direction_loss = 0
    epoch_hvg_loss = 0
    epoch_hvg_autofocus_loss = 0
    epoch_hvg_direction_loss = 0

    visible = False
    num_steps = len(train_loader)

    # -------------------------------
    # Iterate over training batches
    # -------------------------------
    for step, batch in tqdm(enumerate(train_loader)):

        batch.to(model.device)

        # Optional: snapshot parameters to verify update behavior
        if update_visible:
            old_params = {
                name: param.clone().detach()
                for name, param in model.named_parameters()
                if param.requires_grad
            }

        optimizer.zero_grad()

        y = batch.y

        # ----------------------------------
        # Forward pass and loss computation
        # ----------------------------------
        visible = []
        pred = model(batch, visible)

        # Compute multi-component loss
        (
            loss,
            autofocus_loss,
            direction_loss,
            hvg_loss,
            autofocus_hvg_loss,
            direction_hvg_loss
        ) = loss_fct(
            pred,
            y,
            batch.pert,
            model.dataloader.hvg_n_idx,
            ctrl=model.ctrl_expression,
            dict_filter=model.dict_filter,
            loss_version=model.loss_version,
            direction_lambda=config['direction_lambda'],
            direction_alpha=config['direction_alpha'],
            direction_method=config['direction_method'],
            hvg_weight=config['hvg_weight'],
            visible=visible
        )

        # Backpropagation
        loss.backward()

        # Gradient clipping for stability
        nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)

        optimizer.step()

        # ----------------------------------
        # Optional: check parameter updates
        # ----------------------------------
        if update_visible:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if not torch.equal(old_params[name], param):
                        print(f"✅ Parameter '{name}' has been updated.")
                    else:
                        print(f"❌ Parameter '{name}' has NOT been updated.")

        # Accumulate losses
        epoch_loss += loss.item()
        epoch_autofocus_loss += float(autofocus_loss.detach().item())
        epoch_direction_loss += float(direction_loss.detach().item())
        epoch_hvg_loss += float(hvg_loss.detach().item())
        epoch_hvg_autofocus_loss += float(autofocus_hvg_loss.detach().item())
        epoch_hvg_direction_loss += float(direction_hvg_loss.detach().item())


    # ===============================
    # Epoch-level average losses
    # ===============================
    avg_loss = epoch_loss / max(1, num_steps)
    avg_autofocus_loss = epoch_autofocus_loss / max(1, num_steps)
    avg_direction_loss = epoch_direction_loss / max(1, num_steps)
    avg_hvg_loss = epoch_hvg_loss / max(1, num_steps)
    avg_hvg_autofocus_loss = epoch_hvg_autofocus_loss / max(1, num_steps)
    avg_hvg_direction_loss = epoch_hvg_direction_loss / max(1, num_steps)


    # ===============================
    # Evaluation phase
    # ===============================
    model.eval()

    with torch.no_grad():

        train_res = evaluate(train_loader, model, model.uncertainty, model.device)
        val_res   = evaluate(val_loader,   model, model.uncertainty, model.device)

    train_metrics, _ = compute_metrics(train_res)
    val_metrics, val_pert_res = compute_metrics(val_res)

    # Compute validation loss explicitly
    val_loss = compute_avg_loss_on_loader(
        model,
        val_loader,
        model.device,
        config,
        visible_cfg=[]
    )


    # ===============================
    # Additional biological metrics
    # ===============================
    train_out = deeper_analysis(Data.adata, train_res)
    val_out   = deeper_analysis(Data.adata, val_res)

    train_out_result = {}
    val_out_result   = {}

    # Aggregate perturbation-level metrics
    for m in list(list(train_out.values())[0].keys()):
        train_out_result['train_' + m] = np.mean(
            [j[m] for j in train_out.values() if m in j]
        )

    for m in list(list(val_out.values())[0].keys()):
        val_out_result['val_' + m] = np.mean(
            [j[m] for j in val_out.values() if m in j]
        )


    # ===============================
    # Logging dictionary
    # ===============================
    log_dict = {
        "epoch": epoch + 1,
        "avg_loss": avg_loss,
        "avg_autofocus_loss": avg_autofocus_loss,
        "avg_direction_loss": avg_direction_loss,
        "avg_hvg_loss": avg_hvg_loss,
        "avg_hvg_autofocus_loss": avg_hvg_autofocus_loss,
        "avg_hvg_direction_loss": avg_hvg_direction_loss,

        "train_mse": train_metrics['mse'],
        "val_mse": val_metrics['mse'],

        "train_mse_de": train_metrics['mse_de'],
        "val_mse_de": val_metrics['mse_de'],

        "train_pearson": train_metrics['pearson'],
        "val_pearson": val_metrics['pearson'],

        "train_pearson_de": train_metrics['pearson_de'],
        "val_pearson_de": val_metrics['pearson_de'],

        "val_loss": val_loss
    }


    # ===============================
    # Best checkpoint saving
    # ===============================
    if monitor_metric in val_metrics:
        current_value = float(
            val_metrics[monitor_metric.replace("val_", "")]
        )
    else:
        current_value = log_dict.get(monitor_metric, None)

        if current_value is None:
            raise ValueError(
                f"Monitor metric '{monitor_metric}' not found."
            )

    if best_saver.is_better(current_value):

        best_saver.best_value = current_value
        best_saver.best_epoch = epoch + 1

        best_path = os.path.join(checkpoint_dir, "best.pt")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": current_value
        }

        best_saver.best_path = best_path

        torch.save(checkpoint, best_path)

        print(
            f"🌟 New best {monitor_metric} = {current_value:.6f} "
            f"at epoch {epoch+1}. Saved to {best_path}"
        )

    scheduler.step()


    # ===============================
    # Additional evaluation metrics
    # ===============================

    # Direction correctness metrics
    for key in ['frac_correct_direction_20', 'frac_correct_direction_all']:
        log_dict[f"train_{key}"] = train_out_result[f"train_{key}"]
        log_dict[f"val_{key}"]   = val_out_result[f"val_{key}"]

    # HVG-based performance
    for top_k in [100, 250, 500, 1000]:
        for metric in ['mse', 'pearson', 'pearson_delta']:
            key = f"{metric}_top{top_k}_hvg"
            log_dict[f"train_{key}"] = train_out_result.get(f"train_{key}", None)
            log_dict[f"val_{key}"]   = val_out_result.get(f"val_{key}", None)

    # DE gene performance
    for top_k in [20, 50, 100, 200]:
        for metric in ['mse', 'pearson', 'pearson_delta']:
            key = f"{metric}_top{top_k}_de"
            log_dict[f"train_{key}"] = train_out_result.get(f"train_{key}", None)
            log_dict[f"val_{key}"]   = val_out_result.get(f"val_{key}", None)


    # ===============================
    # Training progress print
    # ===============================
    print(f"[{epoch+1}/{config['epochs']}] {project_name}")
    print("Train loss: {:.4f} | Valid loss: {:.4f}".format(avg_loss, val_loss))


# ===============================
# Save final model checkpoint
# ===============================
final_path = os.path.join(checkpoint_dir, "final_model.pt")

save_checkpoint(
    model,
    optimizer,
    config['epochs'],
    avg_loss,
    final_path
)