# ============================================================
# 0. Imports
# ============================================================
import os
import json
import numpy as np

import torch
import anndata as ann

# project modules
from config import *
from data import *
from emb_data import *
from network import *
from result import *
from variant import *
from utils import *
from bsh import *


# ============================================================
# 1. Load experiment configuration
# ============================================================

checkpoint_dir = ""   # Directory containing trained model checkpoints

# Load configuration used during training
with open(f"{checkpoint_dir}/config.json", "r") as f:
    config = json.load(f)


# ============================================================
# 2. Initialize dataset and dataloaders
# ============================================================

# Initialize custom dataset object
Data = CustomConditionData(GEARS_DATA_PATH)

# Load dataset using the stored configuration
Data.load(config)


# ============================================================
# 3. Initialize model
# ============================================================

# Create GEARS model instance
model = GEARS_2(Data, config=config).to(DEVICE)


# ============================================================
# 4. Load trained model checkpoint
# ============================================================

checkpoint_path = f"{checkpoint_dir}/best.pt"

# Load checkpoint (map to CPU/GPU automatically)
checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

# Restore trained model parameters
model.load_state_dict(checkpoint["model_state_dict"])


# ============================================================
# 5. Evaluate model on the test dataset
# ============================================================

# Test dataloader
test_loader = model.dataloader.dataloader['test_loader']

# Run inference
test_res = evaluate(
    test_loader,
    model,
    model.uncertainty,
    model.device
)

# Compute evaluation metrics
test_metrics, test_pert_res = compute_metrics(test_res)


# ============================================================
# 6. Extract control expression profiles
# ============================================================

# Access dataset from test loader
ds = test_loader.dataset

# Extract input expression vectors from each sample
x_list = [data.x.squeeze() for data in ds]

# Stack tensors and convert to numpy array
ctrl = torch.stack(x_list, dim=0).numpy()



# ============================================================
# 7. Create AnnData object for predicted expression
# ============================================================
pred_adata = ann.AnnData(X=np.concatenate((test_res['pred'],ctrl)))
pred_adata.obs_names = np.concatenate((test_res['pert_cat'],np.array(['ctrl']*ctrl.shape[0])))
pred_adata.obs['condition'] = np.concatenate((test_res['pert_cat'],np.array(['ctrl']*ctrl.shape[0])))
pred_adata.obs.loc[:, 'dose_val'] = pred_adata.obs.condition.apply(lambda x: '+'.join(['1'] * len(x.split('+'))))
pred_adata.obs.loc[:, 'cell_type'] = model.dataloader.adata.obs.cell_type.unique()[0]
pred_adata.obs.loc[:, 'condition_name'] =  pred_adata.obs.apply(lambda x: '_'.join([x.cell_type, x.condition, x.dose_val]), axis = 1)
pred_adata.var_names = model.dataloader.adata.var.index
pred_adata.var['gene_name'] = model.dataloader.adata.var['gene_name']
pred_adata.uns = model.dataloader.adata.uns
pred_adata.write_h5ad(os.path.join(checkpoint_dir, "pred_adata.h5ad"))


# ============================================================
# 8. Create AnnData object for ground truth expression
# ============================================================
truth_adata = ann.AnnData(X=np.concatenate((test_res['truth'],ctrl)))
truth_adata.obs_names = np.concatenate((test_res['pert_cat'],np.array(['ctrl']*ctrl.shape[0])))
truth_adata.obs['condition'] = np.concatenate((test_res['pert_cat'],np.array(['ctrl']*ctrl.shape[0])))
truth_adata.obs.loc[:, 'dose_val'] = truth_adata.obs.condition.apply(lambda x: '+'.join(['1'] * len(x.split('+'))))
truth_adata.obs.loc[:, 'cell_type'] = model.dataloader.adata.obs.cell_type.unique()[0]
truth_adata.obs.loc[:, 'condition_name'] =  truth_adata.obs.apply(lambda x: '_'.join([x.cell_type, x.condition, x.dose_val]), axis = 1)
truth_adata.var_names = model.dataloader.adata.var.index
truth_adata.var['gene_name'] = model.dataloader.adata.var['gene_name']
truth_adata.uns = model.dataloader.adata.uns
truth_adata.write_h5ad(os.path.join(checkpoint_dir, "truth_adata.h5ad"))