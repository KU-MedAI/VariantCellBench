# ============================================================
# 0. Imports
# ============================================================
import os
import glob
import pickle
from datetime import datetime
from copy import deepcopy

import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

import anndata as ad

# project modules
from config import *
from data import *
from emb_data import *
from network import *
from result import *
from variant import *
from utils import *

# ============================================================
# 1. Automation settings
# ============================================================

CELL_LINES = ["HCT116","U2OS"] # "HCT116", 
FOLDS = ["1", "2", "3"]

PERT_IDX_VALUE = 2501
DEVICE = "cuda"

BASE_DATALOADER_DIR = "/NFS_DATA/samsung/database/gears"
BASE_CHECKPOINT_DIR = "/NFS_DATA/samsung/variantGEARS/log"

OUT_DIR = "/NFS_DATA/samsung/database/benchmark_figure/ann_dataset_oncoKB/GEARS"
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 2. Checkpoint map
# ============================================================

# CHECKPOINT_MAP = {
#     "HCT116": {
#         "1": "GEARS_+_kim2023_hct116_[benchmark][1_3-fold]_+_esm_msa1_t12_100M_UR50S_+_DIFF_+_position_embedding_+_20251127_2037",
#         "2": "GEARS_+_kim2023_hct116_[benchmark][2_3-fold]_+_esm_msa1_t12_100M_UR50S_+_DIFF_+_position_embedding_+_20251127_2239",
#         "3": "GEARS_+_kim2023_hct116_[benchmark][3_3-fold]_+_esm_msa1_t12_100M_UR50S_+_DIFF_+_position_embedding_+_20251128_0206",
#     },
#     "U2OS": {
#         "1": "GEARS_+_kim2023_u2os_[benchmark][1_3-fold]_+_esm_msa1_t12_100M_UR50S_+_DIFF_+_position_embedding_+_20251128_0617",
#         "2": "GEARS_+_kim2023_u2os_[benchmark][2_3-fold]_+_esm_msa1_t12_100M_UR50S_+_DIFF_+_position_embedding_+_20251128_1051",
#         "3": "GEARS_+_kim2023_u2os_[benchmark][3_3-fold]_+_esm_msa1_t12_100M_UR50S_+_DIFF_+_position_embedding_+_20251128_2120",
#     },
# }

CHECKPOINT_MAP = {
    "HCT116": {
        "1": "GEARS_+_kim2023_hct116_[benchmark][1_3-fold]_+_esm_msa1_t12_100M_UR50S_+_DIFF_+_position_embedding_+_20260128_2330",
        "2": "GEARS_+_kim2023_hct116_[benchmark][2_3-fold]_+_esm_msa1_t12_100M_UR50S_+_DIFF_+_position_embedding_+_20260129_0439",
        "3": "GEARS_+_kim2023_hct116_[benchmark][3_3-fold]_+_esm_msa1_t12_100M_UR50S_+_DIFF_+_position_embedding_+_20260129_0941",
    },
    "U2OS": {
        "1": "GEARS_+_kim2023_u2os_[benchmark][1_3-fold]_+_esm_msa1_t12_100M_UR50S_+_DIFF_+_position_embedding_+_20260129_1528",
        "2": "GEARS_+_kim2023_u2os_[benchmark][2_3-fold]_+_esm_msa1_t12_100M_UR50S_+_DIFF_+_position_embedding_+_20260129_1959",
        "3": "GEARS_+_kim2023_u2os_[benchmark][3_3-fold]_+_esm_msa1_t12_100M_UR50S_+_DIFF_+_position_embedding_+_20260130_0015",
    },
}

# ============================================================
# 3. Config builder
# ============================================================

def build_config(cell_line, fold):
    ckpt_name = CHECKPOINT_MAP[cell_line][fold]

    if "esm_msa1" in ckpt_name:
        embedding_model = "esm_msa1_t12_100M_UR50S"
    elif "esm2_t33" in ckpt_name:
        embedding_model = "esm2_t33_650M_UR50D"
    else:
        raise ValueError(f"Unknown embedding model in {ckpt_name}")

    return {
        "model": "GEARS",
        "data_name": f"kim2023_{cell_line.lower()}_[benchmark][{fold}_3-fold]",
        "adata_name": "perturb_processed_metadata",
        "split": "exist",
        "embedding_model": embedding_model,
        "mutation_type": "aminoMSA",
        "gears_mode": "variant",
        "embedding_merge_method": "no_pert",
        "variant_representation": "DIFF",
        "compression": "position_embedding",
        "pert_graph": False,
        "epochs": 20,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "weight_decay": 5e-4,
        "uncertainty": False,
        "n_gene_layer": 1,
        "n_condition_layer": 1,
        "compress_by": "hidden-wise",
        "direction_method": "tanh",
        "direction_lambda": 5,
        "direction_alpha": 0.5,
        "loss_version": 1,
        "top_n_hvg": 100,
        "hvg_weight": 0,
        "trainer": "Ban",
        "use_dummy_embedding": False,
        "visible_emb": False,
    }


# ============================================================
# 4. Adapter Dataset
# ============================================================

class AlignToDataSchemaDataset(Dataset):
    def __init__(self, base_dataset, pert_idx_value=2501):
        self.base = base_dataset
        self.pert_idx_value = pert_idx_value

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        data = deepcopy(self.base[idx])

        data.pert_idx = [self.pert_idx_value]

        if isinstance(data.variant, str):
            data.variant = [data.variant]

        if hasattr(data, "pert_"):
            delattr(data, "pert_")

        return data


# ============================================================
# 5. Main automation loop
# ============================================================

for cell_line in CELL_LINES:
    for fold in FOLDS:
        print(f"\n🚀 Inference start: {cell_line} | fold {fold}")

        # ---------- dataloader files ----------
        dataloader_dir = (
            f"{BASE_DATALOADER_DIR}/"
            f"kim2023_{cell_line.lower()}_[benchmark][oncoKB]/"
            f"dataloader"
        )

        dataloader_files = [
            os.path.join(dataloader_dir, f)
            for f in os.listdir(dataloader_dir)
            if f.endswith(".pkl")
        ]

        assert len(dataloader_files) > 0, f"No pkl files in {dataloader_dir}"

        print(f"[INFO] Found {len(dataloader_files)} dataloaders")

        # ---------- load Data object ----------
        DataObj = CustomConditionData(GEARS_DATA_PATH)
        config = build_config(cell_line, fold)
        DataObj.load(config)
        var_save = DataObj.adata.var

        # ---------- load model ----------
        checkpoint_dir = (
            f"{BASE_CHECKPOINT_DIR}/"
            f"{CHECKPOINT_MAP[cell_line][fold]}"
        )

        model = GEARS_2(DataObj, config=config).to(DEVICE)
        ckpt = torch.load(
            os.path.join(checkpoint_dir, "best.pt"),
            map_location=DEVICE,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        print(
            f"[INFO] {cell_line} fold {fold} | "
            f"embedding_model = {config['embedding_model']}"
        )

        # ---------- dataloader loop ----------
        for dl_path in dataloader_files:
            dl_name = os.path.basename(dl_path).replace(".pkl", "")
            print(f"\n🧪 Processing dataloader: {dl_name}")

            with open(dl_path, "rb") as f:
                dataset_processed = pickle.load(f)

            aligned_dataset = AlignToDataSchemaDataset(
                dataset_processed["test_loader"].dataset,
                pert_idx_value=PERT_IDX_VALUE,
            )

            aligned_loader = DataLoader(
                aligned_dataset,
                batch_size=dataset_processed["test_loader"].batch_size,
                shuffle=False,
            )

            with torch.no_grad():
                test_res = evaluate(
                    aligned_loader,
                    model,
                    model.uncertainty,
                    model.device,
                )

            # ---------- build AnnData ----------
            pred = test_res["pred"]
            pert_cat = test_res["pert_cat"]

            pred_adata = ad.AnnData(X=pred)
            pred_adata.obs_names = [
                f"{p}_{i}" for i, p in enumerate(pert_cat)
            ]
            pred_adata.obs["condition"] = pert_cat
            pred_adata.obs["dose_val"] = pred_adata.obs.condition.apply(
                lambda x: "+".join(["1"] * len(x.split("+")))
            )
            pred_adata.obs["cell_type"] = cell_line
            pred_adata.obs["condition_name"] = pred_adata.obs.apply(
                lambda x: "_".join([x.cell_type, x.condition, x.dose_val]),
                axis=1,
            )
            pred_adata.var = var_save

            # ---------- save ----------
            timestamp = datetime.now().strftime("%m%d_%H%M")
            out_name = (
                f"{timestamp}_{cell_line}_msa_diff_{fold}-3_"
                f"{dl_name}.h5ad"
            )
            out_path = os.path.join(OUT_DIR, out_name)

            pred_adata.write_h5ad(out_path)
            print(f"✅ Saved: {out_path}")
