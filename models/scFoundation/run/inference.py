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




# ============================================================
# 1. Global settings
# ============================================================

CELL_LINES = ["HCT116","U2OS"]
FOLDS = ["1", "2", "3"]
DEVICE = "cuda"
PERT_IDX_VALUE = 2501

BASE_LOG_DIR = "/NFS_DATA/samsung/variant-scFoundation/log"
BASE_DATA_DIR = "/NFS_DATA/samsung/database/gears"

OUT_DIR = "/NFS_DATA/samsung/database/benchmark_figure/ann_dataset_oncoKB/scFoundation"
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 2. Checkpoint map
# ============================================================

CHECKPOINT_MAP = {
    "HCT116": {
        "1": 'scFoundation_+_kim2023_hct116_[benchmark][1_3-fold]_+_esm_msa1_t12_100M_UR50S_+_DIFF_+_position_embedding_+_20260202_1121',
        "2": "scFoundation_+_kim2023_hct116_[benchmark][2_3-fold]_+_esm_msa1_t12_100M_UR50S_+_DIFF_+_position_embedding_+_20260205_2303",
        "3": 'scFoundation_+_kim2023_hct116_[benchmark][3_3-fold]_+_esm_msa1_t12_100M_UR50S_+_DIFF_+_position_embedding_+_20260204_0819',
    },
    "U2OS": {
        "1": 'scFoundation_+_kim2023_u2os_[benchmark][1_3-fold]_+_esm_msa1_t12_100M_UR50S_+_DIFF_+_position_embedding_+_20260202_0352',
        "2": 'scFoundation_+_kim2023_u2os_[benchmark][2_3-fold]_+_esm_msa1_t12_100M_UR50S_+_DIFF_+_position_embedding_+_20260204_1441',
        "3": 'scFoundation_+_kim2023_u2os_[benchmark][3_3-fold]_+_esm_msa1_t12_100M_UR50S_+_DIFF_+_position_embedding_+_20260129_2358',
    },
}



# ============================================================
# 4. Main automation loop
# ============================================================

for cell_line in CELL_LINES:
    for fold in FOLDS:
        print(f"\n🚀 scFoundation | {cell_line} | fold {fold}")

        # ====================================================
        # config
        # ====================================================
        config = build_config(cell_line, fold)

        # 🔥🔥🔥 [추가] variantseq 반드시 생성
        variantseq = CustomConditionData(GEARS_DATA_PATH)
        variantseq.load(config)


        checkpoint_dir = os.path.join(
            BASE_LOG_DIR,
            CHECKPOINT_MAP[cell_line][fold]
        )

        # ---------- load model config ----------
        with open(f"{checkpoint_dir}/model_config.json") as f:
            model_config = json.load(f)

        # depth 줄이기 (optional)
        model_config["encoder"]["depth"] = 6
        model_config["decoder"]["depth"] = 3

        # ---------- load AnnData (gene order reference) ----------
        adata_path = (
            f"{BASE_DATA_DIR}/"
            f"kim2023_{cell_line.lower()}_[benchmark][oncoKB]/"
            f"perturb_processed_[01].h5ad"
        )
        adata = ad.read_h5ad(adata_path)
        data_genes = adata.var_names.tolist()

        # ---------- model gene index ----------
        gene_index_tsv = "./OS_scRNA_gene_index.19264.tsv"
        sc_genes = pd.read_csv(gene_index_tsv, sep="\t")["gene_name"].tolist()

        gom = build_gene_order_map(data_genes, sc_genes, device="cpu")

        # ---------- embedding cache ----------
        emb_model = "esm_msa1_t12_100M_UR50S"
        cache_path = (
            f"/NFS_DATA/samsung/database/gears/embedding/"
            f"embedding_cache_variant_position_[{emb_model}].pkl"
        )
        embedding_cache = load_embedding_cache(cache_path)

        seq_emb_hidden_dim = len(
            random.choice(list(random.choice(list(embedding_cache.values())).values()))
        )

        # ---------- build model ----------
        model = MaeAutobin(
            num_tokens=model_config["num_tokens"],
            max_seq_len=model_config["seq_len"],
            embed_dim=model_config["encoder"]["hidden_dim"],
            decoder_embed_dim=model_config["decoder"]["hidden_dim"],
            bin_num=model_config["bin_num"],
            bin_alpha=model_config["bin_alpha"],
            pad_token_id=model_config["pad_token_id"],
            mask_token_id=model_config["mask_token_id"],
            cond_in_dim=seq_emb_hidden_dim,
            gene_index_tsv=gene_index_tsv,
            embedding_cache=embedding_cache,
            model_config=model_config,
            config=build_config(cell_line, fold),
        ).to(DEVICE)

        ckpt = torch.load(f"{checkpoint_dir}/best.pt", map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        model.eval()

        # ---------- dataloader directory ---------- #
        dl_dir = (
            f"{BASE_DATA_DIR}/"
            f"kim2023_{cell_line.lower()}_[benchmark][oncoKB]/"
            f"dataloader"
        )

        dl_files = sorted(
            os.path.join(dl_dir, f)
            for f in os.listdir(dl_dir)
            if f.endswith(".pkl")
        )

        # ====================================================
        # dataloader loop
        # ====================================================
        for dl_path in dl_files:
            dl_name = os.path.basename(dl_path).replace(".pkl", "")
            print(f"🧪 {dl_name}")

            with open(dl_path, "rb") as f:
                dataset_processed = pickle.load(f)

            aligned_dataset = AlignToDataSchemaDataset(
                dataset_processed["test_loader"].dataset,
                pert_idx_value=PERT_IDX_VALUE,
            )

            aligned_loader = DataLoader(
                aligned_dataset,
                batch_size=3,
                shuffle=False
            )

            test_ds = ReorderedVariantSeqDataset(
                aligned_loader.dataset,
                gom,
                fill_value=0.0,
                clone=True,
            )

            # ---------- inference ----------
            pred_rows, obs_records = [], []

            for batch in test_ds:
                batch_ = build_finetune_batch(
                    batch.x,
                    batch.y,
                    model_config["pad_token_id"],
                    model_config["seq_len"],
                )

                if isinstance(batch, dict) and "variant" in batch:
                    batch_["variant"] = batch["variant"]
                elif hasattr(batch, "variant"):
                    batch_["variant"] = batch.variant

                for k, v in batch_.items():
                    if torch.is_tensor(v):
                        batch_[k] = v.to(DEVICE, non_blocking=True)

                with torch.inference_mode():
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
                        data=variantseq,   # ✅ 이제 정상
                    )


                pred = to_data_order(y_pred, gom, "BG", "BG").detach().cpu().float().numpy()
                B = pred.shape[0]
                pred_rows.append(pred)
                var_labels = _normalize_variant_labels(batch_.get("variant", None), B)
                for i in range(B):
                    obs_records.append({
                        "condition": var_labels[i],
                    })

            X = np.vstack(pred_rows)
            genes = list(gom.data_genes)

            adata_pred = ad.AnnData(X=X)
            adata_pred.var_names = genes
            adata_pred.obs["condition"] = pd.DataFrame(obs_records)["condition"]
            # adata_pred.obs["dose_val"] = adata_pred.obs.condition.apply(
            #     lambda x: "+".join(["1"] * len(x.split("+")))
            # )
            # adata_pred.obs["cell_type"] = cell_line
            # adata_pred.obs["condition_name"] = adata_pred.obs.apply(
            #     lambda x: "_".join([x.cell_type, x.condition, x.dose_val]),
            #     axis=1,
            # )

            # ---------- save ----------
            ts = datetime.now().strftime("%m%d_%H%M")
            out_name = f"{ts}_{cell_line}_scFoundation_{fold}-3_{dl_name}.h5ad"
            # ---- obs string sanitization (필수) ----
            for col in adata_pred.obs.columns:
                if adata_pred.obs[col].dtype == "object":
                    adata_pred.obs[col] = adata_pred.obs[col].astype(str)



            adata_pred.write_h5ad(os.path.join(OUT_DIR, out_name))

            print(f"✅ Saved: {out_name}")