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






# AnnData에서 나의 유전자 순서
data_genes = variantseq.adata.var_names.tolist()

# 모델이 학습한 유전자 순서 (체크포인트/설정에서 로드)
scfoundation_gene_list = pd.read_csv('/home/tech/variantseq/foundation/scFoundation/model/OS_scRNA_gene_index.19264.tsv', header=0, delimiter='\t')
model_genes = scfoundation_gene_list['gene_name'].to_list()  # 예: List[str]

# GeneOrderMap 객체 생성. device = next(model.parameters()).device
gom = build_gene_order_map(data_genes, model_genes, device='cpu')


root_dir = f"/NFS_DATA/samsung/database/scFoundation/{config['data_name']}"

train_cache = os.path.join(root_dir, "train.pt")
valid_cache = os.path.join(root_dir, "valid.pt")
test_cache  = os.path.join(root_dir, "test.pt")

# ================================
#  1) 캐시 존재 여부 확인
# ================================
if all(os.path.exists(x) for x in [train_cache, valid_cache, test_cache]):
    print("[INFO] Found existing cached datasets. Loading…")

    train_ds = load_dataset(train_cache)
    valid_ds = load_dataset(valid_cache)
    test_ds  = load_dataset(test_cache)


# ================================
#  2) DataLoader 구성
# ================================
from torch_geometric.loader import DataLoader

test_loader = DataLoader(
    test_ds,
    batch_size=4,
    shuffle=False,
    num_workers=60,
    pin_memory=False,
    persistent_workers=False,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ckpt_path = '/NFS_DATA/samsung/foundation/scFoundation/models.ckpt'
ckpt_path = f'{checkpoint_dir}/best.pt'
seq_emb_hidden_dim = max(arr.shape[2] for arr in list(variantseq.embedding_cache.values()))

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
    cond_in_dim=seq_emb_hidden_dim,  # 캐시 임베딩 차원에 맞추세요
    gene_index_tsv="/home/tech/variantseq/foundation/scFoundation/model/OS_scRNA_gene_index.19264.tsv",
    embedding_cache=variantseq.embedding_cache,
    model_config = model_config,
    config = config
)
ckpt = torch.load(ckpt_path, map_location="cuda")
model.load_state_dict(ckpt["model"])


# ---- 사전 준비: 누적 컨테이너 ----
pred_rows, truth_rows = [], []
obs_records = []  # 각 샘플(행)의 메타데이터
global_row = 0

# 유전자(var) 메타데이터: 데이터 순서(=gom.data_genes)
genes = list(gom.data_genes)
var_df = pd.DataFrame(index=pd.Index(genes, name="gene"))


model.to(device)
model.eval()

truth_x_rows, pred_y_rows, truth_y_rows = [], [], []
for batch_idx, batch in enumerate(test_loader):
    batch_ = build_finetune_batch(batch.x, batch.y, model_config["pad_token_id"], model_config["seq_len"])
    if isinstance(batch, dict) and "variant" in batch:
        batch_["variant"] = batch["variant"]
    elif hasattr(batch, "variant"):
        batch_["variant"] = batch.variant

    for k, v in list(batch_.items()):
        if torch.is_tensor(v):
            batch_[k] = v.to(device, non_blocking=True)

    use_amp = True
    ctx = amp_autocast(
        device_type="cuda",
        enabled=(use_amp and str(device).startswith("cuda")),
        dtype=torch.bfloat16,
    )
    
    with torch.inference_mode():
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

        # # 관측 위치만 골라서 MSE 계산 (당신 validate와 일관)
        # observed = ~batch_["decoder_pad_labels"]
        # loss = torch.nn.functional.mse_loss(y_pred[observed], batch_["target"][observed])
    # 데이터 순서로 변환 (BG -> BG)
    truth_x = to_data_order(batch_["x"],            gom, "BG", "BG")  # [B, Gd]
    pred_y = to_data_order(y_pred,            gom, "BG", "BG")  # [B, Gd]
    truth_y = to_data_order(batch_["target"],  gom, "BG", "BG")  # [B, Gd]

    # 텐서 → numpy
    truth_x_np  = torch.atleast_2d(truth_x).detach().cpu().float().numpy()
    pred_y_np  = torch.atleast_2d(pred_y).detach().cpu().float().numpy()
    truth_y_np = torch.atleast_2d(truth_y).detach().cpu().float().numpy()

    B = pred_y_np.shape[0]
    # variant 라벨 정규화
    var_labels = _normalize_variant_labels(batch_.get("variant", None), B)

    # 누적
    truth_x_rows.append(truth_x_np)
    pred_y_rows.append(pred_y_np)
    truth_y_rows.append(truth_y_np)

    # obs 메타데이터(샘플별 한 행)
    for i in range(B):
        obs_records.append({
            "condition": var_labels[i],
        })

import anndata as ann

x_truth  = np.vstack(truth_x_rows)   # [N, G]
y_pred   = np.vstack(pred_y_rows)    # [N, G]
y_truth  = np.vstack(truth_y_rows)   # [N, G]

# gom.missing_in_model # model에 없던 유전자
genes = list(gom.data_genes)
condition = pd.DataFrame(obs_records)['condition'] + '+ctrl'

truth_matrix = np.vstack([y_truth, x_truth])
pred_matrix = np.vstack([y_pred, x_truth])


pred_adata = ann.AnnData(X=pred_matrix)
pred_adata.obs_names = np.concatenate((condition,np.array(['ctrl']*len(obs_records))))
pred_adata.obs['condition']= np.concatenate((condition,np.array(['ctrl']*len(obs_records))))
pred_adata.obs.loc[:, 'dose_val'] = pred_adata.obs.condition.apply(lambda x: '+'.join(['1'] * len(x.split('+'))))
pred_adata.obs.loc[:, 'cell_type'] = variantseq.adata.obs.cell_type.unique()[0]
pred_adata.obs.loc[:, 'condition_name'] =  pred_adata.obs.apply(lambda x: '_'.join([x.cell_type, x.condition, x.dose_val]), axis = 1)
pred_adata.var_names = genes
pred_adata.var['gene_name'] = genes
pred_adata.var.loc[:,'exist'] = True
pred_adata.var.loc[gom.missing_in_model,'exist'] = False
pred_adata.uns = variantseq.adata.uns
pred_adata.write_h5ad(os.path.join(checkpoint_dir, "pred_adata.h5ad"))



truth_adata = ann.AnnData(X=truth_matrix)
truth_adata.obs_names = np.concatenate((condition,np.array(['ctrl']*len(obs_records))))
truth_adata.obs['condition']= np.concatenate((condition,np.array(['ctrl']*len(obs_records))))
truth_adata.obs.loc[:, 'dose_val'] = truth_adata.obs.condition.apply(lambda x: '+'.join(['1'] * len(x.split('+'))))
truth_adata.obs.loc[:, 'cell_type'] = variantseq.adata.obs.cell_type.unique()[0]
truth_adata.obs.loc[:, 'condition_name'] =  truth_adata.obs.apply(lambda x: '_'.join([x.cell_type, x.condition, x.dose_val]), axis = 1)
truth_adata.var_names = genes
truth_adata.var['gene_name'] = genes
truth_adata.var.loc[:,'exist'] = True
truth_adata.var.loc[gom.missing_in_model,'exist'] = False
truth_adata.uns = variantseq.adata.uns
truth_adata.write_h5ad(os.path.join(checkpoint_dir, "truth_adata.h5ad"))