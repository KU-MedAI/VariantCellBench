#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 기본 라이브러리, PerturbNet 모듈, Jupyter 매직 설정

import sys   # 파이썬 인터프리터 관련 기능
import os    # OS(파일 경로 등) 관련 기능
import pickle  # 파이썬 객체를 직렬화/역직렬화하기 위한 모듈
from scipy import sparse  # 희소 행렬(sparse matrix) 처리용
import scanpy as sc       # single-cell 분석용 핵심 패키지
import anndata as ad      # AnnData 자료구조 패키지
from anndata import AnnData  # AnnData 클래스를 직접 import
import scvi               # scVI(single-cell variational inference) 패키지
import pandas as pd       # 테이블 데이터(DataFrame) 처리용
import numpy as np        # 수치 계산용

import torch              # PyTorch 딥러닝 프레임워크
import torch.nn.functional as F  # PyTorch 함수형 API (loss, activation 등)
import torch.nn as nn             # 신경망 레이어 정의용
import esm                        # Facebook ESM protein language model

# PerturbNet에서 제공하는 유틸/모델/데이터 관련 모듈들 import
from perturbnet.util import * 
from perturbnet.cinn.flow import * 
from perturbnet.genotypevae.genotypeVAE import *
from perturbnet.data_vae.vae import *
from perturbnet.cinn.flow_generate import SCVIZ_CheckNet2Net

# Jupyter에서 코드 자동 재로딩 및 inline plotting을 위한 매직 명령어
# # import된 모듈 변경 시 자동으로 다시 로드
# %load_ext autoreload   
# # 모든 모듈에 대해 autoreload 활성화
# %autoreload 2         
# # matplotlib 그래프를 노트북 셀 안에 바로 표시
# %matplotlib inline


# In[ ]:


# jupyter nbconvert --to script Tutorial_PerturbNet_coding_variants_with_explaination.ipynb


# In[ ]:
TARGET_SUM = {
    "hct116": 13750,
    "u2os": 12611,
}

# 'kim2023_hct116'  or 'kim2023_u2os' or 'kim2023_hct116_total_gene' or 'kim2023_u2os_total_gene' or 'kim2023_hct116_total_gene_v2' or 'kim2023_u2os_total_gene_v2'
config = {
    'model':'PerturbNet',                                 # 훈련하고 있는 모델의 이름은?
    'data_name':'kim2023_hct116_[benchmark][3_3-fold]',   # 훈련 데이터의 이름
    'adata_name':'perturb_processed_metadata',            # 데이터 파일 이름
    'embedding_model': 'esm2_t33_650M_UR50D',             # 'esm2_t33_650M_UR50D' / 'esm_msa1_t12_100M_UR50S' / 'ProtT5-XXL-U50' / 'Ankh3-Large' / 'xTrimoPGLM-10B-MLM'
    'variant_representation': 'ALT',                      # 'ALT' / 'DIFF'
    'compression': 'position_embedding',                  # 'position_embedding' / 'full_sequence_average'
}


# In[ ]:


# --- read override json if provided -- #
import os, json
_override_path = os.environ.get("CONFIG_OVERRIDE_PATH")
if _override_path and os.path.exists(_override_path):
    with open(_override_path) as f:
        config.update(json.load(f))
# ------------------------------------- #


# In[11]:


# AnnData 로드
# ToDo: config 기반 데이터 변경 코드 작성
adata = ad.read_h5ad(f"/NFS_DATA/samsung/database/gears/{config['data_name']}/{config['adata_name']}.h5ad")


# In[22]:


# 변이 임베딩 벡터 로드
# ToDo: 남은 임베딩들도 프로토콜 변경 해야함
with open(f"/NFS_DATA/samsung/database/gears/embedding/embedding_cache_variant_position_[{config['embedding_model']}]_proto4.pkl", 'rb') as f:
    emb = pickle.load(f)


# In[ ]:


# 데이터 전처리 과정

import re
import numpy as np

def parse_condition_to_gene_var(cond: str):
    base = cond.split('+')[0]

    if '~' in base:
        gene, var = base.split('~', 1)
    else:
        m = re.match(r"([A-Za-z0-9]+)_(p\.[A-Za-z0-9]+)", base)
        if m:
            gene, var = m.group(1), m.group(2)
        else:
            raise ValueError(f"조건 문자열 포맷을 모르겠음: {cond}")

    return gene, var


def _get_any_vec_list(embedding_cache, expr_type: str):
    """
    embedding_cache에서 expr_type에 해당하는 벡터(list[float]) 아무거나 하나를 찾아 반환.
    """
    for key, d in embedding_cache.items():
        if expr_type in d:
            vec_list = d[expr_type]
            if vec_list is None:
                continue
            if isinstance(vec_list, (list, tuple)) and len(vec_list) > 0:
                return vec_list
    raise KeyError(f"embedding_cache 안에서 expr_type='{expr_type}' 벡터를 찾지 못함")


def cache_to_sequence_representations_from_adata(
    adata,
    embedding_cache,
    expr_type="ALT",
    condition_col="condition",
):
    conditions = adata.obs[condition_col].unique().tolist()
    sequence_representations = []

    # ctrl masking용 reference dim
    ref_vec_list = _get_any_vec_list(embedding_cache, expr_type)
    D = len(ref_vec_list)

    for cond in conditions:
        # ctrl → zero vector
        if cond == "ctrl":
            vec_arr = np.zeros((D,), dtype=np.float32)
            sequence_representations.append(vec_arr)
            continue

        gene, var = parse_condition_to_gene_var(cond)
        key = (gene, var)

        if key not in embedding_cache:
            raise KeyError(f"{key} not in embedding_cache (condition={cond})")
        if expr_type not in embedding_cache[key]:
            raise KeyError(f"{expr_type} not found for {key} in embedding_cache")

        vec_list = embedding_cache[key][expr_type]
        vec_arr = np.asarray(vec_list, dtype=np.float32)  # (D,)

        if vec_arr.shape[0] != D:
            raise ValueError(
                f"Embedding dim mismatch: expected {D}, got {vec_arr.shape[0]} (key={key})"
            )

        sequence_representations.append(vec_arr)

    # 🔑 핵심: stack 해서 (N, D) ndarray로 변환
    sequence_representations = np.stack(sequence_representations, axis=0)

    return conditions, sequence_representations



conditions, sequence_representations = cache_to_sequence_representations_from_adata(
    adata,
    emb,
    expr_type=config['variant_representation'],
)


# In[59]:


# split 기준으로 train/test 세트 분리

# obs["split1"] == "train" 인 셀만 선택해 새로운 AnnData 생성
adata_train = adata[adata.obs.split == "train", :].copy()

# obs["split1"] == "test" 인 셀만 선택해 새로운 AnnData 생성
adata_test = adata[adata.obs.split == "test", :].copy()


# In[60]:


import os
import scvi

# -------------------------------------------------
# SCVI 모델 저장 경로
# -------------------------------------------------
scvi_model_save_path = f"/NFS_DATA/samsung/variantPerturbNet/cellvae/{config['data_name']}"

# -------------------------------------------------
# AnnData에 SCVI 세팅 (counts layer 사용)
# ⚠️ 학습/로드 모두 동일하게 setup_anndata 필요
# -------------------------------------------------
scvi.data.setup_anndata(
    adata_train,
    layer="counts"
)

# -------------------------------------------------
# SCVI 모델이 존재하지 않으면 학습
# -------------------------------------------------
if not os.path.exists(scvi_model_save_path):

    print("SCVI model not found. Training a new model...")

    # 잠재 공간(latent dimension) = 10
    scvi_model = scvi.model.SCVI(
        adata_train,
        n_latent=10
    )

    # 튜토리얼 권장 설정: 700 epochs
    scvi_model.train(
        n_epochs=700,
        frequency=20  # 20 step마다 history 기록
    )

    # 모델 저장
    scvi_model.save(scvi_model_save_path)
    print(f"SCVI model saved to: {scvi_model_save_path}")

# -------------------------------------------------
# 이미 학습된 모델이 있으면 로드
# -------------------------------------------------
else:
    print("Loading existing SCVI model...")

    scvi_model = scvi.model.SCVI.load(
        scvi_model_save_path,
        adata=adata_train,
        use_cuda=False  # GPU 사용 시 True
    )

    print("SCVI model loaded successfully.")


# In[61]:


# 
adata_train = adata[adata.obs.split != "test", :].copy()
adata_train.obsm["X_scVI"] = scvi_model.get_latent_representation(adata_train)
adata_train.uns['ordered_all_trt'] = conditions
adata_train.uns['ordered_all_embedding'] = sequence_representations


# In[66]:


def prepare_embeddings_cinn_2(adata, perturbation_key, trt_key, embed_key):
    """
    CINN (Cell Inference of Network Navigation) 분석을 위해
    perturbation embedding을 준비하는 함수.

    - 각 cell의 perturbation label을 가져오고
    - perturbation type별 embedding을 추출한 뒤
    - perturbation → embedding index 매핑 딕셔너리를 생성한다
    """

    # --------------------------------------------------
    # 1) 각 cell의 perturbation label 추출
    # --------------------------------------------------
    # adata.obs[perturbation_key]는 cell 단위 perturbation 정보
    # 예: ['TP53', 'TP53', 'CTRL', 'BRCA1', ...]
    perturb_with_onehot = np.array(adata.obs[perturbation_key])

    # --------------------------------------------------
    # 2) perturbation 종류 목록 추출 (중복 제거)
    # --------------------------------------------------
    # 예: ['BRCA1', 'CTRL', 'TP53']
    trt_list = np.unique(perturb_with_onehot)

    # --------------------------------------------------
    # 3) 각 perturbation에 대응하는 embedding index 찾기
    # --------------------------------------------------
    trt_all = np.array(adata.uns[trt_key])

    emb_all = np.array(adata.uns[embed_key])

    embed_idx = []
    for i in range(len(trt_list)):
        trt = trt_list[i]

        # adata.uns[trt_key]는 embedding이 정의된 perturbation 목록
        # 예: ['CTRL', 'TP53', 'BRCA1', ...]
        # 그중 trt가 위치한 index를 찾음
        idx = np.where(trt_all == trt)[0][0]

        embed_idx.append(idx)

    # --------------------------------------------------
    # 4) perturbation별 embedding 추출
    # --------------------------------------------------
    # embed_key에 저장된 전체 embedding 중
    # 우리가 사용하는 perturbation에 해당하는 것만 선택
    embeddings = emb_all[embed_idx]

    # --------------------------------------------------
    # 5) perturbation → embedding index 매핑 생성
    # --------------------------------------------------
    # CINN에서 perturbation label을 embedding index로 바꾸기 위해 사용
    # 예: {'CTRL': 0, 'TP53': 1, 'BRCA1': 2}
    perturbToEmbed = {}
    for i in range(trt_list.shape[0]):
        perturbToEmbed[trt_list[i]] = i

    # --------------------------------------------------
    # 6) 결과 반환
    # --------------------------------------------------
    return perturb_with_onehot, embeddings, perturbToEmbed


# In[67]:


# [Cell 17] CINN(flow) 모델 학습을 위해 조건(변이) / 임베딩 / 매핑 준비

# prepare_embeddings_cinn:
#  - cond_stage_data        : 각 셀에 대한 perturbation label
#  - embeddings             : 변이에 해당하는 ESM 임베딩
#  - perturbToEmbed         : perturbation ↔ embedding 인덱스 맵
cond_stage_data, embeddings, perturbToEmbed = prepare_embeddings_cinn_2(
    adata_train,
    perturbation_key="condition",   # 어떤 obs 컬럼을 perturbation label로 쓸지
    trt_key="ordered_all_trt",        # adata.uns에 저장된 treatment 순서 키
    embed_key="ordered_all_embedding" # adata.uns에 저장된 임베딩 키
)


# In[72]:


len(cond_stage_data)


# In[73]:


embeddings.shape


# In[75]:


len(perturbToEmbed)


# In[77]:


len(cond_stage_data)


# In[80]:


scvi_model.adata = adata_train


# In[81]:


# [Cell 19] Conditional INN(Flow) 모델 정의 및 래퍼 객체(Net2NetFlow_scVIFixFlow) 생성

# 재현성을 위한 PyTorch 시드 고정
torch.manual_seed(42)

# perturbation 임베딩(1280차원)을 condition으로,
# scVI latent(10차원)를 target으로 하는 Conditional Flat Coupling Flow 정의
flow_model = ConditionalFlatCouplingFlow(
    conditioning_dim=embeddings[0].shape[0],   # perturbation embedding 차원
    embedding_dim=10,        # cell state(scVI latent) 차원
    conditioning_depth=2,    # conditioner 네트워크 깊이
    n_flows=20,              # 플로우 블록 개수
    in_channels=10,          # 입력 채널 수 (latent 차원과 같게 설정)
    hidden_dim=1024,         # 내부 fully-connected layer hidden 크기
    hidden_depth=2,          # hidden layer 개수
    activation="none",       # 활성화 함수 (예: "relu" 대신 none 사용)
    conditioner_use_bn=True  # batch normalization 사용 여부
)

# cond_stage_data: 각 셀의 perturbation label(variant_seq 등)
cond_stage_data = np.array(adata_train.obs["condition"])

# Net2NetFlow_scVIFixFlow:
#  - configured_flow: 위에서 정의한 flow_model
#  - cond_stage_data: 조건(label) 정보
#  - perturbToEmbed : perturbation → embedding index 매핑
#  - embedData      : 실제 ESM 임베딩 배열
#  - scvi_model     : 사전에 학습된 scVI 모델
model_c = Net2NetFlow_scVIFixFlow(
    configured_flow=flow_model,
    cond_stage_data=cond_stage_data,
    perturbToEmbedLib=perturbToEmbed,
    embedData=embeddings,
    scvi_model=scvi_model,
)
model_c.adata = adata_train


# In[112]:


import os, re, json, time, signal, math, torch
from torch import nn
from torch.cuda import amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
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
def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M")

from datetime import datetime
seed_all(42)

timestamp = get_timestamp()
project_name = make_project_name(config, timestamp)

checkpoint_dir = os.path.join('/NFS_DATA/samsung/variantPerturbNet/cinn', project_name)


# In[ ]:


# [Cell 21] Flow 모델을 GPU/CPU로 옮겨 학습 후 저장

# CUDA 사용 가능 여부에 따라 device 선택
device = "cuda" if torch.cuda.is_available() else "cpu"

# CINN 모델 저장 경로
path_cinn_model_save = checkpoint_dir

# 모델을 해당 device로 이동
model_c.to(device=device)

# Flow 모델 학습
# - n_epochs: 총 epoch 수
# - batch_size: 미니배치 크기
# - lr: learning rate
# - train_ratio: train/validation split 비율
model_c.train(
    n_epochs=25,
    batch_size=128,
    lr=4.5e-6,
    train_ratio=0.8,
)

#### save the model
model_c.save(path_cinn_model_save)

import pickle
with open(f"{checkpoint_dir}/config.pkl", "wb") as f:
    pickle.dump(config, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(f"{checkpoint_dir}/config.json", "w") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)


# In[85]:


# [Cell 23] 저장된 Flow 모델을 다시 로드하는 단계일 가능성이 큼
# (실제 코드가 3줄이라면 예: model_c.load(...) 등의 형태일 수 있음)

model_c.load(path_cinn_model_save)
model_c.to(device = device)


# In[86]:


# test set에서 SCVI latent 추출 등의 준비 코드
Zsample_test = scvi_model.get_latent_representation(adata = adata_test, give_mean = False)


# In[87]:


model_c.eval()
scvi_model_de = scvi_predictive_z(scvi_model)
perturbnet_model = SCVIZ_CheckNet2Net(model_c, device, scvi_model_de)
Lsample_obs = scvi_model.get_latent_library_size(adata = adata_train, give_mean = False)


# In[88]:


# 데이터 할당
adata_test.obsm["X_scVI"] = scvi_model.get_latent_representation(adata_test)
adata_test.uns['ordered_all_trt'] = np.array(conditions)
adata_test.uns['ordered_all_embedding'] = np.array(sequence_representations)


# In[ ]:


# # 특정 perturbation에 대해 Flow 모델로 예측 샘플 생성

# # unseen_pert: 테스트할 perturbation(예: 실험에서 보지 못한 variant label)
# unseen_pert = adata_test.obs["condition"].unique()[0]
# pert_idx = np.where(adata_test.uns["ordered_all_trt"] == unseen_pert)[0][0]
# unseen_pert_embed = adata_test.uns["ordered_all_embedding"][pert_idx]


# In[122]:


# # 정답 데이터
# real_data = adata_test.layers["counts"].A[np.where(adata_test.obs.condition  == unseen_pert)[0]]


# In[97]:


import numpy as np
import anndata as ad

# data_name에서 cell line 추출
data_name = config["data_name"].lower()

if "hct116" in data_name:
    cell_line_l = "hct116"
elif "u2os" in data_name:
    cell_line_l = "u2os"
else:
    raise ValueError(f"Unknown cell line in data_name: {data_name}")

# TARGET_SUM 사용
# target_sum = TARGET_SUM[cell_line_l]


adata_truth_list = []
adata_pred_list  = []

all_conditions = adata_test.obs["condition"].unique()

for i, unseen_pert in enumerate(all_conditions):

    # -----------------------------
    # 1. perturbation embedding
    # -----------------------------
    pert_idx = np.where(adata_test.uns["ordered_all_trt"] == unseen_pert)[0][0]
    unseen_pert_embed = adata_test.uns["ordered_all_embedding"][pert_idx]

    # -----------------------------
    # 2. real data (truth)
    # -----------------------------
    cell_idx = np.where(adata_test.obs["condition"] == unseen_pert)[0]
    real_data = adata_test.layers["counts"].A[cell_idx]
    n_cells = real_data.shape[0]

    adata_truth = ad.AnnData(
        X=real_data,
        obs=adata_test.obs.iloc[cell_idx].copy(),
        var=adata_test.var.copy()
    )
    adata_truth.obs["source"] = "truth"
    adata_truth.obs["condition"] = unseen_pert

    adata_truth_list.append(adata_truth)

    # -----------------------------
    # 3. predicted data (Flow sampling)
    # -----------------------------
    Lsample_idx = np.random.choice(
        range(Lsample_obs.shape[0]),
        n_cells,
        replace=True
    )
    library_trt_latent = Lsample_obs[Lsample_idx]

    trt_input_fixed = np.tile(unseen_pert_embed, (n_cells, 1))
    pert_embed = trt_input_fixed + np.random.normal(
        scale=0.001,
        size=trt_input_fixed.shape
    )

    predict_latent, predict_data = perturbnet_model.sample_data(
        pert_embed,
        library_trt_latent
    )

    adata_pred = ad.AnnData(
        X=predict_data,
        obs=adata_test.obs.iloc[cell_idx].copy(),
        var=adata_test.var.copy()
    )
    adata_pred.obs["source"] = "pred"
    adata_pred.obs["condition"] = unseen_pert

    adata_pred_list.append(adata_pred)

# -----------------------------
# 4. concat
# -----------------------------
adata_truth_all = ad.concat(
    adata_truth_list,
    axis=0,
    merge="same",
    index_unique=None
)

adata_pred_all = ad.concat(
    adata_pred_list,
    axis=0,
    merge="same",
    index_unique=None
)


# In[103]:


adata_truth_all.uns = adata_test.uns
adata_pred_all.uns = adata_test.uns


# In[ ]:


adata_truth_all.layers["counts"] = adata_truth_all.X.copy()
adata_pred_all.layers["counts"]  = adata_pred_all.X.copy()

# sc.pp.normalize_total(adata_truth_all, target_sum=1e4)
# sc.pp.log1p(adata_truth_all)

# sc.pp.normalize_total(adata_pred_all, target_sum=1e4)
# sc.pp.log1p(adata_pred_all)


sc.pp.normalize_total(
    adata_truth_all,
    target_sum=TARGET_SUM[cell_line_l],
)
sc.pp.log1p(adata_truth_all)


sc.pp.normalize_total(
    adata_pred_all,
    target_sum=TARGET_SUM[cell_line_l],
)
sc.pp.log1p(adata_pred_all)


# In[110]:


pred_path  = os.path.join(checkpoint_dir, "pred_adata.h5ad")
truth_path = os.path.join(checkpoint_dir, "truth_adata.h5ad")

adata_pred_all.write(pred_path)
adata_truth_all.write(truth_path)


# In[ ]:


# # 예측 데이터
# n_cells = real_data.shape[0] # original: 108
# Lsample_idx = np.random.choice(range(Lsample_obs.shape[0]), n_cells, replace=True)
# trt_input_fixed = np.tile(unseen_pert_embed, (n_cells, 1))
# pert_embed = trt_input_fixed + np.random.normal(scale = 0.001, size = trt_input_fixed.shape)
# library_trt_latent = Lsample_obs[Lsample_idx] 
# predict_latent, predict_data = perturbnet_model.sample_data(pert_embed, library_trt_latent)


# In[100]:


# # Flow 예측 결과와 실제 데이터를 비교하기 위한 준비

# # real_data: 실제 test 셀들의 count 혹은 normalized expression
# # predict_data: Flow를 통해 생성한 가상(예측) expression
# # 두 데이터를 같은 형태로 맞춰서 후속 평가/시각화에 사용
# real_latent = Zsample_test[np.where(adata_test.obs.condition  == unseen_pert)[0]]
# real_latent.shape


# In[103]:


# all_latent = np.concatenate([predict_latent, real_latent], axis = 0)
# cat_t = ["Real"] * real_latent.shape[0]
# cat_g = ["Predict"] * predict_latent.shape[0]
# cat_rf_gt = np.append(cat_g, cat_t)
# trans = umap.UMAP(random_state=42, min_dist = 0.5, n_neighbors=30).fit(all_latent)
# X_embedded_pr = trans.transform(all_latent)
# df = X_embedded_pr.copy()
# df = pd.DataFrame(df)
# df['x-umap'] = X_embedded_pr[:,0]
# df['y-umap'] = X_embedded_pr[:,1]
# df['category'] = cat_rf_gt
    
# chart_pr = ggplot(df, aes(x= 'x-umap', y= 'y-umap', colour = 'category') ) \
#     + geom_point(size=0.5, alpha = 0.5) \
#     + ggtitle("UMAP dimensions")
# chart_pr


# In[ ]:


# # [Cell 37] Normalized Revision R-square 계산을 위한 객체 생성

# # largeCountData에 raw count matrix를 전달하여 정규화된 R²를 계산할 준비
# # adata_test.layers["counts"]는 sparse matrix이므로 .A로 dense array로 변환
# normModel = NormalizedRevisionRSquare(largeCountData = adata_test.layers["counts"].A)


# In[106]:


# # [Cell 38] 실제 데이터 vs 예측 데이터에 대한 정규화된 R² 계산

# # r2_value: overall R² 값
# # 두 번째/세 번째 반환값은 gene-wise 또는 cell-wise R² 등일 수 있음 (튜토리얼 정의에 따름)
# r2_value, _, _ = normModel.calculate_r_square(real_data, predict_data)
# r2_value


# In[109]:


adata.obsm["X_scVI"] = scvi_model.get_latent_representation(adata)
adata.uns['ordered_all_trt'] = np.array(conditions)
adata.uns['ordered_all_embedding'] = np.array(sequence_representations)


# In[ ]:


# background_pert = []
# background_cell = []
# n_cells_bk = 20
# highlights = [unseen_pert]
# for i in tqdm(range(len(adata.uns["ordered_all_trt"]))):
#     pert = adata.uns["ordered_all_trt"][i]
#     if pert in highlights:
#         continue
#     pert_embed_tmp = adata.uns["ordered_all_embedding"][i]
#     Lsample_idx = np.random.choice(range(Lsample_obs.shape[0]), n_cells, replace=True)
#     trt_input_fixed_bk = np.tile(pert_embed_tmp, (n_cells_bk, 1))
#     pert_embed_bk = trt_input_fixed_bk + np.random.normal(scale = 0.001, size = trt_input_fixed_bk.shape)
#     library_trt_latent = Lsample_obs[Lsample_idx]
#     predict_latent_bk, predict_data_bk = perturbnet_model.sample_data(pert_embed_bk, library_trt_latent)
    
#     background_pert.append(pert_embed_bk )
#     background_cell.append(predict_latent_bk)
    
# background_pert = np.concatenate(background_pert)
# background_cell = np.concatenate(background_cell)
    


# In[ ]:


# # [Cell 41] unseen perturbation vs background 분포를 contour plot으로 시각화

# %matplotlib inline

# # contourplot_space_mapping:
# #  - predict_latent     : unseen perturbation에서 샘플링한 latent
# #  - pert_embed         : unseen perturbation의 embedding
# #  - background_pert    : 다른 perturbation들의 embedding 샘플
# #  - background_cell    : 그 perturbation들에서 샘플링한 latent
# #  - highlight_labels   : 강조하고 싶은 perturbation 레이블
# #  - colors             : 해당 highlight 색상
# contourplot_space_mapping(
#     predict_latent,
#     pert_embed,
#     background_pert,
#     background_cell,
#     highlight_labels=["TP53~V274A+ctrl"],
#     colors=["red"],
# )


# In[ ]:




