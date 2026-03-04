#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# # 주피터 to 파이썬 변환 명령어
# jupyter nbconvert --to script variantGEARS_1126.ipynb
import os

# 물리 GPU 2번만 보이게
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
torch.set_num_threads(8)           # 연산에 사용하는 intra-op 스레드 수
torch.set_num_interop_threads(2)   # 연산 간 병렬화 정도 (보통 작게)
# In[1]:


from zipfile import ZipFile
from collections import defaultdict
import numpy as np
import pandas as pd
import networkx as nx
import scanpy as sc
import os
import pickle

import torch 
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.data import DataLoader

import zarr
import numpy as np
import requests
import mygene
import json
import os
import torch


# In[2]:


from config import *
from data import *
from emb_data import *
from network import *
from result import *
from variant import *
from utils import *
from bsh import *



config = {
    'model':'GEARS',                                      # 훈련하고 있는 모델의 이름은?
    'data_name':'kim2023_hct116_[benchmark][1_3-fold]',   # 훈련 데이터의 이름
    'adata_name':'perturb_processed_metadata',            # 데이터 파일 이름
    'split':'exist',                                      # 데이터 분할 조건. 이미 생성된 분할 데이터를 사용한다면, exist
    'embedding_model': 'esm2_t33_650M_UR50D',             # 'esm2_t33_650M_UR50D' / 'esm_msa1_t12_100M_UR50S' / 'ProtT5-XXL-U50' / 'Ankh3-Large' / 'xTrimoPGLM-10B-MLM'
    'mutation_type': 'amino',                             # 크게 중요 X
    'gears_mode':'variant',                               # 'variant' / 'gene'
    'embedding_merge_method': 'no_pert',                  # 'cat' or 'no_pert' or 'element' or 'bilinear'
    'variant_representation': 'ALT',                      # 'ALT' / 'DIFF'
    'compression': 'position_embedding',                  # 'position_embedding' / 'full_sequence_average'
    'pert_graph' : False,                                 # Pert graph gnn 적용 시 True. {default: False}
    'epochs': 20,                                         # Epoch 수
    'batch_size': 32,                                     # Batch size
    # 'hidden_dim': 64,                                   # model hidden dim size
    'learning_rate': 1e-3,                                # Learning rate
    'weight_decay': 5e-4,                                 # Learning rate decay
    'uncertainty': False,                                 # 사용하지 않는 옵션. uncertainty loss 적용 유무
    'n_gene_layer': 1,                                    # Gene GNN layer
    'n_condition_layer': 1,                               # Pert GNN layer
    'compress_by': 'hidden-wise',                         # 'hidden-wise' or 'sequence-wise'

    'direction_method': 'tanh',                           # 'sign' or 'tanh' or 'hybrid'
    # loss_version이 1이 아닐 때 적용되는 옵션
    'direction_lambda': 5,                                #
    'direction_alpha': 0.5,                               #

    'loss_version': 1,                                    # 
    'top_n_hvg': 100,                                     #
    'hvg_weight': 0,                                      #

    'trainer': 'Ban',                                     # 모델 훈련 담당자
    'use_dummy_embedding': False,                         # Ablation 목적의 0 embedding
    'visible_emb': False                                  # Embedding 꺼내먹기 False or True
}


# In[ ]:


# --- read override json if provided -- #
import os, json
_override_path = os.environ.get("CONFIG_OVERRIDE_PATH")
if _override_path and os.path.exists(_override_path):
    with open(_override_path) as f:
        config.update(json.load(f))
# ------------------------------------- #


# In[9]:


# embedding_merge_method
# learning_rate
# compress_by


# In[10]:


# with open(file=f'{GEARS_DATA_PATH}/{config['data_name']}/data_pyg/cell_graphs.pkl', mode='rb') as f:
#     dataset_processed=pickle.load(f)


# In[11]:


# dataset_processed['TP53~V274A+ctrl'][0].de_idx


# In[12]:


# path = os.path.join(GEARS_DATA_PATH, "norman/perturb_processed.h5ad")
# adata = sc.read_h5ad(path)


# In[13]:


Data = CustomConditionData(GEARS_DATA_PATH)


# In[14]:


# dict_keys(['hvg', 'log1p', 'non_dropout_gene_idx', 'non_zeros_gene_idx', 'not_in_go_pert', 'rank_genes_groups_cov_all', 'top_non_dropout_de_20', 'top_non_zero_de_20'])
# Data.adata.uns.keys()

# Data.adata.uns['top_non_zero_de_20']


# In[15]:


# collect_data("KRAS", "REF", "esm2_t33_650M_UR50D", "amino")


# In[16]:


# Embedding Data 생성을 위한 코드.

# condition_list = get_condition_lists(Data.adata.obs[Data.condition_col]).tolist()
# for gene in tqdm(condition_list): # Gene level
#     print(gene)
#     # enable_error_logging()
#     collect_data(gene, "REF", "esm2_t33_650M_UR50D", "amino")
# for gene_var_ in tqdm(condition_list): # Variant level
#     gene, var = gene_var_.split('~')
#     collect_data(gene, var, "esm2_t33_650M_UR50D", "amino")



# In[17]:


# Data.adata.uns['non_zeros_gene_idx']['HCT116_TP53~A161T+ctrl_1+1']


# In[18]:


Data.load(config)


# In[19]:


# len (random.choice(list(random.choice(list(Data.embedding_cache.values())).values())))


# In[20]:


# # 타 모델 학습을 위한 data split까지 완료한 데이터 저장
# split_dir = f'/NFS_DATA/samsung/database/gears/{config['data_name']}/dataloader'

# if not os.path.exists(split_dir):
# 	os.makedirs(split_dir)
# with open(f'{split_dir}/dataloader.pkl', 'wb') as f:
# 	pickle.dump(Data.dataloader, f, protocol=pickle.HIGHEST_PROTOCOL)

# print(f"{config['data_name']} split data saved")


# In[21]:


# Gene-Go dictionary에 없어, 예측에서 제외된 유전자 확인
# Data.adata.uns['not_in_go_pert']


# In[22]:


# rank_genes_groups_cov_all에서 head 20 인덱싱하면 top_non_dropout_de_20임
# Data.adata.uns['top_non_dropout_de_20']['A549_AHR+FEV_1+1'] == Data.adata.uns['rank_genes_groups_cov_all']['A549_AHR+FEV_1+1'][:20] # all True


# In[23]:


# device = 'cuda'
# model = GEARS(Data, gears_mode = config['gears_mode'], loss_version=config['loss_version']).to(device)


# In[24]:


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

from datetime import datetime
seed_all(42)

timestamp = get_timestamp()
project_name = make_project_name(config, timestamp)

checkpoint_dir = os.path.join(OUTPUT_DIR, project_name)
os.makedirs(checkpoint_dir, exist_ok=True)
import pickle
with open(f"{checkpoint_dir}/config.pkl", "wb") as f:
    pickle.dump(config, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(f"{checkpoint_dir}/config.json", "w") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)


# In[25]:


project_name


# In[26]:


# import wandb
# wandb.finish()


# In[27]:


# variant module 수정 중인 GEARS 모델 클래스
device = 'cuda'
model = GEARS_2(Data, config=config).to(device)


# In[28]:


# 프로젝트 이름과 실험 이름 지정
# wandb.init(project="Variant-seq", name=f"{project_name}", config=config)

# config 값을 wandb에 등록
# wandb.config.update(config)


# In[29]:


# collect_data("TP53","REF", "esm2_t33_650M_UR50D", "amino")
# load_data("TP53","REF", "esm2_t33_650M_UR50D", "amino")


# In[30]:


# # 모델 클래스 또는 외부 코드에서
# model.hook_outputs = {}

# def save_output_hook(name):
#     def hook(module, input, output):
#         model.hook_outputs[name] = output
#     return hook

# # Variant Embedding MLP hook
# if hasattr(model, 'variant_condition_mix_MLP'):
#     model.variant_condition_mix_MLP.register_forward_hook(save_output_hook("variant_condition_mix_MLP"))

# # Condition Embedding MLP hook
# model.condition_mix_MLP.register_forward_hook(save_output_hook("condition_mix_MLP"))

# # Gene Embedding flow hook 예시
# model.position_MLP.register_forward_hook(save_output_hook("position_MLP"))
# model.postpert_MLP.register_forward_hook(save_output_hook("postpert_MLP"))
# model.cross_gene_state.register_forward_hook(save_output_hook("cross_gene_state"))


# In[31]:


update_visible = False


# In[32]:


model.dataloader.dataloader


# In[33]:


Data.adata


# In[ ]:


from datetime import datetime
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from tqdm import tqdm

loss_log = []
save_every = 5
checkpoint_dir = os.path.join(OUTPUT_DIR,project_name)
log_path = os.path.join(checkpoint_dir, f"loss_log_{timestamp}.csv")
os.makedirs(checkpoint_dir, exist_ok=True)


optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay = config['weight_decay'])
scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

#모니터링 설정
monitor_metric = 'val_loss'
monitor_mode   = 'min'
min_delta      = float(config.get('min_delta', 0.0))
# patience       = int(config.get('patience', 20))
best_saver = BestSaver(monitor=monitor_metric, mode=monitor_mode, min_delta=min_delta)

train_loader = model.dataloader.dataloader['train_loader']
val_loader   = model.dataloader.dataloader['val_loader']


for epoch in tqdm(range(config['epochs'])):
    print(f'[Epoch {epoch+1}/{config['epochs']}] {project_name}')
    model.train()
    epoch_loss = 0
    epoch_autofocus_loss = 0
    epoch_direction_loss = 0
    epoch_hvg_loss = 0
    epoch_hvg_autofocus_loss = 0
    epoch_hvg_direction_loss = 0
    visible = False

    num_steps = len(train_loader)
    for step, batch in tqdm(enumerate(train_loader)):
        batch.to(model.device)

        # 파라미터 snapshot
        if update_visible:
            old_params = {name: param.clone().detach() for name, param in model.named_parameters() if param.requires_grad}

        optimizer.zero_grad()
        y = batch.y
        if model.uncertainty:
            pred, logvar = model(batch)
            loss = uncertainty_loss_fct(pred, logvar, y, batch.pert,
                    reg = model.param_uncertainty_reg,
                    ctrl = model.ctrl_expression,
                    dict_filter = model.dict_filter,
                    direction_lambda = model.param_direction_lambda)
        else:
            # visible = ["variant_condition_index"]
            visible = []
            pred = model(batch, visible)
            # def loss_fct(pred, y, perts, hvgs, ctrl=None, direction_lambda=1e-1, direction_alpha = 0.5, direction_method=None, hvg_weight=2.0, dict_filter=None, loss_version=None, visible=False):
            loss, autofocus_loss, direction_loss, hvg_loss, autofocus_hvg_loss, direction_hvg_loss = loss_fct(
                pred, y, batch.pert, model.dataloader.hvg_n_idx,
                ctrl=model.ctrl_expression,
                dict_filter=model.dict_filter,
                loss_version=model.loss_version,
                direction_lambda=config['direction_lambda'],
                direction_alpha=config['direction_alpha'],
                direction_method=config['direction_method'],
                hvg_weight=config['hvg_weight'],
                visible=visible
            )
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
        optimizer.step()

        # 파라미터 업데이트 여부 확인
        if update_visible:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if not torch.equal(old_params[name], param):
                        print(f"✅ Parameter '{name}' has been updated.")
                    else:
                        print(f"❌ Parameter '{name}' has NOT been updated.")

        # 누적
        epoch_loss += loss.item()
        epoch_autofocus_loss += float(autofocus_loss.detach().item())
        epoch_direction_loss += float(direction_loss.detach().item())
        epoch_hvg_loss += float(hvg_loss.detach().item())
        epoch_hvg_autofocus_loss += float(autofocus_hvg_loss.detach().item())
        epoch_hvg_direction_loss += float(direction_hvg_loss.detach().item())


    # 평균
    avg_loss = epoch_loss / max(1, num_steps)
    avg_autofocus_loss = epoch_autofocus_loss / max(1, num_steps)
    avg_direction_loss = epoch_direction_loss / max(1, num_steps)
    avg_hvg_loss = epoch_hvg_loss / max(1, num_steps)
    avg_hvg_autofocus_loss = epoch_hvg_autofocus_loss / max(1, num_steps)
    avg_hvg_direction_loss = epoch_hvg_direction_loss / max(1, num_steps)


    # -------- Evaluate --------
    model.eval()
    with torch.no_grad():
        train_res = evaluate(train_loader, model, model.uncertainty, model.device)
        val_res   = evaluate(val_loader,   model, model.uncertainty, model.device)
    train_metrics, _ = compute_metrics(train_res)
    val_metrics, val_pert_res = compute_metrics(val_res)

    # log_embedding_heatmap(model, epoch, mode="val")
    val_loss = compute_avg_loss_on_loader(model, val_loader, model.device, config, visible_cfg=[])
    #--------------#
    # 추가 mertric #
    train_out = deeper_analysis(Data.adata, train_res)
    val_out = deeper_analysis(Data.adata, val_res)
    train_out_result = {}
    val_out_result = {}
    for m in list(list(train_out.values())[0].keys()):  # 첫 perturbation의 metric 키 기준
        train_out_result['train_' + m] = np.mean([j[m] for j in train_out.values() if m in j])
    for m in list(list(val_out.values())[0].keys()):  # 첫 perturbation의 metric 키 기준
        val_out_result['val_' + m] = np.mean([j[m] for j in val_out.values() if m in j])




    # wandb 로깅
    # 기본 메트릭
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
    }
    log_dict.update({
        "val_loss": val_loss
    })
    # -------- Best checkpoint 저장(가장 중요한 부분) --------
    # 사용자가 지정한 monitor를 가져옴 (val_metrics / 로그 dict에서 찾기)
    # 기본은 val_metrics 우선, 없으면 log_dict 탐색
    if monitor_metric in val_metrics:
        current_value = float(val_metrics[monitor_metric.replace("val_", "")])  # 예: 'val_mse' -> 'mse'
    else:
        current_value = log_dict.get(monitor_metric, None)
        if current_value is None:
            raise ValueError(f"Monitor metric '{monitor_metric}' not found in val_metrics or log_dict.")

    if best_saver.is_better(current_value):
        best_saver.best_value = current_value
        best_saver.best_epoch = epoch + 1
        best_path = os.path.join(checkpoint_dir, "best.pt")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': current_value
        }
        best_saver.best_path = best_path
        torch.save(checkpoint, best_path)
        print(f"🌟 New best {monitor_metric} = {current_value:.6f} at epoch {epoch+1}. Saved to {best_path}")
    scheduler.step()



    # # 방향성 및 범위 메트릭
    for key in ['frac_correct_direction_20', 'frac_correct_direction_all']:
        log_dict[f"train_{key}"] = train_out_result[f"train_{key}"]
        log_dict[f"val_{key}"] = val_out_result[f"val_{key}"]

    # HVG 관련 성능
    for top_k in [100, 250, 500, 1000]:
        for metric in ['mse', 'pearson', 'pearson_delta']:
            key = f"{metric}_top{top_k}_hvg"
            log_dict[f"train_{key}"] = train_out_result.get(f"train_{key}", None)
            log_dict[f"val_{key}"] = val_out_result.get(f"val_{key}", None)

    # DE gene 관련 성능
    for top_k in [20, 50, 100, 200]:
        for metric in ['mse', 'pearson', 'pearson_delta']:
            key = f"{metric}_top{top_k}_de"
            log_dict[f"train_{key}"] = train_out_result.get(f"train_{key}", None)
            log_dict[f"val_{key}"] = val_out_result.get(f"val_{key}", None)

    # wandb 로깅
    # wandb.log(log_dict)
    print(f'[{epoch+1}/{config['epochs']}] {project_name}')
    print('Train loss: {:.4f} | Valid loss: {:.4f}'.format(avg_loss,val_loss))

final_path = os.path.join(checkpoint_dir, f"final_model.pt")
save_checkpoint(model, optimizer, config['epochs'], avg_loss, final_path)


# In[ ]:


checkpoint_path = f"{checkpoint_dir}/best.pt"
# 1. checkpoint 불러오기
checkpoint = torch.load(checkpoint_path, map_location=device)

# 2. 모델 가중치 로드
model.load_state_dict(checkpoint["model_state_dict"])
test_res = evaluate(model.dataloader.dataloader['test_loader'], model, model.uncertainty, model.device)
test_metrics, test_pert_res = compute_metrics(test_res)

# Print epoch performance
print("Test Overall MSE: {:.4f}.".format(test_metrics['mse']))
print("Test Overall PCC: {:.4f}.".format(test_metrics['pearson']))

print(f'✅ Model train done: {project_name}')
print('[Final loss] Train loss: {:.4f} | Valid loss: {:.4f}'.format(avg_loss,val_loss))

print("Test Overall MSE: {:.4f}.".format(test_metrics['mse']))
print("Test Overall PCC: {:.4f}.".format(test_metrics['pearson']))
# print epoch performance for DE genes
print("Test Top 20 DE MSE: {:.4f}.".format(test_metrics['mse_de']))
print("Test Top 20 DE PCC: {:.4f}.".format(test_metrics['pearson_de']))


# In[ ]:


import anndata as ann
from scipy.sparse import csr_matrix

ds = model.dataloader.dataloader['test_loader'].dataset  # list-like of Data
x_list = [data.x.squeeze() for data in ds]
ctrl = torch.stack(x_list, dim=0).numpy()

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


# In[ ]:


out = deeper_analysis(Data.adata, test_res)
out_result = {}
for m in list(list(out.values())[0].keys()):  # 첫 perturbation의 metric 키 기준
    out_result['test_' + m] = np.mean([j[m] for j in out.values() if m in j])


