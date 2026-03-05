import os
import mindspore as ms
import glob
import time
import math
import ast
import datetime
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import itertools
import pickle
import mindspore.numpy as mnp
import mindspore.scipy as msc
import mindspore.dataset as ds
from tqdm import tqdm,trange
from mindspore import nn,ops
from scipy.sparse import csr_matrix as csm
from mindspore.amp import FixedLossScaleManager,all_finite,DynamicLossScaleManager
from mindspore.train import Model, CheckpointConfig, ModelCheckpoint, LossMonitor
from mindspore.communication import init, get_rank, get_group_size
from config import Config
from utils import Wrapper,WrapperWithLossScaleCell,BestModelSaver
from utils import WarmCosineDecay,Adam,AdamWeightDecay,set_weight_decay
from model import CellFM_gene
from data_process_unseen import SCrna,input_gene_filtering
from variant_emb import *
from loss_function import *
import sys
import subprocess


def set_seed(seed=42):
    ms.set_seed(seed)

    import random
    random.seed(seed)
    np.random.seed(seed)


def run_inference(args):
    set_seed(args.seed)

    ms.set_context(device_target='GPU', mode=ms.GRAPH_MODE, device_id=args.npu)
    cfg = Config()

    BATCH_SIZE = getattr(args, 'batch_size', 120) 
    print(f"Inference Batch Size: {BATCH_SIZE}")

    checkpoint_dir = os.path.join(args.modelpath, f'checkpointB_{args.date}')
    datapath = args.datapath
    savedata = "/NFS_DATA/samsung/database/benchmark_figure/ann_dataset_oov/CellFM/"
    
    os.makedirs(savedata, exist_ok=True)
    
    checkpoint_pattern = os.path.join(checkpoint_dir, "*.ckpt")
    models_to_test = glob.glob(checkpoint_pattern)

    if not models_to_test:
        print(f"ERROR: No checkpoints found matching '{checkpoint_pattern}'. Please check the path.")
        return

    print(f"--- Found {len(models_to_test)} checkpoints to process ---")
    for ckpt in models_to_test:
        print(f" - {os.path.basename(ckpt)}")
    print("-" * 50)

    for ckpt_path in models_to_test:
        print(f"\n\n{'='*20} [Processing Model: {os.path.basename(ckpt_path)}] {'='*20}")

        base_name_no_ext = os.path.splitext(os.path.basename(ckpt_path))[0]
        regex = r'(\d{4}_\d{4})_(hct116|u2os)_(protT5|msa|pglm|ankh|esm2)_(alt|diff)_(1-3|2-3|3-3)_ep\d+-best-\d+'
        match = re.search(regex, base_name_no_ext)

        if not match:
            print(f"[WARNING] Checkpoint name '{base_name_no_ext}' does not match expected format. Skipping.")
            continue
        
        date_time, data_name, emb_name, emb_type, num_fold = match.groups()
        print(f"Parsed Info -> Data: {data_name}, Embedding: {emb_name}, Fold: {num_fold}")

        print("Loading corresponding data and variant embeddings...")
        if emb_name == 'esm2':
            variant_pkl = "/NFS_DATA/samsung/database/gears/embedding/embedding_cache_variant_position_[esm2_t33_650M_UR50D].pkl"
        elif emb_name == 'protT5':
            variant_pkl = "/NFS_DATA/samsung/database/gears/embedding/embedding_cache_variant_position_[ProtT5-XXL-U50].pkl"
        elif emb_name == 'msa':
            variant_pkl = "/NFS_DATA/samsung/database/gears/embedding/embedding_cache_variant_position_[esm_msa1_t12_100M_UR50S].pkl"
        elif emb_name == 'pglm':
            variant_pkl = "/NFS_DATA/samsung/database/gears/embedding/embedding_cache_variant_position_[xTrimoPGLM-10B-MLM].pkl"
        elif emb_name == 'ankh':
            variant_pkl = "/NFS_DATA/samsung/database/gears/embedding/embedding_cache_variant_position_[Ankh3-Large].pkl"

        with open(variant_pkl, "rb") as f:
            variant_dict = pickle.load(f)
        variant_dim = infer_variant_dim_from_pkl(variant_dict)
        
        clinvar_dir = f"/NFS_DATA/samsung/database/gears/kim2023_{data_name}_[benchmark][oov]"
        for chunk_i in tqdm(range(1, 24), desc = "Unseen Chunks"):

            chunk_str = f"{chunk_i:02d}"

            ts_data_file = os.path.join(clinvar_dir, f"perturb_processed_[{chunk_str}].h5ad")
            
            if not os.path.exists(ts_data_file):
                print(f"[SKIP] Chunk {chunk_str} not found at {ts_data_file}")
                continue

            print(f"\n   >>> Processing Chunk [{chunk_str}/50] : {os.path.basename(ts_data_file)}")

            scrna_ts = SCrna(datapath, ts_data_file, embedding_type=emb_type, variant_dict=variant_dict, variant_dim=variant_dim, pool='mean')

            tp53_pos_idx = scrna_ts.tp53_pos_idx
            tp53_gene_id = scrna_ts.tp53_id

            if tp53_pos_idx is not None:
                tp53_gene_id = int(scrna_ts.gene[tp53_pos_idx])
                print(f"[INFO] TP53 Pos Index: {tp53_pos_idx}, Real Gene ID: {tp53_gene_id}")
            else:
                tp53_gene_id = None
                print("[WARNING] TP53 not found in dataset!")

            print(f"Initializing model for variant dim: {variant_dim}")
            model = CellFM_gene(len(scrna_ts.geneset), cfg, variant_dim=variant_dim, tp53_idx=tp53_gene_id)
            
            print(f"Loading weights from: {ckpt_path}")
            para = ms.load_checkpoint(ckpt_path)
            ms.load_param_into_net(model, para)
            model.set_train(False)
            model.to_float(ms.float32)
            
            print("Starting prediction ...")

            all_gene_ids_filtered = scrna_ts.gene
            n_genes_to_predict = len(all_gene_ids_filtered)
            n_cells_to_test = len(scrna_ts)
            chunk_size = cfg.nonz_len

            all_gw_preds_full = np.zeros((n_cells_to_test, n_genes_to_predict), dtype=np.float32)
            all_cw_preds_full = np.zeros((n_cells_to_test, n_genes_to_predict), dtype=np.float32)

            num_batches = int(np.ceil(n_cells_to_test / BATCH_SIZE))

            for b_idx in tqdm(range(num_batches), desc="Predicting Batches"):
                start_idx = b_idx * BATCH_SIZE
                end_idx = min(start_idx + BATCH_SIZE, n_cells_to_test)
                current_batch_size = end_idx - start_idx
                
                batch_variant_np = np.stack(scrna_ts.variant_cls_raw[start_idx:end_idx])
                variant_ms = ms.Tensor(batch_variant_np, dtype=ms.float32)
                
                batch_X_raw = scrna_ts.X[start_idx:end_idx]
                if hasattr(batch_X_raw, "toarray"):
                    batch_X_np = batch_X_raw.toarray()
                else:
                    batch_X_np = batch_X_raw
                
                for i in range(0, n_genes_to_predict, chunk_size):
                    gene_id_chunk = all_gene_ids_filtered[i : i + chunk_size]
                    current_len = len(gene_id_chunk)
                    if current_len == 0:
                        continue

                    input_gene_batch = np.zeros((current_batch_size, chunk_size), dtype=np.int32)
                    input_gene_batch[:, :current_len] = gene_id_chunk 

                    value_chunk_batch = batch_X_np[:, i : i + chunk_size]
                    input_x_data_batch = np.zeros((current_batch_size, chunk_size), dtype=np.float32)
                    input_x_data_batch[:, :current_len] = value_chunk_batch

                    input_zidx_batch = np.zeros((current_batch_size, chunk_size + 1), dtype=np.float32)
                    input_zidx_batch[:, 0] = 1.0
                    input_zidx_batch[:, 1 : current_len + 1] = 1.0

                    gene_ms = ms.Tensor(input_gene_batch)
                    x_data_ms = ms.Tensor(input_x_data_batch)
                    zidx_ms = ms.Tensor(input_zidx_batch)

                    gw_pred, cw_pred = model(None, x_data_ms, gene_ms, None, zidx_ms, variant_ms)
                    
                    gw_pred_np = gw_pred.asnumpy() 
                    cw_pred_np = cw_pred.asnumpy()
                    
                    all_gw_preds_full[start_idx:end_idx, i : i + current_len] = gw_pred_np[:, :current_len]
                    all_cw_preds_full[start_idx:end_idx, i : i + current_len] = cw_pred_np[:, :current_len]

            if num_fold:
                fold = num_fold.replace('-', '_') 
            else:
                fold = "unknown"

            master_template_path = f"/NFS_DATA/samsung/database/gears/kim2023_{data_name}_[benchmark][{fold}-fold]/perturb_processed_metadata.h5ad"       
            adata_master = sc.read_h5ad(master_template_path) 
            adata_master = adata_master[adata_master.obs['split']=='test'].copy()
            master_gene_list = adata_master.var_names.tolist()            
            predicted_gene_list = scrna_ts.adata.var_names.tolist()        
        
            variants = scrna_ts.adata.obs['condition']
            counts = variants.groupby(variants, observed=False).cumcount()
            scrna_index = variants.astype(str) + '_' + counts.astype(str)

            df_gw_pred_test = pd.DataFrame(all_gw_preds_full, index=scrna_index, columns=predicted_gene_list).reindex(columns=master_gene_list, fill_value=0)            
            df_cw_pred_test = pd.DataFrame(all_cw_preds_full, index=scrna_index, columns=predicted_gene_list).reindex(columns=master_gene_list, fill_value=0)    

            df_gw_pred_final = df_gw_pred_test
            df_cw_pred_final = df_cw_pred_test

            obs_with_names = adata_master.obs[['condition', 'condition_name']].copy()
            obs_with_names['clean_key'] = obs_with_names['condition'].str.split('+').str[0]  ## tp53~mut+ctrl
            condition_map = obs_with_names.drop_duplicates(subset=['clean_key']).set_index('clean_key')['condition_name']

            obs_test = pd.DataFrame(index=scrna_index)
            obs_test['condition'] = scrna_ts.adata.obs['variant'].values
            mapping_keys = obs_test['condition'].str.rsplit('_', n=1).str[0]
            obs_test['condition_name'] = mapping_keys.map(condition_map)
    
            final_obs = obs_test
            final_var = adata_master.var.copy()            
            final_uns = adata_master.uns.copy()
            
            adata_gw_pred = sc.AnnData(X=df_gw_pred_test.values, obs=final_obs, var=final_var, uns=final_uns)
            adata_cw_pred = sc.AnnData(X=df_cw_pred_final.values, obs=final_obs, var=final_var, uns=final_uns)

            filename_gw = f"{date_time}_{data_name}_{emb_name}_{emb_type}_gw_{num_fold}_{chunk_str}"
            filename_cw = f"{date_time}_{data_name}_{emb_name}_{emb_type}_cw_{num_fold}_{chunk_str}"

            gw_pred_path = os.path.join(savedata, f"{filename_gw}.h5ad")
            cw_pred_path = os.path.join(savedata, f"{filename_cw}.h5ad")
            
            adata_gw_pred.write_h5ad(gw_pred_path)
            adata_cw_pred.write_h5ad(cw_pred_path)

            csv_path = os.path.join(savedata, 'csv')
            os.makedirs(csv_path,  exist_ok=True)

            print(f"Successfully saved AnnData objects to '{savedata}'")


if __name__ == "__main__":
    pwd=os.getcwd()

    parser = argparse.ArgumentParser()

    parser.add_argument('--npu', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batch', type=int, default=8)        
    
    parser.add_argument('--workpath', type=str, default=f'{pwd}')
    parser.add_argument('--datapath', type=str, default='/home/tech/variantseq/DATASETS')
    parser.add_argument('--modelpath', default='/NFS_DATA/samsung/CellFM')
    parser.add_argument('--date', type=int, required=True, help='1031, 1103 etc')
    parser.add_argument('--num_fold', type=str)

    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
        
        