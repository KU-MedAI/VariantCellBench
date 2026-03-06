import os
import re
import pickle
import argparse
import itertools
import subprocess
import functools
import logging
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import optax
from sklearn.metrics import r2_score
from tqdm import tqdm
import datetime
import random
import torch
import sys
import gc
import jax
import jax.numpy as jnp

from cellflow.model import CellFlow
from cellflow.preprocessing import reconstruct_pca
from cellflow.utils import match_linear
from cellflow.training import Metrics
from cellflow.metrics import compute_e_distance
from cellflow.training import ComputationCallback

from variant_emb import format_variant_dict_strict
from data_process import unroll_and_align_data_unseen

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
    
def get_paths(args, cell_line=None, plm=None, emb_type=None, num_fold=None, run_id=None):

    c_line = (cell_line or args.cell_line).lower()
    p_model = plm or args.plm
    fold = num_fold or args.num_fold
    e_type = emb_type or args.emb_type
    
    emb_map = {
        'esm2': "embedding_cache_variant_position_[esm2_t33_650M_UR50D].pkl",
        'protT5': "embedding_cache_variant_position_[ProtT5-XXL-U50].pkl",
        'msa': "embedding_cache_variant_position_[esm_msa1_t12_100M_UR50S].pkl",
        'pglm': "embedding_cache_variant_position_[xTrimoPGLM-10B-MLM].pkl",
        'ankh': "embedding_cache_variant_position_[Ankh3-Large].pkl"
    }
        
    run_id = f"{c_line}_{p_model}_{e_type}_{fold}"
    
    return {
        "pkl_path": os.path.join(args.emb_dir, emb_map.get(p_model, "")),
        "master_adata": os.path.join(args.data_dir, "master", f"perturb_processed_{args.cell_line}.h5ad"),
        "result_save_dir": os.path.join(args.anndata_save, f"eval_{args.date}")
    }


def run_unseen_inference(args):
    
    paths = get_paths(args) 
    inference_root = paths['result_save_dir']
    os.makedirs(inference_root, exist_ok=True)

    checkpoint_root = args.train_result 
    logger.info(f"Searching for trained models in: {checkpoint_root}")

    model_dirs = [
        os.path.join(checkpoint_root, d) for d in os.listdir(checkpoint_root) 
        if os.path.isdir(os.path.join(checkpoint_root, d)) and "_ep" in d
    ]

    if not model_dirs:
        logger.error(f"No valid checkpoints found in {checkpoint_root}"); return

    for model_path in model_dirs:
        run_id = os.path.basename(model_path)

        regex = r'(\d{4}_\d{4})_([a-zA-Z0-9]+)_([a-zA-Z0-9]+)_([a-zA-Z0-9]+)_([0-9-]+)_ep\d+'
        match = re.search(regex, run_id)

        if not match:
            logger.warning(f"Could not parse run_id: {run_id}. Skipping.")
            continue

        timestamp, cell_line, plm, emb_type_str, fold = match.groups()

        pca_model_file = os.path.join(model_path, "pca_model.pkl")            
        with open(pca_model_file, "rb") as f:
            pca_model = pickle.load(f)
        pca_mean, pca_components = pca_model
        
        emb_path = get_paths(args, plm=plm)['pkl_path']
        emb_dict = format_variant_dict_strict(emb_path, mode=emb_type.upper())
        
        adata_master = sc.read_h5ad(paths['anndata_master'])
        master_gene_list = adata_master.var_names.tolist()

        cf = CellFlow.load(model_path)   

        if hasattr(cf, "solver"):
            cf.solver.is_trained = True
        if hasattr(cf, "is_trained"):
            cf.is_trained = True

        unseen_base_dir = args.unseen_dir
        chunk_indices = range(1, 15)    

        for chunk_idx in tqdm(chunk_indices, desc="Processing Unseen Chunks"):
            found_path = None
            found_str = ""
            
            for fmt in ["{:03d}", "{:02d}", "{}"]:
                chunk_str = fmt.format(chunk_idx)
                filename = f"perturb_processed_[{chunk_str}].h5ad"
                candidate_path = os.path.join(unseen_base_dir, filename)
                if os.path.exists(candidate_path):
                    found_path = candidate_path
                    found_str = chunk_str
                    break

            adata_source, _ = unroll_and_align_data_unseen(found_path, emb_dict, pca_params=pca_model)
            adata_source.obs_names_make_unique()

            if 'variant_input' in adata_source.obs.columns:
                adata_source.obs['variant_clean'] = adata_source.obs['variant_input'].astype(str).str.split('+').str[0]
            else:
                col = 'condition' if 'condition' in adata_source.obs else 'perturbation'
                adata_source.obs['variant_clean'] = adata_source.obs[col].astype(str).str.split('+').str[0]

            adata_source.uns['variant_embeddings'] = emb_dict

            covariate_data = adata_source.obs.drop_duplicates(subset=["variant_clean"]).copy()
            covariate_data['variant_pert'] = covariate_data['variant_clean']

            logger.info(f"Predicting {len(covariate_data)} conditions...")
            
            preds_pca_dict = cf.predict(
                adata=adata_source,
                sample_rep="X_pca",
                condition_id_key="variant_pert",  
                covariate_data=covariate_data
            )

            source_conditions = adata_source.obs['variant_clean'].values            
            reconstructed_pca_list = []            
            for cond in source_conditions:
                if cond in preds_pca_dict:
                    pred_pool = preds_pca_dict[cond]
                    rand_idx = np.random.randint(0, pred_pool.shape[0])
                    reconstructed_pca_list.append(pred_pool[rand_idx])
            
            final_pca_arr = np.array(reconstructed_pca_list)

            temp_recon_ad = ad.AnnData(X=np.zeros((len(final_pca_arr), len(adata_source.var_names))))
            temp_recon_ad.obsm["X_pca"] = final_pca_arr
            reconstruct_pca(temp_recon_ad, use_rep="X_pca", ref_means=pca_mean, ref_pcs=pca_components, layers_key_added="X_recon")

            import scipy.sparse
            input_vals = adata_source.X
            if scipy.sparse.issparse(input_vals):
                input_vals = input_vals.toarray()
            
            pred_vals = temp_recon_ad.layers["X_recon"]
            if scipy.sparse.issparse(pred_vals):
                pred_vals = pred_vals.toarray()

            df_pred_pert = pd.DataFrame(pred_vals, index=adata_source.obs_names, columns=adata_source.var_names)            
            df_pred = df_pred_pert.reindex(columns=master_gene_list, fill_value=0)

            var_data = pd.DataFrame(index=master_gene_list)
            final_obs = adata_source.obs.copy()
            final_layers = adata_source.layers.copy()
            adata_out_pred = ad.AnnData(X=df_pred.values, obs=final_obs, var=var_data, layers=final_layers)

            filename_base = f"{timestamp}_{cell_line}_{plm}_{emb_type_str}_{fold}_{chunk_str}"
            adata_out_pred.write_h5ad(os.path.join(inference_root, f"{filename_base}_pred.h5ad"))

    logger.info("Process Completed Successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True, help="데이터셋 루트 경로")
    parser.add_argument("--emb_dir", type=str, required=True, help="임베딩 파일 경로")
    parser.add_argument("--unseen_dir", type=str, required=True, help="추론할 Unseen 데이터셋 경로")
    parser.add_argument("--train_result", type=str, required=True, help="체크포인트가 저장된 경로")
    parser.add_argument("--anndata_save", type=str, default="./eval_results")
    parser.add_argument("--date", type=str, required=True, help="버전 관리용 날짜 태그")

    parser.add_argument("--cell_line", type=str, default="u2os")
    parser.add_argument("--plm", type=str, default="esm2")
    parser.add_argument("--emb_type", type=str, default="alt")

    args = parser.parse_args()