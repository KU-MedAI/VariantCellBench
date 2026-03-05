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
import datetime
import random
import torch
import sys
import gc
import jax
import jax.numpy as jnp
from tqdm import tqdm

from cellflow.model import CellFlow
from cellflow.preprocessing import reconstruct_pca
from cellflow.utils import match_linear
from cellflow.training import Metrics, ComputationCallback

from variant_emb import format_variant_dict_strict
from data_process import unroll_and_align_data


# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EarlyStoppingException(Exception):
    def __init__(self, message, stopped_epoch, best_score, best_epoch):
        super().__init__(message)
        self.stopped_epoch = stopped_epoch
        self.best_score = best_score
        self.best_epoch = best_epoch


class MetricsAndSave(ComputationCallback):
    def __init__(self, metrics_callback, model, save_dir, monitor="validation_mmd_mean", mode="min", patience=10, start_delay=1):
        self.metrics_callback = metrics_callback
        self.model = model                       
        self.save_dir = save_dir      
        self.monitor = monitor
        self.mode = mode
        self.best_score = float('inf') if mode == "min" else -float('inf')
        self.patience = patience
        self.start_delay = start_delay
        self.wait = 0
        self.current_epoch = 0
        self.best_epoch = 0 

    def on_log_iteration(self, valid_source_data, valid_true_data, valid_pred_data, solver):
        self.current_epoch += 1
        logs = self.metrics_callback.on_log_iteration(valid_source_data, valid_true_data, valid_pred_data, solver)    
        
        if not isinstance(logs, dict): return logs
        
        # Clean JAX/Numpy types for logging
        logs = {k: (float(v.item()) if hasattr(v, 'item') else v) for k, v in logs.items()}
        current_score = logs.get(self.monitor)

        if current_score is None: return logs
        
        gc.collect() 
        jax.clear_caches()

        if self.current_epoch <= self.start_delay:
            logger.info(f"[Epoch {self.current_epoch}] Warm-up phase. Score: {current_score:.4f}.")
            return logs

        improved = (current_score < self.best_score) if self.mode == "min" else (current_score > self.best_score)

        if improved:
            logger.info(f"[Epoch {self.current_epoch}] Best {self.monitor} updated: {current_score:.4f}")
            self.best_score = current_score
            self.best_epoch = self.current_epoch
            self.wait = 0
            os.makedirs(self.save_dir, exist_ok=True)
            self.model.save(self.save_dir, overwrite=True)
        else:
            self.wait += 1
            logger.info(f"[Epoch {self.current_epoch}] No improvement. Patience: {self.wait}/{self.patience}")
            if self.wait >= self.patience:
                raise EarlyStoppingException("Early stopping triggered", self.current_epoch, self.best_score, self.best_epoch)
            
        return logs

    def on_train_begin(self):
        pass

    def on_train_end(self, valid_source_data, valid_true_data, valid_pred_data, solver):
        return self.metrics_callback.on_train_end(valid_source_data, valid_true_data, valid_pred_data, solver)
    
    
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
        "train_path": os.path.join(args.data_dir, f"{c_line}_train_{fold}.h5ad"),
        "val_path": os.path.join(args.data_dir, f"{c_line}_valid_{fold}.h5ad"),
        "test_path": os.path.join(args.data_dir, f"{c_line}_test_{fold}.h5ad"),
        "master_adata": os.path.join(args.data_dir, "master", f"perturb_processed_{c_line}.h5ad"),
        "model_save_dir": os.path.join(args.output_dir, "checkpoints", run_id),
        "result_save_dir": os.path.join(args.output_dir, "results", run_id),
        "run_id": run_id
    }



def run_training(args):
    paths = get_paths(args)
    logger.info(f"Starting Training: {paths['run_id']}")

    emb_mode = 'ALT' if args.emb_type == 'alt' else 'DIFF'
    emb_dict = format_variant_dict_strict(paths['pkl_path'], mode=emb_mode)

    adata_train, pca_model = unroll_and_align_data(paths['train_path'], emb_dict, pca_params=None, n_comps=50)
    adata_val, _ = unroll_and_align_data(paths['val_path'], emb_dict, pca_params=pca_model)

    cf = CellFlow(adata_train, solver="otfm")
    cf.prepare_data(
        sample_rep="X_pca",
        control_key="is_control",
        perturbation_covariates={"variant_pert": ("variant_input",)}, 
        perturbation_covariate_reps={"variant_pert": "variant_embeddings"},
        max_combination_length=1
    )

    cf.prepare_validation_data(adata_val, name="validation")

    match_fn = functools.partial(match_linear, epsilon=0.1, tau_a=0.9, tau_b=0.9)
    layers_before_pool = {"variant_pert": {"layer_type": "mlp", "dims": [512, 256], "dropout_rate": 0.2}}
    layers_after_pool = {"layer_type": "mlp", "dims": [128], "dropout_rate": 0.1}

    cf.prepare_model(
        condition_mode="deterministic",
        regularization=0.3,
        pooling="attention_token",   
        layers_before_pool=layers_before_pool,
        layers_after_pool=layers_after_pool,
        condition_embedding_dim=128,
        cond_output_dropout=0.2,
        hidden_dims=[512, 512, 512],
        decoder_dims=[1024, 1024],
        time_freqs=128,              
        time_encoder_dims=[256, 256], 
        probability_path={"constant_noise": 0.1}, 
        match_fn=match_fn,
        optimizer=optax.MultiSteps(optax.adam(learning_rate=1e-5), every_k_schedule=5),
    )

    os.makedirs(paths['model_save_dir'], exist_ok=True)
    with open(os.path.join(paths['model_save_dir'], "pca_model.pkl"), "wb") as f:
        pickle.dump(pca_model, f)

    metrics_cb  = Metrics(metrics=["mmd", "e_distance"])
    train_cb = MetricsAndSave(
        metrics_callback=metrics_cb ,
        model=cf,
        save_dir=paths['model_save_dir'],
        monitor="validation_mmd_mean",
        patience=args.patience,
        start_delay=5
    )

    try:
        cf.train(num_iterations=args.epochs * 1000, batch_size=args.batch_size, callbacks=[train_cb], valid_freq=1000)
    except EarlyStoppingException as e:
        logger.info(f"Early Stopping at epoch {e.stopped_epoch}. Best score: {e.best_score:.4f}")  
        
    logger.info("Training Complete.")

def run_inference(args):
    
    paths = get_paths(args)
    logger.info(f"Starting Inference: {paths['run_id']}")

    with open(os.path.join(paths['model_save_dir'], "pca_model.pkl"), "rb") as f:
        pca_model = pickle.load(f)
    pca_mean, pca_components = pca_model

    emb_mode = 'ALT' if args.emb_type == 'alt' else 'DIFF'
    emb_dict = format_variant_dict_strict(paths['pkl_path'], mode=emb_mode)

    adata_test, _ = unroll_and_align_data(paths['test_path'], emb_dict, pca_params=pca_model)    
    adata_master = sc.read_h5ad(paths['master_adata'])
    master_gene_list = adata_master.var_names.tolist()

    cf = CellFlow.load(paths['model_save_dir']) 
    if hasattr(cf, "solver"): cf.solver.is_trained = True
    cf.is_trained = True

    mask_ctrl = adata_test.obs['is_control'].to_numpy(dtype=bool, na_value=False)
    adata_test_ctrl = adata_test[mask_ctrl].copy()
    adata_test_pert = adata_test[~mask_ctrl].copy()

    covariate_data = adata_test_pert.obs.drop_duplicates(subset=["variant_input"]).copy()
    covariate_data['variant_pert'] = covariate_data['variant_input']

    preds_pca_dict = cf.predict(
        adata=adata_test_ctrl,
        sample_rep="X_pca",
        condition_id_key="variant_pert", 
        covariate_data=covariate_data
    )

    list_pred_X, list_obs_indices, list_conditions_raw = [], [], []

    unique_conditions = adata_test_pert.obs['variant_input'].unique()
    for cond in tqdm(unique_conditions, desc="Reconstructing"):
 
        mask = adata_test_pert.obs['variant_input'] == cond
        real_cells = adata_test_pert[mask]
        n_cells = real_cells.n_obs        
        list_truth_X.append(real_cells.X)
        list_obs_indices.extend(real_cells.obs_names)
        list_conditions_raw.extend([cond] * n_cells)

        pred_pool_pca = preds_pca_dict[cond]
        idx = np.random.choice(len(pred_pool_pca), n_cells, replace=(n_cells > len(pred_pool_pca)))
        sampled_pca = pred_pool_pca[idx]
        
        temp_recon_ad = ad.AnnData(X=np.zeros((n_cells, len(adata_test.var_names))))
        temp_recon_ad.obsm["X_pca"] = sampled_pca
        reconstruct_pca(temp_recon_ad, use_rep="X_pca", ref_means=pca_mean, ref_pcs=pca_components, layers_key_added="X_recon")
        list_pred_X.append(temp_recon_ad.layers["X_recon"])

    df_pred_pert = pd.DataFrame(np.vstack(list_pred_X), index=list_obs_indices, columns=adata_test.var_names)
    
    master_ctrl_mask = adata_master.obs['condition'] == 'ctrl'
    adata_master_ctrl = adata_master[master_ctrl_mask].copy()
    df_ctrl = pd.DataFrame(adata_master_ctrl.X, index=adata_master_ctrl.obs_names, columns=adata_master.var_names)

    df_pred_pert = df_pred_pert.reindex(columns=master_gene_list, fill_value=0)
    df_ctrl = df_ctrl.reindex(columns=master_gene_list, fill_value=0)

    df_pred_final = pd.concat([df_pred_pert, df_ctrl])

    master_unique_obs = adata_master.obs[['condition', 'condition_name']].drop_duplicates()
    cond_map = {row['condition'].split('+')[0].replace('~', '_'): row['condition'] for _, row in master_unique_obs.iterrows()}
    name_map = {row['condition'].split('+')[0].replace('~', '_'): row['condition_name'] for _, row in master_unique_obs.iterrows()}

    obs_pert = pd.DataFrame(index=list_obs_indices)
    obs_pert['condition'] = pd.Series(list_cond_raw, index=list_obs_indices).map(cond_map).fillna([f"{c}+ctrl" for c in list_cond_raw])
    obs_pert['condition_name'] = pd.Series(list_cond_raw, index=list_obs_indices).map(name_map).fillna([f"{args.cell_line}_{c}+ctrl_1+1" for c in list_cond_raw])
    obs_ctrl = adata_master_ctrl.obs.copy()
    final_obs = pd.concat([obs_pert, obs_ctrl])

    final_var = adata_master.var.copy()
    final_uns = adata_master.uns.copy()

    adata_pred = ad.AnnData(X=df_pred_final.values, obs=final_obs, var=final_var, uns=final_uns)

    res_dir = paths['result_save_dir']; os.makedirs(res_dir, exist_ok=True)
    adata_pred.write_h5ad(os.path.join(res_dir, f"{paths['run_id']}_pred.h5ad"))

    logger.info(f"Inference Successfully Completed. Results in: {res_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference', 'pipeline'])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--emb_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plm", type=str, default="esm2")
    parser.add_argument("--cell_line", type=str, default="u2os")
    parser.add_argument("--emb_type", type=str, default="alt")
    parser.add_argument("--num_fold", type=str, default="1-3")

    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.mode == 'train':
        run_training(args)
    elif args.mode == 'inference':
        run_inference(args)
    elif args.mode == 'pipeline':
        run_training(args)
        run_inference(args)