
import os
import sys
import gc
# scLAMBDA 프로젝트 경로 설정 (환경에 맞게 수정)
sys.path.append("/home/tech/variantseq/eugenie/scLAMBDA/variant_scLAMBDA")
import argparse
import pandas as pd
import anndata as ad
from anndata import AnnData
from typing import Dict
import numpy as np
import scanpy as sc
import sclambda
import warnings
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error as mse
import wandb
import pickle
import torch


def main(args):
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config={ "epochs": args.epochs, "batch_size": args.batch_size, "emb_type": args.emb_type }
    )
    # --- 1. 데이터 준비 ---
    print("--- 1. Loading data and gene embeddings ---")

    gene_embeddings = pd.read_pickle(args.gene_emb_path)
    print(f"Loading pre-split AnnData from: {args.adata_path}")
    adata_processed = ad.read_h5ad(args.adata_path)
    adata_processed.var.index = adata_processed.var['gene_name']

    if 'split' not in adata_processed.obs.columns:
        raise ValueError(f"'split' column not found in {args.adata_path}. Please ensure the data is pre-split.")
    
    print("\nUsing existing 'split' column from AnnData object:")
    print(adata_processed.obs['split'].value_counts())
    
    model_full_path = args.model_path
    print(f"\n--- 2. Initializing and training scLAMBDA model ---")
    model = sclambda.model.Model(
        adata_processed, 
        gene_embeddings,
        model_path=model_full_path,
        training_epochs=args.epochs,
        batch_size=args.batch_size,
        wandb_run=run,
        split_name='split',
        emb_type=args.emb_type
    )
    model.train()
    print(f"\n--- Training complete. Model saved to {model_full_path} ---")

    print("\n--- 3. Starting evaluation ---")
    test_conditions = adata_processed[adata_processed.obs['split'] == 'test'].obs['condition'].unique().tolist()
    pert_test = [p for p in test_conditions if p != 'ctrl']
    print(f"Found {len(pert_test)} test perturbations to evaluate.")
    
    res = model.predict(pert_test, return_type='cells')

    print("\nLoading original AnnData to be used as ground truth...")
    try:
        original_adata = ad.read_h5ad(args.adata_path)
        print("Original AnnData loaded successfully.")
    except Exception as e:
        print(f"Error loading original AnnData file: {e}")
        return
    print("\nPreparing data for metrics calculation...")
    results_dict = {}
    all_preds, all_truths, all_perts = [], [], []
    for i in pert_test:
        if i not in res:
            print(f"Warning: No prediction found for condition '{i}'. Skipping.")
            continue
        prediction_adata = res[i]
        prediction_cells = prediction_adata.X
        true_cells_adata = original_adata[original_adata.obs['condition'] == i]
        if true_cells_adata.n_obs == 0:
            print(f"Warning: No cells found for condition '{i}' in the original AnnData. Skipping.")
            continue
    
        true_cells = true_cells_adata.X.toarray() if hasattr(true_cells_adata.X, 'toarray') else true_cells_adata.X

        num_true_cells = len(true_cells)
        all_preds.append(prediction_cells[:num_true_cells, :])
        all_truths.append(true_cells)
        
        pert_labels = [i] * num_true_cells
        all_perts.extend(pert_labels)
    if not all_perts:
        print("No test perturbations found to evaluate.")
        wandb.finish()
        return
    results_dict['pred'] = np.concatenate(all_preds, axis=0)
    results_dict['truth'] = np.concatenate(all_truths, axis=0)
    results_dict['pert_cat'] = np.array(all_perts)

    print("\n--- 4. Finalizing and saving AnnData objects ---")
    
    try:
        print("Extracting control samples from original AnnData...")

        ctrl_adata_subset = original_adata[original_adata.obs['condition'] == 'ctrl']
        
        if ctrl_adata_subset.n_obs == 0:
            raise ValueError("No control samples ('ctrl') found in the original AnnData.")

        test_x_controls = ctrl_adata_subset.X
        
        import scipy.sparse
        if scipy.sparse.issparse(test_x_controls):
            test_x_controls = test_x_controls.toarray()
        elif isinstance(test_x_controls, np.matrix):
            test_x_controls = np.asarray(test_x_controls)
            
        n_ctrl = test_x_controls.shape[0]
        print(f"Loaded {n_ctrl} control samples from AnnData.")

        var_df = original_adata.var.copy()
        if 'gene_name' not in var_df.columns:
            var_df['gene_name'] = var_df.index

        pred_array = results_dict['pred']
        truth_array = results_dict['truth']
        pert_cat_array = results_dict['pert_cat']

        pred_array = np.asarray(pred_array) 
        truth_array = np.asarray(truth_array)
        test_x_controls = np.asarray(test_x_controls)

        n_obs = pred_array.shape[0]
        
        obs_df = pd.DataFrame(
            {'condition': pert_cat_array},
            index=[f"pred_{i}" for i in range(n_obs)]
        )
        ctrl_obs_df = pd.DataFrame(
            {'condition': ['ctrl'] * n_ctrl},
            index=[f"ctrl_{i}" for i in range(n_ctrl)]
        )

        ctrl_adata = ad.AnnData(X=test_x_controls, obs=ctrl_obs_df, var=var_df)

        print("Processing Prediction AnnData...")
        pred_adata = ad.AnnData(X=pred_array, obs=obs_df.copy(), var=var_df)
        
        final_pred_adata = ad.concat([pred_adata, ctrl_adata], join='outer', fill_value=0, label='source')

        cats = final_pred_adata.obs['source'].cat.categories

        source_map = {cats[0]: 'pred', cats[1]: 'ctrl_input'}
        final_pred_adata.obs['source'] = final_pred_adata.obs['source'].map(source_map).astype('category')
        final_pred_adata.var = var_df
        
        del pred_adata
        gc.collect()

        print("Processing Truth AnnData...")
        truth_adata = ad.AnnData(X=truth_array, obs=obs_df.copy(), var=var_df)
        
        final_truth_adata = ad.concat([truth_adata, ctrl_adata], join='outer', fill_value=0, label='source')
        
        cats_t = final_truth_adata.obs['source'].cat.categories
        source_map_t = {cats_t[0]: 'truth', cats_t[1]: 'ctrl_input'}
        final_truth_adata.obs['source'] = final_truth_adata.obs['source'].map(source_map_t).astype('category')
        final_truth_adata.var = var_df

        del truth_adata, ctrl_adata, pred_array, truth_array, test_x_controls
        gc.collect()

        print("Merging metadata and Saving...")
        metadata_cols = ['condition', 'gene', 'cell_type', 'variant_count', 'dose_val', 'control', 'condition_name']
        valid_cols = [c for c in metadata_cols if c in original_adata.obs.columns]
        metadata_df = original_adata.obs[valid_cols].drop_duplicates(subset=['condition']).copy()

        os.makedirs(args.model_path, exist_ok=True)
        pred_path = os.path.join(args.model_path, "pred.h5ad")
        truth_path = os.path.join(args.model_path, "truth.h5ad")

        merged_obs = pd.merge(final_pred_adata.obs, metadata_df, on='condition', how='left')
        merged_obs.index = final_pred_adata.obs.index
        final_pred_adata.obs = merged_obs
        final_pred_adata.uns = original_adata.uns.copy()
        
        for col in final_pred_adata.obs.columns:
            if final_pred_adata.obs[col].dtype == 'object':
                final_pred_adata.obs[col] = final_pred_adata.obs[col].astype(str)

        if isinstance(final_pred_adata.X, np.matrix):
            print("Converting Pred X from matrix to ndarray...")
            final_pred_adata.X = np.asarray(final_pred_adata.X)
        else:
            final_pred_adata.X = np.asarray(final_pred_adata.X)

        print(f"Saving prediction to {pred_path}...")
        final_pred_adata.write_h5ad(pred_path)
        print("✅ Pred saved.")
        
        del final_pred_adata
        gc.collect()

        merged_obs_t = pd.merge(final_truth_adata.obs, metadata_df, on='condition', how='left')
        merged_obs_t.index = final_truth_adata.obs.index
        final_truth_adata.obs = merged_obs_t
        final_truth_adata.uns = original_adata.uns.copy()

        for col in final_truth_adata.obs.columns:
            if final_truth_adata.obs[col].dtype == 'object':
                final_truth_adata.obs[col] = final_truth_adata.obs[col].astype(str)

        if isinstance(final_truth_adata.X, np.matrix):
            print("Converting Truth X from matrix to ndarray...")
            final_truth_adata.X = np.asarray(final_truth_adata.X)
        else:
            final_truth_adata.X = np.asarray(final_truth_adata.X)

        print(f"Saving truth to {truth_path}...")
        final_truth_adata.write_h5ad(truth_path)
        print("✅ Truth saved.")

    except Exception as e:
        print(f"\n❌ An error occurred during AnnData processing: {e}")
        import traceback
        traceback.print_exc()
    
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train, evaluate, and save results for the scLAMBDA model.')
    
    parser.add_argument('--emb_type', type=str, choices=['ALT', 'DIFF'], required=True,
                        help="사용할 임베딩 타입: 'ALT' (변이 임베딩 직접 사용) 또는 'DIFF' (참조와의 차이 사용)")

    parser.add_argument('--gene_emb_path', type=str, required=True,
                        help='유전자 임베딩(.pkl) 파일 경로')
    parser.add_argument('--adata_path', type=str, required=True,
                        help='AnnData(.h5ad) 파일 경로')
    parser.add_argument('--model_path', type=str, required=True,
                        help='모델을 저장할 기본 폴더 경로')

    parser.add_argument('--epochs', type=int, default=100,
                        help='학습 에포크 수 (기본값: 100)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='배치 크기 (기본값: 16)')
    parser.add_argument('--wandb_project', type=str, default='sclambda-project', help='W&B 프로젝트 이름')
    parser.add_argument('--wandb_entity', type=str, required=True, help='W&B 사용자 또는 팀 이름')
    
    args = parser.parse_args()
    main(args)
