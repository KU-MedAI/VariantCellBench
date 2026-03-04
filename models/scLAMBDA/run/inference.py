'''
python inference.py \
    --emb_type DIFF \
    --gene_emb_path "/NFS_DATA/samsung/database/gears/embedding/embedding_cache_variant_position_[esm_msa1_t12_100M_UR50S].pkl" \
    --adata_path "/NFS_DATA/samsung/database/gears/kim2023_hct116_[benchmark][clinvar]/perturb_processed_[01].h5ad" \
    --ckpt_path "/NFS_DATA/samsung/scLAMBDA/251208/1_3-fold/hct116_MSA_DIFF/ckpt.pth" \
    --cell_type hct116 \
    --model_name msa \
    --fold 1-3 \
    --data_id 01
GPU 병렬 작업
conda activate rtd
./run_inference_v2.py

'''

import argparse
import pandas as pd
import scanpy as sc
import anndata as ad
import numpy as np
import os
import pickle
import torch
import gc
import re
# scLAMBDA 프로젝트 경로 설정 (환경에 맞게 수정)
import sys
sys.path.append("/home/tech/variantseq/eugenie/scLAMBDA/variant_scLAMBDA")
import sclambda

try:
    from sclambda.network import Net
except ImportError:
    try:
        from sclambda.model import Net
    except ImportError:
        raise ImportError("Cannot import 'Net' class from sclambda. Please check the package structure.")

OUTPUT_DIR = "/NFS_DATA/samsung/database/benchmark_figure/ann_dataset_oncoKB/scLAMBDA" # 변경

def inference(args):
    print("--- 1. Loading data and gene embeddings ---")

    gene_embeddings = pd.read_pickle(args.gene_emb_path)
    target_adata = ad.read_h5ad(args.adata_path)
    
    if 'gene_name' in target_adata.var.columns:
        target_adata.var.index = target_adata.var['gene_name']
    
    final_adata = target_adata.copy()
    
    if hasattr(final_adata.X, 'toarray'):
        final_adata.X = final_adata.X.toarray()
    else:
        final_adata.X = np.asarray(final_adata.X)

    print(f"Template loaded. Total cells: {final_adata.n_obs}")
    print(f"\n--- 2. Initializing scLAMBDA model for Inference ---")
    
    model = sclambda.model.Model(
        target_adata, 
        gene_embeddings,
        model_path=args.model_path, 
        training_epochs=args.epochs,
        batch_size=args.batch_size,
        wandb_run=None,
        split_name='split',
        emb_type=args.emb_type
    )
    
    print(f"Loading checkpoint manually from {args.ckpt_path}...")
    try:
        model.Net = Net(
            x_dim=model.x_dim, 
            p_dim=model.p_dim, 
            latent_dim=model.latent_dim, 
            hidden_dim=model.hidden_dim
        )
        
        checkpoint = torch.load(args.ckpt_path, map_location=model.device)
        
        if isinstance(checkpoint, dict) and 'Net' in checkpoint:
            model.Net.load_state_dict(checkpoint['Net'])
        else:
            model.Net.load_state_dict(checkpoint)

        model.Net.to(model.device)
        model.Net.eval()
        print("✅ Model weights loaded successfully.")
        
    except Exception as e:
        print(f"❌ Failed to load model weights: {e}")
        return

    print("\n--- 3. Predicting and Filling Template ---")

    if 'split' in target_adata.obs.columns:
        target_mask = (target_adata.obs['split'] == 'test') & (target_adata.obs['condition'] != 'ctrl')
    else:
        target_mask = target_adata.obs['condition'] != 'ctrl'
        
    target_conditions = target_adata.obs[target_mask]['condition'].unique().tolist()
    print(f"Found {len(target_conditions)} conditions to predict.")
    
    valid_conditions = []
    
    print("Validating conditions against embedding keys...")
    for cond in target_conditions:

        try:

            parts = cond.split('+')
            target_variant = parts[0] 
            
            if '~' in target_variant:
                gene_name, mutation = target_variant.split('~')
                lookup_key = (gene_name, mutation)
                
                if lookup_key in gene_embeddings:
                    valid_conditions.append(cond)
            else:
                continue
                
        except Exception:
            continue
            
    if not valid_conditions:
        print("Warning: No valid conditions found in embeddings to predict.")
        print(f"Example Target: {target_conditions[0] if target_conditions else 'None'}")
        print(f"Example Key: {list(gene_embeddings.keys())[0] if gene_embeddings else 'None'}")
        return

    print(f"Proceeding with {len(valid_conditions)} valid conditions.")

    try:
        preds_dict = model.predict(valid_conditions, return_type='cells')
    except Exception as e:
        print(f"Prediction failed: {e}")
        return

    fill_count = 0
    
    for cond in valid_conditions:
        if cond not in preds_dict:
            continue
            
        generated_X = preds_dict[cond].X
        n_generated = generated_X.shape[0]
        
        cond_indices = np.where((target_adata.obs['condition'] == cond) & target_mask)[0]
        n_needed = len(cond_indices)
        
        if n_needed == 0:
            continue

        if n_generated >= n_needed:

            selected_X = generated_X[:n_needed]
        else:

            choice_indices = np.random.choice(n_generated, size=n_needed, replace=True)
            selected_X = generated_X[choice_indices]
            
        final_adata.X[cond_indices] = selected_X
        fill_count += 1
        
    print(f"Successfully filled predictions for {fill_count} conditions.")
    print("Note: 'ctrl' samples remain unchanged as context.")


    print("--- 4. Saving Results ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    filename = f"1127_2142_{args.cell_type}_{args.model_name}_{args.emb_type.lower()}_{args.fold}_{args.data_id}_pred.h5ad"
    save_path = os.path.join(OUTPUT_DIR, filename)
    
    for col in final_adata.obs.columns:
        if final_adata.obs[col].dtype == 'object':
            final_adata.obs[col] = final_adata.obs[col].astype(str)

    print(f"Saving to: {save_path}")
    final_adata.write_h5ad(save_path)
    print("✅ Process Complete.")

    del final_adata, target_adata, model, preds_dict
    gc.collect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    

    parser.add_argument('--emb_type', type=str, required=True, choices=['ALT', 'DIFF'])
    parser.add_argument('--gene_emb_path', type=str, required=True)
    parser.add_argument('--adata_path', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, default='./output')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--cell_type', type=str, required=True, help='예: hct116')
    parser.add_argument('--model_name', type=str, required=True, help='예: msa, esm2')
    parser.add_argument('--fold', type=str, required=True, help='예: 02, 1-3')
    parser.add_argument('--data_id', type=str, required=True, help='예: 02, 01')

    args = parser.parse_args()
    inference(args)