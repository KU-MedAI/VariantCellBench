
import json
import os
import sys
import time
import argparse
import pickle
import warnings
from pathlib import Path
from collections import OrderedDict

import torch
import numpy as np
import pandas as pd
import anndata as ad
import torch.nn as nn
from torchtext.vocab import Vocab
from torchtext._torchtext import Vocab as VocabPybind
from torch_geometric.loader import DataLoader

# scGPT 관련 import (환경에 맞게 경로 설정 필요)
sys.path.insert(0, "../")
import scgpt as scg
from scgpt.model import TransformerGenerator
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import set_seed, map_raw_id_to_vocab_id

warnings.filterwarnings("ignore")

class FakePertData:
    def __init__(self, dataloader_dict, adata):
        self.adata = adata
        self.dataloader_dict = dataloader_dict

def build_expr_binarypert_x_list(adata, gene_to_idx, gene_names):
    x_tensor_list = []
    n_genes = len(gene_names)
    
    for i in range(adata.n_obs):
        expr = adata.X[i].toarray().flatten()
        pert_gene = adata.obs["gene"][i]
        pert_idx = gene_to_idx.get(pert_gene, -1)

        if pert_idx == -1:
            raise ValueError(f"Gene '{pert_gene}' not found in gene_names.")

        pert_flag = torch.zeros(n_genes, dtype=torch.float32)
        pert_flag[pert_idx] = 1.0
        x_i = torch.stack([torch.tensor(expr, dtype=torch.float32), pert_flag], dim=1)
        x_tensor_list.append(x_i)

    return x_tensor_list

def eval_perturb(loader, model, device, args, gene_ids, variant_vocab, variant_embs_proj):
    model.eval()
    model.to(device)
    pert_cat = []
    pred = []
    truth = []
    
    # Inference loop
    with torch.no_grad():
        for itr, batch in enumerate(loader):
            batch.to(device)
            pert_cat.extend(batch.pert)

            # Predict using all genes (or non-zero depending on args)
            p = model.pred_perturb(
                batch,
                include_zero_gene=args.include_zero_gene,
                gene_ids=gene_ids,
                variant_vocab=variant_vocab,
                variant_embs_proj=variant_embs_proj,
            ).cpu()

            t = batch.y.cpu()
            pred.extend(p)
            truth.extend(t)
            
            torch.cuda.empty_cache()

    results = {}
    results["pert_cat"] = np.array(pert_cat)
    results["pred"] = torch.stack(pred).numpy().astype(np.float32)
    results["truth"] = torch.stack(truth).numpy().astype(np.float32)
    return results

# =============================================================================
# Main Execution
# =============================================================================

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    print(f"Loading data from {args.dataloader_path}...")
    with open(args.dataloader_path, 'rb') as f:
        dataloader_dict = pickle.load(f)
    
    print(f"Loading AnnData from {args.adata_path}...")
    adata = ad.read_h5ad(args.adata_path)
    gene_names = list(adata.var_names)
    n_genes = len(gene_names)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    # Apply x tensor to dataloaders
    print("Building inputs...")
    x_tensor_list = build_expr_binarypert_x_list(adata, gene_to_idx, gene_names)
    for split in ["test_loader"]:
        loader = dataloader_dict[split]
        data_list = loader.dataset
        for i, data in enumerate(data_list):
            data.x = x_tensor_list[i]

    pert_data = FakePertData(dataloader_dict, adata)

    # --- Model & Vocab Setup ---
    special_tokens = [args.pad_token, "<cls>", "<eoc>"]
    
    # 1. Load Config from pretrained path (usually scGPT_human)
    if args.model_config_path:
        model_dir = Path(args.model_config_path)
        model_config_file = model_dir / "args.json"
        vocab_file = model_dir / "vocab.json"
        
        vocab = GeneVocab.from_file(vocab_file)
        for s in special_tokens:
            if s not in vocab: vocab.append_token(s)
            
        # Vocab matching
        pert_data.adata.var["id_in_vocab"] = [
            1 if gene in vocab else -1 for gene in pert_data.adata.var["gene_name"]
        ]
        gene_ids_in_vocab = np.array(pert_data.adata.var["id_in_vocab"])
        print(f"Match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes in vocabulary.")
        
        genes = pert_data.adata.var["gene_name"].tolist()
        
        with open(model_config_file, "r") as f:
            model_configs = json.load(f)
        embsize = model_configs["embsize"]
        nhead = model_configs["nheads"]
        d_hid = model_configs["d_hid"]
        nlayers = model_configs["nlayers"]
        n_layers_cls = model_configs["n_layers_cls"]
    else:
        # Fallback manual config
        genes = pert_data.adata.var["gene_name"].tolist()
        vocab = Vocab(VocabPybind(genes + special_tokens, None))
        embsize = args.embsize
        nhead = args.nhead
        d_hid = args.d_hid
        nlayers = args.nlayers
        n_layers_cls = args.n_layers_cls

    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array([vocab[gene] if gene in vocab else vocab["<pad>"] for gene in genes], dtype=int)

    # --- Variant Embeddings ---
    print(f"Loading variant embeddings from {args.pkl_path}...")
    unique_conditions = adata.obs['condition'].unique().tolist()
    variant_list_full_names = []
    for cond in unique_conditions:
        if cond == 'ctrl': continue
        variant_name = cond.split('+')[0]
        variant_list_full_names.append(variant_name)
    variant_list_full_names = sorted(list(set(variant_list_full_names)))
    
    short_to_full_map = {name.split('~')[1]: name for name in variant_list_full_names}
    short_to_full_map['REF'] = 'REF' # REF 키 추가
    
    variant_vocab = OrderedDict([("<pad>", 0), ("ctrl", 1), ("REF", 2)] + [(v, i + 3) for i, v in enumerate(variant_list_full_names)])
    
    with open(args.pkl_path, "rb") as f:
        cache = pickle.load(f)
        
    processed_embeddings = {}
    GENE_TARGET = "TP53"
    embedding_dim = None

    for (gene, variant_short_name), emb_dict in cache.items():
        if gene == GENE_TARGET:
            if variant_short_name in short_to_full_map:
                full_name = short_to_full_map[variant_short_name]
                if args.embedding_key in emb_dict:
                    raw_vector = emb_dict[args.embedding_key]
                    embedding_tensor = torch.tensor(raw_vector).float()
                    processed_embeddings[full_name] = embedding_tensor
                    if embedding_dim is None: embedding_dim = embedding_tensor.shape[0]

    if 'REF' not in processed_embeddings:
         if embedding_dim: processed_embeddings['REF'] = torch.zeros(embedding_dim) # Fallback
         else: raise ValueError("Could not determine embedding dimension from cache.")

    processed_embeddings["<pad>"] = torch.zeros(embedding_dim)
    processed_embeddings["ctrl"] = torch.zeros(embedding_dim)
    
    variant_emb_list = [None] * len(variant_vocab)
    for name, idx in variant_vocab.items():
        if name not in processed_embeddings:
            variant_emb_list[idx] = torch.zeros(embedding_dim)
        else:
            variant_emb_list[idx] = processed_embeddings[name]
    
    variant_embs_cpu = torch.stack(variant_emb_list)
    projector = nn.Linear(embedding_dim, embsize).to(device)
    
    # --- Load Checkpoint ---
    print(f"Loading checkpoint from {args.checkpoint_path}...")
    ntokens = len(vocab)
    model = TransformerGenerator(
        ntokens, embsize, nhead, d_hid, nlayers,
        nlayers_cls=n_layers_cls, n_cls=1, vocab=vocab,
        dropout=args.dropout, pad_token=args.pad_token,
        pad_value=args.pad_value, pert_pad_id=args.pert_pad_id,
        use_fast_transformer=True
    )
    
    # Load state dict
    state_dict = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    
    with torch.no_grad():
        variant_embs_proj = projector(variant_embs_cpu.to(device))

    # --- Inference ---
    print("Running inference on test_loader...")
    test_loader = pert_data.dataloader_dict["test_loader"]
    test_res = eval_perturb(
        test_loader, model, device, args, 
        gene_ids, variant_vocab, variant_embs_proj
    )

    # --- Post-processing & Saving ---
    print("\n--- Processing Results ---")

    original_adata = ad.read_h5ad(args.metadata_adata_path)

    print("Extracting control inputs from test_loader...")
    all_x_controls = []
    for batch_data in test_loader:
        batch_size = len(batch_data.pert)
        x_tensor = batch_data.x
        control_values = x_tensor[:, 0].view(batch_size, n_genes)
        all_x_controls.append(control_values)
    test_x_controls = torch.cat(all_x_controls, dim=0).cpu().numpy()

    obs_df = pd.DataFrame({'condition': test_res['pert_cat']})

    n_obs = len(obs_df)
    obs_df.index = [f"obs_{i}" for i in range(n_obs)]
    
    var_df = original_adata.var.copy()
    if 'gene_name' not in var_df.columns: var_df['gene_name'] = var_df.index
    
    if 'gene_ids_in_vocab' in locals():
        matched_gene_mask = (gene_ids_in_vocab != -1)
    else:
        matched_gene_mask = np.array([1 if g in vocab else -1 for g in var_df['gene_name']]) != -1

    var_df['exist'] = matched_gene_mask.astype(int)
        
    pred_adata = ad.AnnData(X=test_res['pred'], obs=obs_df.copy(), var=var_df.copy())
    truth_adata = ad.AnnData(X=test_res['truth'], obs=obs_df.copy(), var=var_df.copy())
    
    ctrl_obs_df = pd.DataFrame({'condition': [f"ctrl" for _ in range(len(test_x_controls))]})
    ctrl_obs_df.index = [f"ctrl_{i}" for i in range(len(ctrl_obs_df))]
    
    ctrl_adata = ad.AnnData(X=test_x_controls, obs=ctrl_obs_df, var=var_df) # Original vars
    
    print("Concatenating control samples...")
    final_pred_adata = ad.concat([pred_adata, ctrl_adata], join='outer', fill_value=0, label='source')
    final_pred_adata.obs['source'] = final_pred_adata.obs['source'].map({'0': 'pred', '1': 'ctrl_input'}).astype('category')
    
    final_truth_adata = ad.concat([truth_adata, ctrl_adata], join='outer', fill_value=0, label='source')
    final_truth_adata.obs['source'] = final_truth_adata.obs['source'].map({'0': 'truth', '1': 'ctrl_input'}).astype('category')
    
    final_pred_adata.var = var_df.copy()
    final_truth_adata.var = var_df.copy()

    print("Merging metadata...")
    metadata_cols = ['condition', 'gene', 'cell_type', 'variant_count', 'dose_val', 'control', 'condition_name']
    valid_cols = [c for c in metadata_cols if c in original_adata.obs.columns]
    metadata_df = original_adata.obs[valid_cols].drop_duplicates(subset=['condition']).copy()
    
    for adata_obj in [final_pred_adata, final_truth_adata]:
        idx = adata_obj.obs.index
        merged = pd.merge(adata_obj.obs, metadata_df, on='condition', how='left')
        merged.index = idx
        adata_obj.obs = merged
        
        adata_obj.uns = original_adata.uns.copy()
        
        for col in adata_obj.obs.columns:
            if adata_obj.obs[col].dtype.name == 'category': continue
            if adata_obj.obs[col].dtype == object:
                adata_obj.obs[col] = adata_obj.obs[col].astype(str)

    # --- Save ---
    os.makedirs(args.save_dir, exist_ok=True)
    pred_path = os.path.join(args.save_dir, f"{args.save_name}_pred.h5ad")
    
    final_pred_adata.write_h5ad(pred_path)
    
    print(f"✅ Saved prediction: {pred_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference scGPT with Variant Perturbations")

    # Paths
    parser.add_argument("--dataloader_path", type=str, required=True, help="Path to dataloader.pkl")
    parser.add_argument("--adata_path", type=str, required=True, help="Path to processed .h5ad")
    parser.add_argument("--metadata_adata_path", type=str, required=True, help="Path to metadata .h5ad")
    parser.add_argument("--pkl_path", type=str, required=True, help="Path to embedding pickle")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to best_model.pt")
    parser.add_argument("--model_config_path", type=str, default="../save/scGPT_human", help="Path containing args.json/vocab.json")
    parser.add_argument("--save_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--save_name", type=str, default="hct116_result", help="Prefix for saved files")

    # Options
    parser.add_argument("--embedding_key", type=str, default="DIFF", choices=["ALT", "DIFF"])
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--include_zero_gene", type=str, default="all")
    
    # Model Params (Should match training)
    parser.add_argument("--pad_token", type=str, default="<pad>")
    parser.add_argument("--pad_value", type=int, default=0)
    parser.add_argument("--pert_pad_id", type=int, default=0)
    parser.add_argument("--embsize", type=int, default=512)
    parser.add_argument("--d_hid", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--nlayers", type=int, default=12)
    parser.add_argument("--n_layers_cls", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)

    args = parser.parse_args()
    main(args)