#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import random
import argparse
import logging
import json
import datetime
import shutil
import re
import glob
import sys
import subprocess
import itertools

import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig 

from variant_emb import VariantEmbeddingGenerator, infer_variant_dim_from_pkl
from data_process_unseen import AnnDataDataset, PerturbationDataCollator, get_ensembl_gene_mapping
from loss import compute_metrics
from transformers import BertConfig, TrainingArguments, Trainer, EarlyStoppingCallback
from genecompass.modeling_bert import GeneCompass_gene
from genecompass.utils import load_prior_embedding
from visualization import train_valid_lossplot

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_inference(args, device):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    anndata_dir = "/home/tech/variantseq/eval_data_oov/"
    os.makedirs(anndata_dir, exist_ok=True)

    model_search_root = os.path.join(args.finetuned_model, f"checkpoint_{args.date}")
    logger.info(f"Searching for models in root directory: {model_search_root}")

    models_to_test = []
    if not os.path.isdir(model_search_root):
        logger.error(f"ERROR: Model directory '{model_search_root}' not found. Please check the path and date.")
        return
    
    for dirname in os.listdir(model_search_root):
        dirpath = os.path.join(model_search_root, dirname)
        if os.path.isdir(dirpath) and re.search(r'ep\d+-\d+$', dirname):
            models_to_test.append(dirpath)

    if not models_to_test:
        logger.error(f"ERROR: No trained model folders found in '{model_search_root}'. Please check the path.")
        return

    logger.info(f"--- Found {len(models_to_test)} model folders to process ---")
    for model_path in models_to_test:
        logger.info(f" - {os.path.basename(model_path)}")
    logger.info("-" * 50)

    for model_path in models_to_test:
        logger.info(f"\n\n{'='*20} [Processing Model: {os.path.basename(model_path)}] {'='*20}")

        base_name = os.path.basename(model_path)
        regex = r'(\d{4}_\d{4})_(hct116|u2os)_(protT5|msa|pglm|ankh|esm2)_(alt|diff)_(1-3|2-3|3-3)_ep\d+-\d+'
        match = re.search(regex, base_name)

        if not match:
            logger.warning(f"[WARNING] Model folder name '{base_name}' does not match expected format. Skipping.")
            continue
        
        date_time, data_name, emb_name, emb_type, fold = match.groups()
        logger.info(f"Parsed Info -> Data: {data_name}, Embedding: {emb_name}, Type: {emb_type}")

        logger.info("Loading token dictionary, gene map, and variant embeddings...")
        with open(args.token_dict_path, "rb") as fp:
            token_dictionary = pickle.load(fp)
        
        gene_map = get_ensembl_gene_mapping()
        if gene_map is None:
            return
        
        knowledges = dict()
        out = load_prior_embedding(token_dictionary_or_path=args.token_dict_path)
        knowledges['promoter'] = out[0]
        knowledges['co_exp'] = out[1]
        knowledges['gene_family'] = out[2]
        knowledges['peca_grn'] = out[3]
        knowledges['homologous_gene_human2mouse'] = out[4]
        logger.info("Prior knowledge loaded successfully.")

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

        with open(variant_pkl, 'rb') as f:
            variant_dict = pickle.load(f)

        variant_dim = infer_variant_dim_from_pkl(variant_dict)
        variant_embed_generator = VariantEmbeddingGenerator(variant_dict, variant_dim)

        clinvar_dir = f"/NFS_DATA/samsung/database/gears/kim2023_{data_name}_[benchmark][oncoKB]"
        for chunk_i in range(1, 24):
            chunk_str = f"{chunk_i:02d}" 
            ts_data_file = os.path.join(clinvar_dir, f"perturb_processed_[{chunk_str}].h5ad")
            
            if not os.path.exists(ts_data_file):
                print(f"[SKIP] Chunk {chunk_str} not found at {ts_data_file}")
                continue

            test_dataset = AnnDataDataset(
                ts_data_file, 
                token_dictionary, 
                gene_map, 
                variant_embed_generator, 
                embedding_type=emb_type,
                num_top_genes=args.num_top_genes 
            )
            
            data_collator = PerturbationDataCollator(pad_token_id=token_dictionary.get("<pad>"))
            
            test_dataloader = DataLoader(
                    test_dataset, 
                    batch_size=args.eval_batch_size, 
                    shuffle=False, 
                    collate_fn=data_collator
                )
            num_known_genes = len(test_dataset.known_gene_symbols)
            known_gene_indices = test_dataset.known_gene_indices        
            num_total_genes = test_dataset.adata.n_vars
            logger.info(f"Number of total genes to predict: {num_total_genes}")

            logger.info(f"Loading fine-tuned model from: {model_path}")

            gene_symbol_to_ensembl = {v: k for k, v in gene_map.items()}
            tp53_ensembl_id = gene_symbol_to_ensembl.get('TP53')
            tp53_token_id = token_dictionary.get(tp53_ensembl_id)
            if tp53_token_id is None:
                raise ValueError("Could not find TP53 token ID during inference setup.")
            logger.info(f"Found TP53 Token ID for inference: {tp53_token_id}")

            config = AutoConfig.from_pretrained(model_path)

            if hasattr(config, "use_value_emb"):
                config.use_value_embed = config.use_value_emb
            elif hasattr(config, "use_value_embed"):
                pass
            else:
                config.use_value_embed = False
                
            logger.info(f"Inference Config - use_value_embed set to: {config.use_value_embed}")

            model = GeneCompass_gene.from_pretrained(
                model_path,
                config=config,
                knowledges=knowledges,
                num_genes_to_predict=num_known_genes,
                tp53_token_id=tp53_token_id,
                variant_dim=variant_dim
            )
            
            model.to(device)
            model.eval() 

            logger.info("Starting prediction...")
            all_predictions = []
            all_labels = []

            with torch.no_grad(): 
                for batch in tqdm(test_dataloader, desc="Predicting batches"):
                    inputs = {k: v.to(device) for k, v in batch.items()}
                    
                    outputs = model(**inputs)
                    
                    all_predictions.append(outputs.logits.cpu().numpy())
                    all_labels.append(inputs["labels"].cpu().numpy())

            all_predictions = np.concatenate(all_predictions, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)

            logger.info("Restoring predictions to full gene space...")
            all_predictions_full = np.zeros((all_predictions.shape[0], num_total_genes))
            all_labels_full = np.zeros((all_labels.shape[0], num_total_genes))

            all_predictions_full[:, known_gene_indices] = all_predictions
            all_labels_full[:, known_gene_indices] = all_labels

            if fold:
                fold2 = fold.replace('-', '_') 
            else:
                fold2 = "unknown"

            master_template_path = f"/NFS_DATA/samsung/database/gears/kim2023_{data_name}_[benchmark][{fold2}-fold]/perturb_processed_metadata.h5ad"
            adata_master = sc.read_h5ad(master_template_path)
            adata_master = adata_master[adata_master.obs['split']=='test'].copy()
            master_gene_list = adata_master.var_names.tolist()
            
            predicted_gene_list = test_dataset.known_gene_symbols
            variants = test_dataset.adata.obs['condition']
            counts = variants.groupby(variants, observed=False).cumcount()
            adata_index = variants.astype(str) + '_' + counts.astype(str)

            df_pred_test = pd.DataFrame(all_predictions_full, index=adata_index, columns=test_dataset.adata.var_names).reindex(columns=master_gene_list, fill_value=0)
            df_pred_final = df_pred_test
            final_obs = test_dataset.adata.obs.copy()
            final_var = test_dataset.adata.var.copy()
            final_layers = test_dataset.adata.layers.copy()

            adata_pred = ad.AnnData(X=df_pred_final.values, obs=final_obs, var=final_var, layers=final_layers)
            
            filename_base = f"{date_time}_{data_name}_{emb_name}_{emb_type}_{fold}_{chunk_str}"
            
            pred_path = os.path.join(anndata_dir, f"{filename_base}_pred.h5ad")          
            adata_pred.write_h5ad(pred_path)
            
           logger.info(f"Inference Successfully Completed.)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_path", type=str, default="/home/tech/variantseq/DATASETS/")
    parser.add_argument("--token_dict_path", type=str, default="/home/tech/variantseq/GeneCompass/prior_knowledge/human_mouse_tokens.pickle")
    parser.add_argument("--pretrained_model", type=str, default= "/NFS_DATA/samsung/foundation/genecompass")
    parser.add_argument("--finetuned_model", type=str, default="/NFS_DATA/samsung/GeneCompass/")
    parser.add_argument("--anndata_save", type=str, default="/NFS_DATA/samsung/GeneCompass/")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--date", type=int, required=True)
    parser.add_argument("--num_fold", type=str)
    
    parser.add_argument("--eval_batch_size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_top_genes", type=int, default=2047)

    parser.add_argument("--plm", type=str, choices=['esm2', 'protT5', 'msa', 'pglm', 'ankh'])
    parser.add_argument("--cell_line", type=str, choices=["hct116", "u2os"])
    parser.add_argument("--emb_type", type=str, choices=["alt", "diff"])

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

