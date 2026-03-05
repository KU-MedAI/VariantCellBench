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
from data_process import AnnDataDataset, PerturbationDataCollator, get_ensembl_gene_mapping
from loss import compute_metrics
from transformers import BertConfig, TrainingArguments, Trainer, EarlyStoppingCallback
from genecompass.modeling_bert import GeneCompass_gene
from genecompass.utils import load_prior_embedding
from visualization import train_valid_lossplot

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_training(args): 
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    now = datetime.datetime.now()
    timestamp = now.strftime("%m%d_%H%M")
    base_run_name = f"{timestamp}_{args.cell_line}_{args.plm}_{args.emb_type}_{args.num_fold}_ep{args.num_train_epochs}"
    plot_title = f"{args.cell_line}_{args.plm}_{args.emb_type}"
    logger.info(f"Base run name set to: {base_run_name}")

    parent_dir = os.path.join(args.finetuned_model, f"checkpoint_{args.date}")
    finetuned_model_dir = os.path.join(parent_dir, base_run_name)
    training_output_dir = os.path.join(args.train_result, f"loss_plot_{args.date}")      
    logging_dir = os.path.join(args.train_result, f"log_{args.date}")           
    
    os.makedirs(finetuned_model_dir, exist_ok=True)
    os.makedirs(training_output_dir, exist_ok=True)
    os.makedirs(logging_dir, exist_ok=True)
    
    logger.info("Loading token dictionary and variant embeddings...")
    with open(args.token_dict_path, "rb") as fp:
        token_dictionary = pickle.load(fp)
    
    gene_map = get_ensembl_gene_mapping()
    if gene_map is None:
        logger.error("Gene mapping을 가져올 수 없어 프로그램을 종료합니다.")
        return

    logger.info("Loading prior knowledge embeddings...")
    knowledges = dict()
    out = load_prior_embedding()
    knowledges['promoter'] = out[0]
    knowledges['co_exp'] = out[1]
    knowledges['gene_family'] = out[2]
    knowledges['peca_grn'] = out[3]
    knowledges['homologous_gene_human2mouse'] = out[4]
    logger.info("Prior knowledge loaded successfully.")


    if args.plm == 'esm2':
        variant_pkl = "/NFS_DATA/samsung/database/gears/embedding/embedding_cache_variant_position_[esm2_t33_650M_UR50D].pkl"
    elif args.plm == 'protT5':
        variant_pkl = "/NFS_DATA/samsung/database/gears/embedding/embedding_cache_variant_position_[ProtT5-XXL-U50].pkl"
    elif args.plm == 'msa':
        variant_pkl = "/NFS_DATA/samsung/database/gears/embedding/embedding_cache_variant_position_[esm_msa1_t12_100M_UR50S].pkl"
    elif args.plm == 'pglm':
         variant_pkl = "/NFS_DATA/samsung/database/gears/embedding/embedding_cache_variant_position_[xTrimoPGLM-10B-MLM].pkl"
    elif args.plm == 'ankh':
        variant_pkl = "/NFS_DATA/samsung/database/gears/embedding/embedding_cache_variant_position_[Ankh3-Large].pkl"

    with open(variant_pkl, 'rb') as f:
        variant_dict = pickle.load(f)

    variant_dim = infer_variant_dim_from_pkl(variant_dict)
    logger.info(f"Inferred variant embedding dimension: {variant_dim}")
    variant_embed_generator = VariantEmbeddingGenerator(variant_dict, variant_dim)

    tr_data = os.path.join(args.data_path, f'{args.cell_line}_train_{args.num_fold}.h5ad')
    val_data = os.path.join(args.data_path, f'{args.cell_line}_valid_{args.num_fold}.h5ad')

    logger.info("Initializing datasets...")
    train_dataset = AnnDataDataset(tr_data, token_dictionary, gene_map, variant_embed_generator, embedding_type=args.emb_type, num_top_genes=args.num_top_genes)
    val_dataset = AnnDataDataset(val_data, token_dictionary, gene_map, variant_embed_generator, embedding_type=args.emb_type, num_top_genes=args.num_top_genes)
    num_total_genes = train_dataset.adata.n_vars
    num_known_genes = len(train_dataset.known_gene_symbols)
    logger.info(f"number of total genes: {num_total_genes}")

    pretrained_model = args.pretrained_model
    config = AutoConfig.from_pretrained(pretrained_model)
    if hasattr(config, "use_value_emb"):
        config.use_value_embed = config.use_value_emb

    elif hasattr(config, "use_value_embed"):
        pass 
        
    else:
        config.use_value_embed = False

    logger.info(f"Final Config - use_value_embed: {getattr(config, 'use_value_embed', False)}")

    logger.info("Initializing model in 'gene' variant mode (applying to TP53).")
    gene_symbol_to_ensembl = {v: k for k, v in gene_map.items()}
    tp53_ensembl_id = gene_symbol_to_ensembl.get('TP53')
    if tp53_ensembl_id is None:
        raise ValueError("TP53 not found in token_dictionary, but 'gene' mode requires it.")
    
    tp53_token_id = token_dictionary.get(tp53_ensembl_id)
    if tp53_token_id is None:
        raise ValueError(f"'TP53'(Ensembl ID: {tp53_ensembl_id})이 token_dictionary에 없습니다.")
    
    logger.info(f"Found TP53 -> Ensembl ID: {tp53_ensembl_id} -> Token ID: {tp53_token_id}")

    model = GeneCompass_gene.from_pretrained(
                pretrained_model,
                config=config,
                knowledges=knowledges,
                num_genes_to_predict=num_known_genes,
                tp53_token_id=tp53_token_id,
                variant_dim=variant_dim
            )

    logger.info(f"Successfully loaded {pretrained_model} weights from the pretrained model.")
    
    data_collator = PerturbationDataCollator(pad_token_id=token_dictionary.get("<pad>"))

    training_args = TrainingArguments(
            output_dir = finetuned_model_dir,
            logging_dir = logging_dir,
            run_name = base_run_name,

            num_train_epochs = args.num_train_epochs,
            per_device_train_batch_size = args.train_batch_size,
            per_device_eval_batch_size = args.eval_batch_size,
            dataloader_num_workers = 4, 

            learning_rate = args.learning_rate,
            lr_scheduler_type = "cosine", 
            weight_decay = args.weight_decay,
            warmup_steps = args.warmup_steps,

            fp16 = args.fp16,
            evaluation_strategy = "epoch",
            save_strategy = "epoch",
            logging_steps = 100,

            load_best_model_at_end = True,
            metric_for_best_model = "loss",
            greater_is_better = False,
            remove_unused_columns = False,
        )
    
    early_stopping = EarlyStoppingCallback(early_stopping_patience=5)

    trainer = Trainer(
            model = model,
            args = training_args,
            train_dataset = train_dataset,
            eval_dataset = val_dataset,
            data_collator = data_collator,
            compute_metrics = compute_metrics,
            callbacks=[early_stopping]
        )

    logger.info("Starting fine-tuning...")
    trainer.train()

    best_metric = trainer.state.best_metric
    best_epoch = None

    for log in trainer.state.log_history:
        if log.get(f"eval_{training_args.metric_for_best_model}") == best_metric:
            best_epoch = int(round(log['epoch']))
            break
    
    if best_epoch is not None:
        logger.info(f"Best model found at epoch {best_epoch} with {training_args.metric_for_best_model}: {best_metric}")
        
        final_run_name = f"{base_run_name}-{best_epoch}"
        final_model_save_path = os.path.join(parent_dir, final_run_name)
    else:
        logger.warning("Could not determine the best epoch")

    logger.info(f"Saving final best model to: {final_model_save_path}")
    trainer.save_model(final_model_save_path)
    
    logger.info(f"Cleaning up temporary working directory: {finetuned_model_dir}")
    try:
        import subprocess
        
        if os.path.exists(finetuned_model_dir):
            subprocess.run(["rm", "-rf", finetuned_model_dir], check=True)
            logger.info(">> Temporary directory deleted successfully.")
        else:
            logger.warning("Directory already gone.")
            
    except Exception as e:
        logger.error(f"An error occurred during cleanup: {e}")

    train_valid_lossplot(
            log_history=trainer.state.log_history,
            output_dir=training_output_dir,
            base_run_name=base_run_name,
            plot_title = plot_title
        )


def run_inference(args, device):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    anndata_dir = os.path.join(args.anndata_save, f"eval_data_{args.date}")
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

        ts_data_path = os.path.join(args.data_path, f'{data_name}_test_{fold}.h5ad')
        test_dataset = AnnDataDataset(
            ts_data_path, 
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

        logger.info(f"Loading fine-tuned model from: {model_path}")

        gene_symbol_to_ensembl = {v: k for k, v in gene_map.items()}
        tp53_ensembl_id = gene_symbol_to_ensembl.get('TP53')
        tp53_token_id = token_dictionary.get(tp53_ensembl_id)
        if tp53_token_id is None:
            raise ValueError("Could not find TP53 token ID during inference setup.")
        logger.info(f"Found TP53 Token ID for inference: {tp53_token_id}")

        from transformers import AutoConfig
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
        
        logger.info(f"Prediction complete. Shape: {all_predictions.shape}")

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
        variants = test_dataset.adata.obs['variant']
        counts = variants.groupby(variants, observed=False).cumcount()
        adata_index = variants.astype(str) + '_' + counts.astype(str)

        df_truth_test = pd.DataFrame(all_labels_full, index=adata_index, columns=test_dataset.adata.var_names).reindex(columns=master_gene_list, fill_value=0)
        df_pred_test = pd.DataFrame(all_predictions_full, index=adata_index, columns=test_dataset.adata.var_names).reindex(columns=master_gene_list, fill_value=0)

        control_expression_data = test_dataset.adata.layers['x']
        if hasattr(control_expression_data, 'toarray'):
            control_expression_data = control_expression_data.toarray()

        num_samples = len(adata_index)
        ctrl_index = [f"cell_{i}" for i in range(num_samples)]

        df_ctrl_truth = pd.DataFrame(control_expression_data, index=ctrl_index, columns=test_dataset.adata.var_names).reindex(columns=master_gene_list, fill_value=0)
        print(f"Created control data with shape {df_ctrl_truth.shape} from test_dataset.layers['x']")

        df_truth_final = pd.concat([df_truth_test, df_ctrl_truth])
        df_pred_final = pd.concat([df_pred_test, df_ctrl_truth]) 

        obs_with_names = adata_master.obs[['condition', 'condition_name']].copy()
        obs_with_names['clean_key'] = obs_with_names['condition'].str.split('+').str[0]
        condition_map = obs_with_names.drop_duplicates(subset=['clean_key']).set_index('clean_key')['condition_name']
        obs_test = pd.DataFrame(index=adata_index)
        obs_test['condition'] = test_dataset.adata.obs['variant'].values
        mapping_keys = obs_test['condition'].str.rsplit('_', n=1).str[0]
        obs_test['condition_name'] = mapping_keys.map(condition_map)

        obs_ctrl = pd.DataFrame(index=ctrl_index)
        obs_ctrl['condition'] = 'ctrl'
        if data_name == 'hct116':
            obs_ctrl['condition_name'] = 'HCT116_ctrl_1'
        elif data_name == 'u2os':
            obs_ctrl['condition_name'] = 'U2OS_ctrl_1'

        final_obs = pd.concat([obs_test, obs_ctrl])

        final_var = adata_master.var.copy()
        is_present = final_var.index.isin(predicted_gene_list)
        final_var['exist'] = is_present.astype(int)

        final_uns = adata_master.uns.copy()

        logger.info(f"Final obs shape:  {final_obs.shape}")
        assert df_truth_final.shape[0] == final_obs.shape[0], "CRITICAL ERROR: Data and metadata dimensions do not match!"

        adata_truth = ad.AnnData(X=df_truth_final.values, obs=final_obs, var=final_var, uns=final_uns)
        adata_pred = ad.AnnData(X=df_pred_final.values, obs=final_obs, var=final_var, uns=final_uns)

        filename_base = f"{date_time}_{data_name}_{emb_name}_{emb_type}_{fold}"
        
        truth_path = os.path.join(anndata_dir, f"{filename_base}_truth.h5ad")
        pred_path = os.path.join(anndata_dir, f"{filename_base}_pred.h5ad")
        
        adata_truth.write_h5ad(truth_path)
        adata_pred.write_h5ad(pred_path)
        
        logger.info(f"Inference Successfully Completed.)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='single_pipeline', 
                        choices=['single_pipeline', 'train', 'inference'])
    
    parser.add_argument("--data_path", type=str, default="/home/tech/variantseq/DATASETS/")
    parser.add_argument("--token_dict_path", type=str, default="/home/tech/variantseq/GeneCompass/prior_knowledge/human_mouse_tokens.pickle")
    parser.add_argument("--pretrained_model", type=str, default= "/NFS_DATA/samsung/foundation/genecompass_small")
    parser.add_argument("--finetuned_model", type=str, default="/NFS_DATA/samsung/GeneCompass/")
    parser.add_argument("--train_result", type=str, default="/home/tech/variantseq/GeneCompass/")
    parser.add_argument("--anndata_save", type=str, default="/NFS_DATA/samsung/GeneCompass/")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--date", type=int, required=True)
    parser.add_argument("--num_fold", type=str)
    
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action='store_true')

    parser.add_argument("--num_top_genes", type=int, default=2047)
    parser.add_argument("--from_scratch", action='store_true', help="Train model from scratch (random initialization)")

    parser.add_argument("--plm", type=str, choices=['esm2', 'protT5', 'msa', 'pglm', 'ankh'])
    parser.add_argument("--cell_line", type=str, choices=["hct116", "u2os"])
    parser.add_argument("--emb_type", type=str, choices=["alt", "diff"])

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    
    if args.mode == 'single_pipeline':
        run_training(args)
        run_inference(args, device)

    elif args.mode == 'train':
        if not all([args.plm, args.cell_line, args.emb_type]):
            parser.error("--emb_name, --data_name, --emb_type are required for 'train' mode.")
        
        run_training(args)

    elif args.mode == 'inference':
        run_inference(args, device)

