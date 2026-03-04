'''
python train_scgpt_perturb.py \
    --dataloader_path "/NFS_DATA/samsung/database/gears/kim2023_hct116_[benchmark][3_3-fold]/dataloader/dataloader.pkl" \
    --adata_path "/NFS_DATA/samsung/database/gears/kim2023_hct116_v3_single_variant/perturb_processed.h5ad" \
    --pkl_path "/NFS_DATA/samsung/database/gears/kim2023_hct116_v3_single_variant/embedding_cache/embedding_cache_[ProtT5].pkl" \
    --embedding_key "ALT" \
    --load_model "../save/scGPT_human" \
    --save_root "/NFS_DATA/samsung/scGPT/1207" \
    --data_name "kim2023_hct116_1207" \
    --epochs 100 \
    --wandb_project "My_scGPT_Project"
'''


import json
import os
import sys
import time
import copy
import argparse
import pickle
import warnings
from pathlib import Path
from typing import Dict, Optional
from collections import OrderedDict

import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torchtext.vocab import Vocab
from torchtext._torchtext import Vocab as VocabPybind
from torch_geometric.loader import DataLoader

import wandb
import anndata

# scGPT 관련 import (환경에 맞게 경로 설정 필요)
sys.path.insert(0, "../")
import scgpt as scg
from scgpt.model import TransformerGenerator
from scgpt.loss import masked_mse_loss
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import set_seed, map_raw_id_to_vocab_id, compute_perturbation_metrics

warnings.filterwarnings("ignore")

# =============================================================================
# Helper Functions & Classes
# =============================================================================

class FakePertData:
    def __init__(self, dataloader_dict, adata):
        self.adata = adata
        self.dataloader_dict = dataloader_dict

    def get_dataloader(self, batch_size=None, test_batch_size=None):
        return (
            self.dataloader_dict["train_loader"],
            self.dataloader_dict["val_loader"],
            self.dataloader_dict["test_loader"],
        )

def build_expr_binarypert_x_list(adata, gene_to_idx, gene_names):
    x_tensor_list = []
    n_genes = len(gene_names)
    
    for i in range(adata.n_obs):
        expr = adata.X[i].toarray().flatten()  # (n_genes,)
        pert_gene = adata.obs["gene"][i]
        pert_idx = gene_to_idx.get(pert_gene, -1)

        if pert_idx == -1:
            raise ValueError(f"Gene '{pert_gene}' not found in gene_names.")

        # binary pert flag
        pert_flag = torch.zeros(n_genes, dtype=torch.float32)
        pert_flag[pert_idx] = 1.0

        # [n_genes, 2]: (expression, pert_flag)
        x_i = torch.stack([torch.tensor(expr, dtype=torch.float32), pert_flag], dim=1)
        x_tensor_list.append(x_i)

    return x_tensor_list

def train_epoch(model, train_loader, optimizer, scaler, scheduler, device, n_genes, tp53_raw_id, gene_ids_in_vocab, gene_ids, variant_vocab, variant_embs_proj, args):
    model.train()
    total_loss, total_mse = 0.0, 0.0
    total_de_loss = 0.0
    total_loss_epoch = 0.0
    start_time = time.time()
    
    criterion = masked_mse_loss
    num_batches = len(train_loader)
    
    # Pre-convert to tensor for efficiency
    gene_ids_in_vocab_tensor = torch.from_numpy(gene_ids_in_vocab).to(device)

    for batch, batch_data in enumerate(train_loader):
        batch_size = len(batch_data.y)
        batch_data.to(device)
        x = batch_data.x  # (batch_size * n_genes, 2)
        ori_gene_values = x[:, 0].view(batch_size, n_genes)
        pert_flags = x[:, 1].long().view(batch_size, n_genes)
        target_gene_values = batch_data.y

        if args.include_zero_gene in ["all", "batch-wise"]:
            if args.include_zero_gene == "all":
                initial_candidate_ids = torch.arange(n_genes, device=device, dtype=torch.long)
            else:
                initial_candidate_ids = (
                    ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                )
            
            # 1. Vocab에 있는 유전자만 필터링
            is_matched_mask = (gene_ids_in_vocab_tensor[initial_candidate_ids] >= 0)
            filtered_candidate_ids = initial_candidate_ids[is_matched_mask]

            # 2. TP53 강제 포함 로직
            tp53_id_scalar = tp53_raw_id.item()
            is_tp53_in_pool = (filtered_candidate_ids == tp53_id_scalar).any()

            if len(filtered_candidate_ids) > args.max_seq_len:
                if is_tp53_in_pool:
                    candidate_ids_without_tp53 = filtered_candidate_ids[filtered_candidate_ids != tp53_id_scalar]
                    num_to_sample = args.max_seq_len - 1
                else:
                    candidate_ids_without_tp53 = filtered_candidate_ids
                    num_to_sample = args.max_seq_len - 1

                perm = torch.randperm(len(candidate_ids_without_tp53), device=device)
                sampled_other_ids = candidate_ids_without_tp53[perm[:num_to_sample]]
                
                final_ids_unsuffled = torch.cat([tp53_raw_id, sampled_other_ids])
                final_perm = torch.randperm(args.max_seq_len, device=device)
                input_gene_ids = final_ids_unsuffled[final_perm]
            else:
                if not is_tp53_in_pool and len(filtered_candidate_ids) < n_genes:
                    input_gene_ids = torch.cat([tp53_raw_id, filtered_candidate_ids])
                else:
                    input_gene_ids = filtered_candidate_ids
            
            input_values = ori_gene_values[:, input_gene_ids]
            input_pert_flags = pert_flags[:, input_gene_ids]
            target_values = target_gene_values[:, input_gene_ids]

            mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
            mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

            src_key_padding_mask = torch.zeros_like(
                input_values, dtype=torch.bool, device=device
            )

            # Variant embedding lookup
            pert_names = batch_data.pert
            variant_ids_list = [variant_vocab[name.split("+")[0]] for name in pert_names]
            variant_ids_tensor = torch.tensor(variant_ids_list, dtype=torch.long)
            variant_embs = variant_embs_proj[variant_ids_tensor].detach().to(device)

        with torch.cuda.amp.autocast(enabled=args.amp):
            output_dict = model(
                mapped_input_gene_ids,
                input_values,
                input_pert_flags,
                variant_emb=variant_embs,
                src_key_padding_mask=src_key_padding_mask,
                CLS=args.CLS, CCE=args.CCE, MVC=args.MVC, ECS=args.ECS,
            )
            output_values = output_dict["mlm_output"]

            masked_positions = torch.ones_like(input_values, dtype=torch.bool)
            raw_loss = criterion(output_values, target_values, masked_positions)
            loss = raw_loss
            loss_mse = raw_loss.detach()

            # DE-only loss
            gene_id_map = {gid.item(): idx for idx, gid in enumerate(input_gene_ids)}
            de_values_pred = []
            de_values_true = []
            for i, de_idx_list in enumerate(batch_data.de_idx):
                valid_de = [gene_id_map[did] for did in de_idx_list if did in gene_id_map]
                if valid_de:
                    valid_de = torch.tensor(valid_de, device=device, dtype=torch.long)
                    de_values_pred.append(output_values[i, valid_de])
                    de_values_true.append(target_values[i, valid_de])

            if de_values_pred:
                de_values_pred = torch.cat(de_values_pred)
                de_values_true = torch.cat(de_values_true)
                loss_de = F.mse_loss(de_values_pred, de_values_true).item()
            else:
                loss_de = 0.0

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=False if scaler.is_enabled() else True
            )
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_mse += loss_mse.item()
        total_loss_epoch += loss.item()
        total_de_loss += loss_de

        if batch % args.log_interval == 0 and batch > 0:
            lr_curr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / args.log_interval
            cur_loss = total_loss / args.log_interval
            cur_mse = total_mse / args.log_interval
            print(f"| epoch {args.current_epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                  f"lr {lr_curr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                  f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} | de_loss {loss_de:5.2f} |")
            total_loss = 0
            total_mse = 0
            start_time = time.time()

    avg_loss = total_loss_epoch / num_batches
    avg_de_loss = total_de_loss / num_batches
    return avg_loss, avg_de_loss

def eval_perturb(loader, model, device, args, n_genes, tp53_raw_id, gene_ids, variant_vocab, variant_embs_proj):
    model.eval()
    model.to(device)
    pert_cat = []
    pred = []
    truth = []
    pred_de = []
    truth_de = []
    results = {}

    with torch.no_grad():
        for itr, batch in enumerate(loader):
            batch.to(device)
            pert_cat.extend(batch.pert)

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

            for i, de_idx in enumerate(batch.de_idx):
                pred_de.append(p[i, de_idx])
                truth_de.append(t[i, de_idx])
            
            torch.cuda.empty_cache()

    results["pert_cat"] = np.array(pert_cat)
    results["pred"] = torch.stack(pred).numpy().astype(np.float32)
    results["truth"] = torch.stack(truth).numpy().astype(np.float32)
    results["pred_de"] = torch.cat(pred_de).numpy().astype(np.float32)
    results["truth_de"] = torch.cat(truth_de).numpy().astype(np.float32)
    return results

def validate(model, val_loader, device, n_genes, tp53_raw_id, gene_ids_in_vocab, gene_ids, variant_vocab, variant_embs_proj, args):
    model.eval()
    total_loss = 0.0
    total_masked_positions = 0
    all_de_pred = []
    all_de_truth = []
    
    criterion = masked_mse_loss
    gene_ids_in_vocab_tensor = torch.from_numpy(gene_ids_in_vocab).to(device)

    with torch.no_grad():
        for batch_data in val_loader:
            batch_data.to(device)
            batch_size = len(batch_data.y)
            x = batch_data.x
            ori_gene_values = x[:, 0].view(batch_size, n_genes)
            pert_flags = x[:, 1].long().view(batch_size, n_genes)
            target_gene_values = batch_data.y

            # Sampling logic (same as train)
            initial_candidate_ids = torch.arange(n_genes, device=device, dtype=torch.long)
            is_matched_mask = (gene_ids_in_vocab_tensor[initial_candidate_ids] >= 0)
            filtered_candidate_ids = initial_candidate_ids[is_matched_mask]
            
            tp53_id_scalar = tp53_raw_id.item()
            is_tp53_in_pool = (filtered_candidate_ids == tp53_id_scalar).any()

            if len(filtered_candidate_ids) > args.max_seq_len:
                if is_tp53_in_pool:
                    candidate_ids_without_tp53 = filtered_candidate_ids[filtered_candidate_ids != tp53_id_scalar]
                    num_to_sample = args.max_seq_len - 1
                else:
                    candidate_ids_without_tp53 = filtered_candidate_ids
                    num_to_sample = args.max_seq_len - 1
                perm = torch.randperm(len(candidate_ids_without_tp53), device=device)
                sampled_other_ids = candidate_ids_without_tp53[perm[:num_to_sample]]
                final_ids_unsuffled = torch.cat([tp53_raw_id, sampled_other_ids])
                final_perm = torch.randperm(args.max_seq_len, device=device)
                input_gene_ids = final_ids_unsuffled[final_perm]
            else:
                if not is_tp53_in_pool and len(filtered_candidate_ids) < n_genes:
                    input_gene_ids = torch.cat([tp53_raw_id, filtered_candidate_ids])
                else:
                    input_gene_ids = filtered_candidate_ids

            input_values = ori_gene_values[:, input_gene_ids]
            input_pert_flags = pert_flags[:, input_gene_ids]
            target_values = target_gene_values[:, input_gene_ids]

            mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
            mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)
            src_key_padding_mask = torch.zeros_like(input_values, dtype=torch.bool, device=device)
            
            pert_names = batch_data.pert
            variant_ids_list = [variant_vocab[name.split("+")[0]] for name in pert_names]
            variant_ids = torch.tensor(variant_ids_list, dtype=torch.long)
            variant_embs = variant_embs_proj[variant_ids].detach().to(device)

            with torch.cuda.amp.autocast(enabled=args.amp):
                output_dict = model(
                    mapped_input_gene_ids, input_values, input_pert_flags,
                    variant_emb=variant_embs, src_key_padding_mask=src_key_padding_mask,
                    CLS=args.CLS, CCE=args.CCE, MVC=args.MVC, ECS=args.ECS
                )
                output_values = output_dict["mlm_output"]
                masked_positions = torch.ones_like(input_values, dtype=torch.bool)
                loss = criterion(output_values, target_values, masked_positions)
                num_masked = masked_positions.sum().item()
                total_loss += loss.item() * num_masked
                total_masked_positions += num_masked

                gene_id_map = {gid.item(): idx for idx, gid in enumerate(input_gene_ids)}
                for i, de_idx_list in enumerate(batch_data.de_idx):
                    valid_de = [gene_id_map[did] for did in de_idx_list if did in gene_id_map]
                    if valid_de:
                        valid_de = torch.tensor(valid_de, device=device, dtype=torch.long)
                        all_de_pred.append(output_values[i, valid_de].detach().cpu())
                        all_de_truth.append(target_values[i, valid_de].detach().cpu())

    avg_val_loss = total_loss / total_masked_positions
    all_de_pred = torch.cat(all_de_pred)
    all_de_truth = torch.cat(all_de_truth)
    loss_de = F.mse_loss(all_de_pred, all_de_truth).item()
    return avg_val_loss, loss_de

# =============================================================================
# Main Execution
# =============================================================================

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Directory Setup ---
    save_dir = Path(f"{args.save_root}/")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving checkpoints to {save_dir}")

    logger = scg.logger
    scg.utils.add_file_handler(logger, save_dir / "run.log")
    logger.info(f"Running on {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # --- Data Loading ---
    print(f"Loading data from {args.dataloader_path}...")
    try:
        with open(args.dataloader_path, 'rb') as f:
            dataloader_dict = pickle.load(f)
        print("✅ Dataloader loaded.")
    except Exception as e:
        print(f"❌ Error loading dataloader: {e}")
        sys.exit(1)

    print(f"Loading AnnData from {args.adata_path}...")
    adata = anndata.read_h5ad(args.adata_path)
    gene_names = list(adata.var_names)
    n_genes = len(gene_names)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    # Apply x tensor to dataloaders
    print("Building expression + binary perturbation inputs...")
    x_tensor_list = build_expr_binarypert_x_list(adata, gene_to_idx, gene_names)
    
    for split in ["train_loader", "val_loader", "test_loader"]:
        loader = dataloader_dict[split]
        data_list = loader.dataset
        for i, data in enumerate(data_list):
            data.x = x_tensor_list[i]

    pert_data = FakePertData(dataloader_dict, adata)

    # --- Model & Vocab Setup ---
    special_tokens = [args.pad_token, "<cls>", "<eoc>"]
    
    if args.load_model is not None:
        model_dir = Path(args.load_model)
        model_config_file = model_dir / "args.json"
        model_file = model_dir / "best_model.pt"
        vocab_file = model_dir / "vocab.json"

        vocab = GeneVocab.from_file(vocab_file)
        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)
        
        pert_data.adata.var["id_in_vocab"] = [
            1 if gene in vocab else -1 for gene in pert_data.adata.var["gene_name"]
        ]
        gene_ids_in_vocab = np.array(pert_data.adata.var["id_in_vocab"])
        logger.info(f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes in vocabulary.")
        
        genes = pert_data.adata.var["gene_name"].tolist()
        
        # Load Model Config
        with open(model_config_file, "r") as f:
            model_configs = json.load(f)
        
        embsize = model_configs["embsize"]
        nhead = model_configs["nheads"]
        d_hid = model_configs["d_hid"]
        nlayers = model_configs["nlayers"]
        n_layers_cls = model_configs["n_layers_cls"]
    else:
        # Fallback if no pretrained model
        genes = pert_data.adata.var["gene_name"].tolist()
        vocab = Vocab(VocabPybind(genes + special_tokens, None))
        embsize = args.embsize
        nhead = args.nhead
        d_hid = args.d_hid
        nlayers = args.nlayers
        n_layers_cls = args.n_layers_cls

    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array([vocab[gene] if gene in vocab else vocab["<pad>"] for gene in genes], dtype=int)

    # TP53 Index
    try:
        tp53_raw_id_int = genes.index('TP53')
        print(f"✅ 'TP53' raw index: {tp53_raw_id_int}")
    except ValueError:
        raise ValueError("Error: 'TP53' not found in gene list.")
    tp53_raw_id = torch.tensor([tp53_raw_id_int], device=device)

    # --- Variant Embeddings (Updated for ALT/DIFF dict support) ---
    print(f"Loading variant embeddings from {args.pkl_path}...")
    print(f"Using embedding key: {args.embedding_key}")
    
    # 1. adata.obs['condition']에서 Variant 목록 추출
    unique_conditions = adata.obs['condition'].unique().tolist()
    variant_list_full_names = []
    
    for cond in unique_conditions:
        if cond == 'ctrl':
            continue
        # '+ctrl' 접미사 제거
        variant_name = cond.split('+')[0]
        variant_list_full_names.append(variant_name)
    
    # 중복 제거 및 정렬
    variant_list_full_names = sorted(list(set(variant_list_full_names)))
    print(f"✅ Extracted {len(variant_list_full_names)} variants from adata.")

    # 2. Short name -> Full name 매핑
    short_to_full_map = {name.split('~')[1]: name for name in variant_list_full_names}
    
    # 3. Vocab 생성
    variant_vocab = OrderedDict([("<pad>", 0), ("ctrl", 1), ("REF", 2)] + [(v, i + 3) for i, v in enumerate(variant_list_full_names)])
    
    # 4. Pickle 캐시 로드
    with open(args.pkl_path, "rb") as f:
        cache = pickle.load(f)
        
    processed_embeddings = {}
    GENE_TARGET = "TP53" 
    embedding_dim = None

    # 5. 임베딩 처리 (Dictionary Access)
    for (gene, variant_short_name), emb_dict in cache.items():
        if gene == GENE_TARGET:
            if variant_short_name in short_to_full_map:
                full_name = short_to_full_map[variant_short_name]
                
                # Check if the requested key exists (ALT or DIFF)
                if args.embedding_key in emb_dict:
                    raw_vector = emb_dict[args.embedding_key]
                    
                    # Convert list/array to tensor
                    embedding_tensor = torch.tensor(raw_vector).float()
                    
                    processed_embeddings[full_name] = embedding_tensor
                    
                    # Set dimension from first valid embedding found
                    if embedding_dim is None:
                        embedding_dim = embedding_tensor.shape[0]
                else:
                    print(f"⚠️ Warning: Key '{args.embedding_key}' not found for {full_name}")

    if embedding_dim is None:
         raise ValueError(f"❌ No valid embeddings found with key '{args.embedding_key}'. Check pickle file or key argument.")
    
    print(f"Detected embedding dimension: {embedding_dim}")

    # Special tokens initialization (Zero vectors)
    processed_embeddings["<pad>"] = torch.zeros(embedding_dim)
    processed_embeddings["ctrl"] = torch.zeros(embedding_dim)
    processed_embeddings["REF"] = torch.zeros(embedding_dim) # Handle REF as zero if not in cache or needed
    
    # 6. Vocab 순서대로 리스트 생성
    variant_emb_list = [None] * len(variant_vocab)
    for name, idx in variant_vocab.items():
        if name not in processed_embeddings:
            # If REF or ctrl is missing from processed (likely handled above), fill with zero
            if name in ['REF', 'ctrl', '<pad>']:
                 variant_emb_list[idx] = torch.zeros(embedding_dim)
            else:
                 print(f"⚠️ Warning: Missing embedding for {name}, filling with zero.")
                 variant_emb_list[idx] = torch.zeros(embedding_dim)
        else:
            variant_emb_list[idx] = processed_embeddings[name]
    
    # 7. Stack & Projection (MLP)
    variant_embs_cpu = torch.stack(variant_emb_list)
    projector = nn.Linear(embedding_dim, embsize).to(device)
    
    with torch.no_grad():
        variant_embs_proj = projector(variant_embs_cpu.to(device))
    
    print(f"✅ Variant embeddings prepared & projected. Shape: {variant_embs_proj.shape}")

    # --- Initialize Model ---
    ntokens = len(vocab)
    model = TransformerGenerator(
        ntokens, embsize, nhead, d_hid, nlayers,
        nlayers_cls=n_layers_cls, n_cls=1, vocab=vocab,
        dropout=args.dropout, pad_token=args.pad_token,
        pad_value=args.pad_value, pert_pad_id=args.pert_pad_id,
        use_fast_transformer=True
    )

    if args.load_model is not None:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file)
        # Load params with prefix match, excluding Wqkv if needed (as per notebook)
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items()
            if any([k.startswith(prefix) for prefix in args.load_param_prefixs])
            and "Wqkv" not in k
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logger.info("✅ Pretrained model weights loaded.")
    
    model.to(device)

    # --- Optimizer & WandB ---
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.schedule_interval, gamma=0.9)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    wandb.init(
        project=args.wandb_project,
        name=f"{args.data_name}_{args.emb_name}_{time.strftime('%Y%m%d-%H%M%S')}",
        config=vars(args),
        reinit=True
    )

    # --- Training Loop ---
    best_val_corr = 0
    patience = 0
    
    for epoch in range(1, args.epochs + 1):
        args.current_epoch = epoch # Helper for logging
        train_loader = pert_data.dataloader_dict["train_loader"]
        valid_loader = pert_data.dataloader_dict["val_loader"]

        train_loss, train_de_loss = train_epoch(
            model, train_loader, optimizer, scaler, scheduler, device,
            n_genes, tp53_raw_id, gene_ids_in_vocab, gene_ids,
            variant_vocab, variant_embs_proj, args
        )

        # Validation
        val_res = eval_perturb(
            valid_loader, model, device, args, n_genes, tp53_raw_id, 
            gene_ids, variant_vocab, variant_embs_proj
        )
        val_metrics = compute_perturbation_metrics(
            val_res, pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
        )
        
        val_loss, val_de_loss = validate(
            model, valid_loader, device, n_genes, tp53_raw_id,
            gene_ids_in_vocab, gene_ids, variant_vocab, variant_embs_proj, args
        )

        val_pearson = val_metrics.get("pearson", 0)
        val_de_pearson = val_metrics.get("pearson_de", 0)

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Pearson: {val_pearson:.4f}")

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_de_loss": train_de_loss,
            "val_loss": val_loss,
            "val_de_loss": val_de_loss,
            "val_pearson": val_pearson,
            "val_de_pearson": val_de_pearson,
        })

        if val_pearson > best_val_corr:
            best_val_corr = val_pearson
            best_model_path = save_dir / "best_model.pt"
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved best model with score {val_pearson:.4f} to {best_model_path}")
            patience = 0
        else:
            patience += 1
            if patience >= args.early_stop:
                logger.info(f"Early stop at epoch {epoch}")
                break
        
        scheduler.step()

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train scGPT with Variant Perturbations")

    # Data Paths
    parser.add_argument("--dataloader_path", type=str, required=True, help="Path to dataloader.pkl")
    parser.add_argument("--adata_path", type=str, required=True, help="Path to .h5ad file")
    parser.add_argument("--pkl_path", type=str, required=True, help="Path to embedding pickle")
    parser.add_argument("--save_root", type=str, default="./ck", help="Root directory for checkpoints")
    parser.add_argument("--load_model", type=str, default="../save/scGPT_human", help="Path to pretrained scGPT")

    # Arguments for Embedding Selection
    parser.add_argument("--embedding_key", type=str, default="ALT", choices=["ALT", "DIFF", "REF"], 
                        help="Key to extract from embedding dict (e.g., ALT, DIFF)")

    # Configs
    parser.add_argument("--data_name", type=str, default="kim2023_hct116", help="Name of dataset")
    parser.add_argument("--emb_name", type=str, default="protT5", help="Embedding type name")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--pad_token", type=str, default="<pad>")
    parser.add_argument("--pad_value", type=int, default=0)
    parser.add_argument("--pert_pad_id", type=int, default=0)
    parser.add_argument("--include_zero_gene", type=str, default="all")
    parser.add_argument("--max_seq_len", type=int, default=1536)
    
    # Training Params
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--schedule_interval", type=int, default=1)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--dropout", type=float, default=0.0)

    # Model Params
    parser.add_argument("--embsize", type=int, default=512)
    parser.add_argument("--d_hid", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--nlayers", type=int, default=12)
    parser.add_argument("--n_layers_cls", type=int, default=3)
    
    # Objectives
    parser.add_argument("--MLM", action="store_true", default=True)
    parser.add_argument("--CLS", action="store_true", default=False)
    parser.add_argument("--CCE", action="store_true", default=False)
    parser.add_argument("--MVC", action="store_true", default=False)
    parser.add_argument("--ECS", action="store_true", default=False)

    # WandB
    parser.add_argument("--wandb_project", type=str, default="scGPT_variant_training")

    # Misc
    parser.add_argument("--load_param_prefixs", nargs="+", default=["encoder", "value_encoder", "transformer_encoder"])

    args = parser.parse_args()
    main(args)