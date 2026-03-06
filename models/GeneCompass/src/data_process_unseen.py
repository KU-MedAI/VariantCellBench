#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import random
import argparse
import logging
import json
from typing import Dict, List, Union

import pandas as pd
import numpy as np
import anndata as ad
import torch
from torch import nn
from torch.utils.data import Dataset

from transformers import BertConfig, TrainingArguments, Trainer
from variant_emb import VariantEmbeddingGenerator
from genecompass.utils import load_prior_embedding

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_ensembl_gene_mapping(
            manual_mapping_file="/home/tech/variantseq/suji/GeneCompass/prior_knowledge/ens_gene_map.txt", 
            save_path='/home/tech/variantseq/suji/GeneCompass/prior_knowledge/ensembl_to_gene_symbol.pickle'
        ):

    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            return pickle.load(f)

    try:

        df = pd.read_csv(manual_mapping_file, sep='\t', header=0)
        
        ensembl_col = 'Gene stable ID'
        symbol_col = 'Gene name'
        
        if ensembl_col not in df.columns or symbol_col not in df.columns:
            logger.warning(f"예상된 컬럼명 '{ensembl_col}' 또는 '{symbol_col}'을(를) 찾을 수 없습니다.")
            logger.warning(f"파일의 첫 두 컬럼인 '{df.columns[0]}'과(와) '{df.columns[1]}'을(를) 대신 사용합니다.")
            ensembl_col = df.columns[0]
            symbol_col = df.columns[1]

        df.dropna(subset=[ensembl_col, symbol_col], inplace=True)
        ensembl_to_gene_map = dict(zip(df[ensembl_col], df[symbol_col]))

        with open(save_path, 'wb') as f:
            pickle.dump(ensembl_to_gene_map, f)
        
        logger.info(f"매핑 정보 생성 완료. '{save_path}'에 캐시로 저장했습니다.")
        return ensembl_to_gene_map

    except FileNotFoundError:
        logger.error(f"!!! 수동 다운로드 파일 '{manual_mapping_file}'을(를) 찾을 수 없습니다!")
        logger.error("    파일 이름이 정확한지, 스크립트와 같은 폴더에 있는지 확인해주세요.")
        return None
    except Exception as e:
        logger.error(f"수동 파일을 처리하는 중 오류가 발생했습니다: {e}")
        return None
    
class AnnDataDataset(Dataset):
    def __init__(self, 
                 h5ad_path: str, 
                 token_dictionary: Dict,  # Ensembl ID -> Token ID
                 gene_map: Dict,          # Ensembl ID -> Gene Symbol
                 variant_embed_generator: VariantEmbeddingGenerator,
                 embedding_type: str,
                 num_top_genes: int = 2047
                 ):
        logger.info(f"Loading AnnData from {h5ad_path}...")
        self.adata = ad.read_h5ad(h5ad_path)

        self.token_dictionary = token_dictionary
        self.variant_embed_generator = variant_embed_generator
        self.embedding_type = embedding_type
        self.num_top_genes = num_top_genes
        
        self.gene_symbol_to_token_id = {}
        adata_gene_symbols = set(self.adata.var_names)
        self.known_gene_symbols = []
        for ensembl_id, token_id in self.token_dictionary.items():
            gene_symbol = gene_map.get(ensembl_id)
            if gene_symbol and gene_symbol in adata_gene_symbols:
                self.gene_symbol_to_token_id[gene_symbol] = token_id
                self.known_gene_symbols.append(gene_symbol)

        self.known_gene_indices = np.where(np.isin(self.adata.var_names, self.known_gene_symbols))[0]
        
        logger.info(f"매핑 완료: {len(self.known_gene_symbols)}개의 유효한 유전자(모델과 데이터에 공통)를 찾았습니다.")

        self.tp53_name = 'TP53'
        self.tp53_token_id = self.gene_symbol_to_token_id.get(self.tp53_name)
        
        tp53_index_list = np.where(self.adata.var_names == self.tp53_name)[0]
        self.tp53_anndata_idx = tp53_index_list[0] if len(tp53_index_list) > 0 else -1

        if self.tp53_token_id is None:
             logger.warning(f"Warning: {self.tp53_name} is NOT found in the token dictionary.")
        if self.tp53_anndata_idx == -1:
            logger.warning(f"{self.tp53_name} not found in adata.var_names.")
        
        if 'variant' in self.adata.obs.columns:
            self.variant_list = self.adata.obs['variant'].astype(str).tolist()
        else:
            self.variant_list = self.adata.obs['condition'].astype(str).tolist()
          
        logger.info(f"Loaded AnnData with {self.adata.n_obs} cells and {self.adata.n_vars} genes.")


    def __len__(self):
        return self.adata.n_obs
 
    def __getitem__(self, idx: int) -> Dict:

        input_expression = self.adata.X[idx].toarray().flatten() 
        full_labels = np.zeros_like(input_expression)
        
        filtered_labels = torch.tensor(full_labels[self.known_gene_indices], dtype=torch.float32)
        subset_values = input_expression[self.known_gene_indices]
        nonzero_mask = subset_values > 0
        
        local_nonzero_indices = np.where(nonzero_mask)[0]
        sorted_local_order = np.argsort(subset_values[local_nonzero_indices])[::-1]
        sorted_expressed_indices = self.known_gene_indices[local_nonzero_indices][sorted_local_order]

        if len(sorted_expressed_indices) >= self.num_top_genes:
            final_gene_indices = sorted_expressed_indices[:self.num_top_genes]

            if self.tp53_anndata_idx != -1:
                if self.tp53_anndata_idx not in final_gene_indices:
                    final_gene_indices[-1] = self.tp53_anndata_idx
        else:
            num_to_fill = self.num_top_genes - len(sorted_expressed_indices)
            
            zero_global_indices = self.known_gene_indices[~nonzero_mask]
            is_tp53_in_expressed = (self.tp53_anndata_idx in sorted_expressed_indices)
            
            filler_indices = []
            if not is_tp53_in_expressed and self.tp53_anndata_idx != -1:
                filler_indices.append(self.tp53_anndata_idx)
                num_to_fill -= 1
                
                zero_global_indices = zero_global_indices[zero_global_indices != self.tp53_anndata_idx]
            
            if num_to_fill > 0:
                if len(zero_global_indices) >= num_to_fill:
                    random_filler = np.random.choice(zero_global_indices, size=num_to_fill, replace=False)
                else:
                    random_filler = np.random.choice(zero_global_indices, size=num_to_fill, replace=True)
                
                filler_indices.extend(random_filler.tolist())

            final_gene_indices = np.concatenate([sorted_expressed_indices, np.array(filler_indices).astype(int)])

        top_gene_names = self.adata.var_names[final_gene_indices]
        top_values = input_expression[final_gene_indices]
        input_ids = [self.gene_symbol_to_token_id[name] for name in top_gene_names]
        values = top_values.tolist()
        variant_str = self.variant_list[idx]
        variant_embedding = self.variant_embed_generator.get_embedding(variant_str, embedding_type=self.embedding_type)

        try:
            tp53_pos = input_ids.index(self.tp53_token_id)
        except ValueError:
            tp53_pos = -1
        
        if 'REF' not in variant_str and tp53_pos == -1:
            raise ValueError(f"CRITICAL: Sample {idx} ({variant_str}) missing TP53 in input_ids!")

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "values": torch.tensor(values, dtype=torch.float32),
            "variant_embedding": variant_embedding,
            "tp53_index": torch.tensor(tp53_pos, dtype=torch.long), 
            "labels": filtered_labels
        }
    
class PerturbationDataCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict]) -> Dict:
        batch = {}
        max_len = max(len(f["input_ids"]) for f in features) if features else 0
        
        input_ids_padded, values_padded, attention_mask_padded = [], [], []

        for f in features:
            padding_length = max_len - len(f["input_ids"])
            input_ids_padded.append(torch.cat([f["input_ids"], torch.tensor([self.pad_token_id] * padding_length, dtype=torch.long)]))
            values_padded.append(torch.cat([f["values"], torch.zeros(padding_length, dtype=torch.float32)]))
            attention_mask_padded.append(torch.cat([torch.ones(len(f["input_ids"])), torch.zeros(padding_length)]))
            
        batch["input_ids"] = torch.stack(input_ids_padded)
        batch["values"] = torch.stack(values_padded)
        batch["attention_mask"] = torch.stack(attention_mask_padded)
        batch["variant_embedding"] = torch.stack([f["variant_embedding"] for f in features])
        batch["tp53_index"] = torch.stack([f["tp53_index"] for f in features])
        batch["labels"] = torch.stack([f["labels"] for f in features])
        
        return batch

