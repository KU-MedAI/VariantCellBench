import os
import gc
import time
import math
import numpy as np
import scipy as sp
import pandas as pd
import scanpy as sc
import mindspore as ms
import mindspore.numpy as mnp
import mindspore.scipy as msc
import mindspore.dataset as ds
import multiprocessing as mp
from tqdm import tqdm,trange
from mindspore import nn,ops
from functools import partial
from multiprocessing import Process,Pool
from scipy.sparse import csr_matrix as csm
import scipy.sparse as sps
from variant_emb import *


class SCrna():
    def __init__(self, 
                path,
                 data, 
                 embedding_type,
                 prep=False, 
                 variant_dict=None, 
                 variant_dim=1280, 
                 pool='mean'):
        
        suffix = data.split('.')[-1]
        if suffix == 'h5ad':
            adata = sc.read_h5ad(f"{path}/{data}")
        else:
            adata = sc.read_10x_h5(f"{path}/{data}")
        
        print(f"Original data shape: {adata.shape}")
        
        pert_col = None
        for col in ['variant', 'pert', 'condition', 'perturbation']:
            if col in adata.obs.columns:
                pert_col = col
                break
                

        self.gene_info = pd.read_csv('./csv/gene_info.csv', index_col=0, header=0) 
        self.geneset = {j:i+1 for i,j in enumerate(self.gene_info.index)} 
        print("gene_info:", self.gene_info.shape)

        gene = np.intersect1d(adata.var_names, self.gene_info.index) 
        self.adata = adata[:,gene].copy() 
        
        print("Final adata shape after filtering:", self.adata.shape)
        
        self.gene = np.array([self.geneset[i] for i in self.adata.var_names]).astype(np.int32)

        try:
            final_gene_list = list(self.adata.var_names)
            tp53_pos_idx = final_gene_list.index('TP53')

            self.tp53_id = self.geneset['TP53'] 
            self.tp53_pos_idx = tp53_pos_idx     
            print(f"INFO: Found TP53. Pos: {self.tp53_pos_idx}, ID: {self.tp53_id}")
        except ValueError:
            self.tp53_id = None 
            self.tp53_pos_idx = None
            print("WARNING: TP53 gene not found.")

        def _to_csr_float32(M):
            if sps.issparse(M):
                return M.tocsr().astype(np.float32)
            return sps.csr_matrix(M.astype(np.float32))

        self.X = _to_csr_float32(self.adata.layers['x'])
        self.Y = _to_csr_float32(self.adata.layers['y'])

        self.Tx = np.asarray(self.X.sum(axis=1)).reshape(-1).astype(np.float32)
        self.Ty = np.asarray(self.Y.sum(axis=1)).reshape(-1).astype(np.float32)

        self.variant_dim = int(variant_dim)
        variant_list = variant_list_from_anndata(self.adata)  
        if variant_dict is not None:
            v_tensor = build_variant_batch(variant_list, variant_dict,
                                            variant_dim=self.variant_dim, pool=pool, embedding_type=embedding_type)
            self.variant_cls_raw = v_tensor.asnumpy() 
        else:
            N = self.adata.n_obs
            self.variant_cls_raw = np.zeros((N, 1, self.variant_dim), dtype=np.float32)


    def __len__(self):
        return self.adata.n_obs

    def __getitem__(self, idx):
        x_row = np.array(self.X[idx].todense()).reshape(-1)
        y_row = np.array(self.Y[idx].todense()).reshape(-1)
        Tx = np.float32(self.Tx[idx])
        Ty = np.float32(self.Ty[idx])
        v_cls = self.variant_cls_raw[idx]
        return x_row, y_row, self.gene, Tx, Ty, v_cls
    

def input_gene_filtering(scrna_data, sample_len, tp53_idx, shuffle=False):

    cell_indices = list(range(len(scrna_data)))
    if shuffle:
        np.random.shuffle(cell_indices)

    master_gene_ids = scrna_data.gene
    mask_gene = np.ones(sample_len, dtype=np.float32)
    zero_idx = np.ones(sample_len + 1, dtype=np.float32)

    for cell_idx in cell_indices:
        x_row, y_row, _, _, _, variant_cls_raw = scrna_data[cell_idx]
        w = np.log1p(x_row.astype(np.float64))
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        w = np.clip(w, 0.0, None)
        
        eps = 1e-12
        w = w + eps

        p = w / w.sum()
        p = p / p.sum()

        all_indices = np.arange(len(x_row))
        selected_indices = np.random.choice(
            all_indices, 
            size=sample_len, 
            replace=False,  
            p=p
        )

        if tp53_idx not in selected_indices:
            selected_indices[-1] = tp53_idx

        x_sampled = x_row[selected_indices].astype(np.float32)
        y_sampled = y_row[selected_indices].astype(np.float32)
        gene_sampled = master_gene_ids[selected_indices].astype(np.int32)

        yield (y_sampled, x_sampled, gene_sampled, mask_gene, zero_idx, variant_cls_raw)

