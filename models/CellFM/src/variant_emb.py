import re
import pickle
import numpy as np
import mindspore as ms
import scanpy as sc
import warnings

def variant_list_from_anndata(adata):

    def clean(v):
        v = str(v)
        return v.split('+', 1)[0] if '+' in v else v
    
    if 'variant' in adata.obs.columns:
        return [clean(v) for v in adata.obs['variant'].astype(str).tolist()]
    
    return [clean(v) for v in adata.obs['condition'].astype(str).tolist()]



def parse_variant_key(s: str):

    s = str(s)
    s = s.split('+', 1)[0] 
    if '~' in s:
        gene, mut = s.split('~', 1)
        return gene, mut
    return s, 'REF'


def build_variant_batch(variant_list, variant_dict, variant_dim, embedding_type, pool='mean'):

    vecs = []
    target_key = embedding_type.upper()
    
    for s in variant_list:
        gene, mut = parse_variant_key(s)
        final_vec = np.zeros(variant_dim, dtype=np.float32) 

        if mut == 'REF':
            pass
        
        else:
            embedding_data = variant_dict.get((gene, mut))

            if embedding_data:
                vec = embedding_data.get(target_key)
                if vec is not None:
                    final_vec = np.asarray(vec, dtype=np.float32)
                else:
                    warnings.warn(f"Variant ({gene}, {mut}) found but '{target_key}' is missing. Using zero vector.")
            else:
                pass

        vecs.append(final_vec)

    batch = np.stack(vecs, axis=0).astype(np.float32)
    batch = batch.reshape(batch.shape[0], 1, variant_dim)
    
    return ms.Tensor(batch, dtype=ms.float32)


def infer_variant_dim_from_pkl(variant_dict: dict) -> int:

    if not variant_dict:
        raise ValueError("Variant dictionary is empty. Cannot infer dimension.")

    first_inner_dict = next(iter(variant_dict.values()))
    if not first_inner_dict:
        raise ValueError("First variant entry has an empty inner dictionary.")

    first_embedding_vec = next(iter(first_inner_dict.values()))
    
    arr = np.asarray(first_embedding_vec)
    
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D vector for embedding, but got shape: {arr.shape}")
        
    dimension = arr.shape[0]
    
    return int(dimension)