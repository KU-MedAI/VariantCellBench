# Standard libraries
import sys
import os
import pickle
import numpy as np
import pandas as pd
from scipy import sparse

# Single-cell analysis
import scanpy as sc
import anndata as ad
from anndata import AnnData
import scvi

# Deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F

# Protein language model
import esm

# PerturbNet modules
from perturbnet.util import *
from perturbnet.cinn.flow import *
from perturbnet.genotypevae.genotypeVAE import *
from perturbnet.data_vae.vae import *
from perturbnet.cinn.flow_generate import SCVIZ_CheckNet2Net


def prepare_embeddings_cinn_2(adata, perturbation_key, trt_key, embed_key):
    """
    Prepare perturbation embeddings for CINN (Cell Inference of Network Navigation).

    This function:
    1. Extracts perturbation labels for each cell
    2. Retrieves embeddings for each perturbation type
    3. Creates a mapping between perturbation labels and embedding indices
    """

    # --------------------------------------------------
    # 1. Extract perturbation labels for each cell
    # --------------------------------------------------
    perturb_with_onehot = np.array(adata.obs[perturbation_key])

    # --------------------------------------------------
    # 2. Get unique perturbation types
    # --------------------------------------------------
    trt_list = np.unique(perturb_with_onehot)

    # --------------------------------------------------
    # 3. Find embedding indices corresponding to perturbations
    # --------------------------------------------------
    trt_all = np.array(adata.uns[trt_key])
    emb_all = np.array(adata.uns[embed_key])

    embed_idx = []
    for trt in trt_list:
        idx = np.where(trt_all == trt)[0][0]
        embed_idx.append(idx)

    # --------------------------------------------------
    # 4. Extract embeddings for the selected perturbations
    # --------------------------------------------------
    embeddings = emb_all[embed_idx]

    # --------------------------------------------------
    # 5. Build perturbation → embedding index mapping
    # --------------------------------------------------
    perturbToEmbed = {}
    for i in range(trt_list.shape[0]):
        perturbToEmbed[trt_list[i]] = i

    # --------------------------------------------------
    # 6. Return results
    # --------------------------------------------------
    return perturb_with_onehot, embeddings, perturbToEmbed


# -------------------------------------------------
# Parse condition string → (gene, variant)
# -------------------------------------------------
import re

def parse_condition_to_gene_var(cond: str):

    base = cond.split('+')[0]

    if '~' in base:
        gene, var = base.split('~', 1)
    else:
        m = re.match(r"([A-Za-z0-9]+)_(p\.[A-Za-z0-9]+)", base)
        if m:
            gene, var = m.group(1), m.group(2)
        else:
            raise ValueError(f"Unknown condition string format: {cond}")

    return gene, var

def _get_any_vec_list(embedding_cache, expr_type: str):
    """
    Retrieve any embedding vector corresponding to expr_type.
    Used to determine embedding dimensionality.
    """

    for key, d in embedding_cache.items():

        if expr_type in d:

            vec_list = d[expr_type]

            if vec_list is None:
                continue

            if isinstance(vec_list, (list, tuple)) and len(vec_list) > 0:
                return vec_list

    raise KeyError(f"Embedding vector for expr_type='{expr_type}' not found")


def cache_to_sequence_representations_from_adata(
    adata,
    embedding_cache,
    expr_type="ALT",
    condition_col="condition",
):

    conditions = adata.obs[condition_col].unique().tolist()

    sequence_representations = []

    # Determine embedding dimension
    ref_vec_list = _get_any_vec_list(embedding_cache, expr_type)
    D = len(ref_vec_list)

    for cond in conditions:

        # ctrl condition → zero vector
        if cond == "ctrl":

            vec_arr = np.zeros((D,), dtype=np.float32)
            sequence_representations.append(vec_arr)

            continue

        gene, var = parse_condition_to_gene_var(cond)

        key = (gene, var)

        if key not in embedding_cache:
            raise KeyError(f"{key} not found in embedding_cache")

        if expr_type not in embedding_cache[key]:
            raise KeyError(f"{expr_type} not found for {key}")

        vec_list = embedding_cache[key][expr_type]

        vec_arr = np.asarray(vec_list, dtype=np.float32)

        if vec_arr.shape[0] != D:
            raise ValueError(
                f"Embedding dimension mismatch: expected {D}, got {vec_arr.shape[0]}"
            )

        sequence_representations.append(vec_arr)

    # Convert to (N_conditions, embedding_dim)
    sequence_representations = np.stack(sequence_representations, axis=0)

    return conditions, sequence_representations


def seed_all(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

SAFE_DELIMS = "_+_"

def sanitize_name(s: str) -> str:
    """Remove invalid filename characters."""
    s = re.sub(r'[<>:"/\\|?*]+', '', s)
    s = s.replace(' ', '-')
    return s

def make_project_name(config, timestamp):
    """Generate project name based on configuration."""
    parts = [
        str(config["model"]),
        str(config["data_name"]),
        str(config["embedding_model"]),
        str(config["variant_representation"]),
        str(config["compression"]),
        str(timestamp),
    ]
    return sanitize_name(SAFE_DELIMS.join(parts))

def get_timestamp():
    """Return current timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M")