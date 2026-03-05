from __future__ import annotations

# ============================================================
# Standard library imports
# ============================================================
import zarr
import numpy as np
import requests
import mygene
import json
import os
import torch
import sys
import esm
import pickle
import time
from config import DATA_DIR, RAW_DATA_PATH, OUTPUT_DIR
from utils import *
from ProtT5 import *
import logging

# Initialize module-level logger
logger = logging.getLogger(__name__)

# Typing utilities for clearer function signatures
from typing import List, Tuple, Union, Literal, Dict, Any, Optional

# Utilities for filesystem and subprocess handling
import os, subprocess, shutil
from pathlib import Path


# ============================================================
# Zarr file utilities
# ============================================================

def zhas(root, path: str) -> bool:
    """
    Check whether a given path exists inside a Zarr store.

    This function safely checks nested paths in the Zarr hierarchy.

    Parameters
    ----------
    root : zarr.Group
        Root Zarr group.
    path : str
        Path inside the Zarr store.

    Returns
    -------
    bool
        True if the path exists, otherwise False.
    """
    try:
        _ = root[path]
        return True
    except KeyError:
        return False


def zread_text(root, path: str, encoding: str = "utf-8") -> str:
    """
    Read a text dataset stored in Zarr and return it as a string.

    The text is expected to be stored as a 1D uint8 array.

    Parameters
    ----------
    root : zarr.Group
        Root Zarr group.
    path : str
        Dataset path inside the Zarr store.
    encoding : str
        Encoding used to decode the text (default: utf-8).

    Returns
    -------
    str
        Decoded text content.
    """
    arr = root[path][...]
    return bytes(arr.tolist()).decode(encoding)


def zwrite_text(root, path: str, text: str, encoding: str = "utf-8", **create_kwargs):
    """
    Save text into a Zarr dataset as a 1D uint8 array.

    If the dataset already exists, it will be overwritten.

    Parameters
    ----------
    root : zarr.Group
        Root Zarr group.
    path : str
        Target dataset path.
    text : str
        Text to be stored.
    encoding : str
        Encoding used to convert the string to bytes.
    create_kwargs :
        Additional arguments passed to `create_dataset`
        (e.g., chunks, compressor).

    Returns
    -------
    zarr.Array
        Created dataset object.
    """
    data = np.frombuffer(text.encode(encoding), dtype=np.uint8)

    # Automatically create parent group if it does not exist
    parent = "/".join(path.split("/")[:-1])
    if parent:
        root.require_group(parent)

    # Remove existing dataset if present
    try:
        del root[path]
    except Exception:
        pass

    # Create dataset and write data
    ds = root.create_dataset(path, shape=data.shape, dtype=data.dtype, **create_kwargs)
    ds[...] = data
    ds.attrs["encoding"] = encoding

    return ds


# ============================================================
# Ensure output directory exists
# ============================================================

# Create the output directory if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


import json
import os
import time
import shutil
import logging
import mygene


def get_uniprot_id(
    gene_names,
    cache_file=os.path.join(DATA_DIR, "gene_cache.json"),
    manual_mapping_file=os.path.join(DATA_DIR, "manual_mapping.json"),
    use_manual_mapping=False,
    overwrite=False,
):
    """
    Retrieve UniProt IDs for given gene names.

    The function first checks a local cache file and an optional manual
    mapping file. If the gene is not found locally, it queries the
    MyGeneInfo API. By default, the function operates in read-only mode
    and only updates the cache file when `overwrite=True`.

    Parameters
    ----------
    gene_names : str or list[str]
        Gene name or list of gene names to query.

    cache_file : str, optional
        Path to the UniProt ID cache file
        (default: DATA_DIR/gene_cache.json).

    manual_mapping_file : str, optional
        Path to a user-defined manual mapping file
        (default: DATA_DIR/manual_mapping.json).

    use_manual_mapping : bool, optional
        Whether to use the manual mapping file.

    overwrite : bool, optional
        If True, updated cache values will be written to disk.
        If False, the cache is updated only in memory.

    Returns
    -------
    dict or str
        If input is a single gene name, returns a single UniProt ID (or None).
        If input is a list, returns a dictionary mapping gene names to UniProt IDs.

    Notes
    -----
    - If the cache file is corrupted, it will be ignored and replaced with an empty cache.
    - If the API query fails or no ID is found, None is returned.
    - When multiple UniProt entries exist, Swiss-Prot is prioritized over TrEMBL.
    """

    logger = logging.getLogger(__name__)

    # Normalize input to list
    if isinstance(gene_names, str):
        gene_names = [gene_names]

    # --------------------------------------------------
    # Load cache and manual mapping files
    # --------------------------------------------------
    gene_cache = {}
    manual_mapping = {}

    # Load existing cache
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            try:
                gene_cache = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Cache file may be corrupted: {cache_file}. Using empty cache.")
                gene_cache = {}

    # Load manual mapping file (optional)
    if use_manual_mapping and os.path.exists(manual_mapping_file):
        with open(manual_mapping_file, "r") as f:
            try:
                manual_mapping = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Manual mapping file may be corrupted: {manual_mapping_file}. Using empty mapping.")
                manual_mapping = {}

    mg = mygene.MyGeneInfo()
    results = {}

    # --------------------------------------------------
    # Retrieve UniProt ID for each gene
    # --------------------------------------------------
    for gene_name in gene_names:

        # Return immediately if present in cache
        if gene_name in gene_cache:
            results[gene_name] = gene_cache[gene_name]
            continue

        # Use manual mapping if available
        if use_manual_mapping and gene_name in manual_mapping:
            logger.info(f"[Manual mapping] {gene_name} -> {manual_mapping[gene_name]}")
            gene_cache[gene_name] = manual_mapping[gene_name]
            results[gene_name] = manual_mapping[gene_name]
            continue

        # Query MyGeneInfo API
        try:
            result = mg.query(gene_name, fields="uniprot", species="human")

            if "hits" in result and result["hits"]:
                uniprot_id = None

                for hit in result["hits"]:
                    uniprot = hit.get("uniprot", {})

                    # Handle different formats (dict, str, list)
                    if isinstance(uniprot, dict):
                        uniprot_id = uniprot.get("Swiss-Prot") or uniprot.get("TrEMBL")
                    elif isinstance(uniprot, str):
                        uniprot_id = uniprot

                    if isinstance(uniprot_id, list):
                        uniprot_id = uniprot_id[0]

                    if uniprot_id:
                        gene_cache[gene_name] = uniprot_id
                        results[gene_name] = uniprot_id
                        logger.info(f"[API query] {gene_name} -> {uniprot_id}")
                        break

                if gene_name not in results:
                    logger.warning(f"No UniProt ID found for {gene_name}.")
                    results[gene_name] = None

            else:
                logger.warning(f"No UniProt ID found for {gene_name}.")
                results[gene_name] = None

        except Exception:
            logger.exception(f"Error occurred while querying {gene_name}.")
            results[gene_name] = None

    # --------------------------------------------------
    # Write cache to disk only if overwrite=True
    # --------------------------------------------------
    if overwrite:
        try:
            with open(cache_file, "w") as f:
                json.dump(gene_cache, f)
            logger.info(f"Cache file updated: {cache_file}")
        except Exception:
            logger.exception(f"Failed to save cache file: {cache_file}")

    # --------------------------------------------------
    # Preserve return format (single vs multiple input)
    # --------------------------------------------------
    return results[gene_names[0]] if len(gene_names) == 1 else results


def get_protein_sequence(uniprot_id):
    """
    Retrieve a protein sequence from UniProt using the REST API.

    Parameters
    ----------
    uniprot_id : str
        UniProt accession ID (e.g., "P04637").

    Returns
    -------
    str or None
        Protein sequence if the request is successful,
        otherwise None.

    Notes
    -----
    The function requests the FASTA record from the UniProt REST API
    and extracts the amino acid sequence by removing the FASTA header.
    """

    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"

    try:
        response = requests.get(url)

        if response.status_code == 200:
            # Split FASTA text into lines
            lines = response.text.split("\n")

            # Remove header lines (starting with '>') and join sequence
            sequence = "".join(
                line.strip() for line in lines if not line.startswith(">")
            )

            return sequence

        else:
            logger.error(f"UniProt response error: {uniprot_id} (status {response.status_code})")

    except Exception:
        logger.exception(f"Failed to retrieve UniProt sequence: {uniprot_id}")

    return None


def parse_variant(variant: str):
    """
    Parse variant notation into a normalized representation.

    Supported formats
    -----------------
    Amino acid variants
        - 1-letter:  Y220C, p.Y220C
        - 3-letter:  p.Tyr220Cys
        - Stop:      Q331Ter, Y220*, Y220X, p.Gln331*, p.Tyr220Stop

    DNA variants
        - HGVS-like: c.123A>G, g.123A>G
        - Compact:   A123G

    Parameters
    ----------
    variant : str
        Variant notation string.

    Returns
    -------
    tuple
        (level, ref, pos, alt)

        level : {"amino", "dna"}
        ref   : reference residue/base
        pos   : integer position (1-based)
        alt   : alternate residue/base
    """

    v = variant.strip()

    # --------------------------------------------------
    # Amino acid variant: 1-letter code
    # Example: Y220C, p.Y220C, Y220*, Y220X
    # --------------------------------------------------
    m = re.match(r'^(?:p\.)?([A-Za-z\*])(\d+)([A-Za-z\*]|Ter|Stop|X)$', v, flags=re.IGNORECASE)
    if m:
        ref_raw, pos, alt_raw = m.group(1), int(m.group(2)), m.group(3)

        ref = normalize_stop_token(ref_raw)
        alt = normalize_stop_token(alt_raw)

        if ref == "*" or len(ref) != 1 or (alt != "*" and len(alt) != 1):
            raise ValueError(f"Unsupported amino acid notation: {variant}")

        return "amino", ref, pos, alt

    # --------------------------------------------------
    # Amino acid variant: 3-letter code
    # Example: p.Tyr220Cys, Q331Ter
    # --------------------------------------------------
    m = re.match(r'^(?:p\.)?([A-Za-z]{3})(\d+)([A-Za-z]{3}|Ter|Stop)$', v, flags=re.IGNORECASE)
    if m:
        ref3, pos, alt3 = m.group(1), int(m.group(2)), m.group(3)

        ref1 = aa3_to_1(ref3)
        alt1 = "*" if alt3.upper() in STOP_SYNONYMS_3 else aa3_to_1(alt3)

        if not ref1 or not alt1:
            raise ValueError(f"Unsupported 3-letter AA notation: {ref3}->{alt3}")

        if ref1 == "*":
            raise ValueError(f"Reference cannot be stop codon: {variant}")

        return "amino", ref1, pos, alt1

    # --------------------------------------------------
    # 3. DNA variant: HGVS-like notation
    # Example: c.123A>G, g.123A>G
    # --------------------------------------------------
    m = re.match(r'^(?:[cg]\.)?(\d+)([ACGT])>([ACGT])$', v, flags=re.IGNORECASE)
    if m:
        pos, ref, alt = int(m.group(1)), m.group(2).upper(), m.group(3).upper()
        return "dna", ref, pos, alt

    # --------------------------------------------------
    # 4. DNA variant: compact notation
    # Example: A123G
    # --------------------------------------------------
    m = re.match(r'^([ACGT])(\d+)([ACGT])$', v, flags=re.IGNORECASE)
    if m:
        ref, pos, alt = m.group(1).upper(), int(m.group(2)), m.group(3).upper()
        return "dna", ref, pos, alt

    raise ValueError(f"Unsupported variant notation: {variant}")


def apply_variant(sequence, position, variant):
    """
    Apply a simple substitution or stop mutation to a sequence.

    Parameters
    ----------
    sequence : str
        Input sequence (protein or DNA).
    position : int
        1-based position of the mutation.
    variant : str
        Alternate residue/base.

    Returns
    -------
    str
        Sequence with the variant applied.
    """

    try:
        # Convert 1-based index to Python 0-based index
        position = int(position) - 1

        # Stop mutation: truncate sequence at mutation position
        if variant == "*":
            return sequence[:position]

        # Substitute residue if within valid range
        if 0 <= position < len(sequence):
            new_seq = list(sequence)
            new_seq[position] = variant
            return "".join(new_seq)

        # Return original sequence if position is out of range
        return sequence

    except Exception as e:
        print(f"Variant application error: {e}")
        return sequence


def apply_variant_amino(seq: str, pos: int, ref: str, alt: str) -> str:
    """
    Apply an amino acid variant to a protein sequence.

    Supported operations
    --------------------
    - Insertion:  ref == '' or '-'
    - Deletion:   alt == '' or '-'
    - Substitution or block substitution
    - Stop mutation (alt='*'): truncate sequence at pos

    Parameters
    ----------
    seq : str
        Protein sequence.
    pos : int
        1-based position.
    ref : str
        Reference residue(s).
    alt : str
        Alternate residue(s).

    Returns
    -------
    str
        Modified sequence.
    """

    if not isinstance(seq, str):
        raise TypeError("seq must be str")

    if pos < 1:
        raise IndexError(f"pos={pos} out of range (must be >=1)")

    ref = (ref or "").upper()
    alt = (alt or "").upper()
    seq_u = seq.upper()

    # --------------------------------------------------
    # Insertion
    # --------------------------------------------------
    if ref in ("", "-"):
        if pos > len(seq) + 1:
            raise IndexError(f"insertion pos={pos} out of range (len={len(seq)})")
        return seq[:pos-1] + alt + seq[pos-1:]

    # Position must be within sequence
    if pos > len(seq):
        raise IndexError(f"pos={pos} out of range (len={len(seq)})")

    end = pos - 1 + len(ref)

    if end > len(seq):
        raise IndexError(
            f"ref extends past sequence end: pos={pos}, len(ref)={len(ref)}, len(seq)={len(seq)}"
        )

    # Verify reference sequence
    if seq_u[pos-1:end] != ref:
        raise ValueError(f"Ref mismatch at {pos}: expected {ref}, found {seq_u[pos-1:end]}")

    # --------------------------------------------------
    # Deletion
    # --------------------------------------------------
    if alt in ("", "-"):
        return seq[:pos-1] + seq[end:]

    # --------------------------------------------------
    # Stop mutation (nonsense)
    # --------------------------------------------------
    if alt in {"*", "Ter"}:
        return seq[:pos-1]

    # --------------------------------------------------
    # Substitution / block substitution
    # --------------------------------------------------
    return seq[:pos-1] + alt + seq[end:]


def apply_variant_dna(seq: str, pos: int, ref: str, alt: str) -> str:
    """
    Apply a DNA variant to a nucleotide sequence.

    Simplified HGVS/VCF-like rules.

    Supported operations
    --------------------
    - Insertion:  ref == '' or '-'
    - Deletion:   alt == '' or '-'
    - Substitution / block substitution

    Parameters
    ----------
    seq : str
        DNA sequence.
    pos : int
        1-based position.
    ref : str
        Reference base(s).
    alt : str
        Alternate base(s).

    Returns
    -------
    str
        Modified DNA sequence.
    """

    if not isinstance(seq, str):
        raise TypeError("seq must be str")

    if pos < 1:
        raise IndexError(f"pos={pos} out of range (must be >=1)")

    ref = (ref or "").upper()
    alt = (alt or "").upper()
    seq_u = seq.upper()

    # --------------------------------------------------
    # Insertion
    # --------------------------------------------------
    if ref in ("", "-"):
        if pos > len(seq) + 1:
            raise IndexError(f"insertion pos={pos} out of range (len={len(seq)})")
        return seq[:pos-1] + alt + seq[pos-1:]

    # Position must be valid
    if pos > len(seq):
        raise IndexError(f"pos={pos} out of range (len={len(seq)})")

    end = pos - 1 + len(ref)

    if end > len(seq):
        raise IndexError(
            f"ref extends past sequence end: pos={pos}, len(ref)={len(ref)}, len(seq)={len(seq)}"
        )

    # Reference validation
    if seq_u[pos-1:end] != ref:
        raise ValueError(f"Ref mismatch at {pos}: expected {ref}, found {seq_u[pos-1:end]}")

    # --------------------------------------------------
    # Deletion
    # --------------------------------------------------
    if alt in ("", "-"):
        return seq[:pos-1] + seq[end:]

    # --------------------------------------------------
    # Substitution / block substitution
    # --------------------------------------------------
    return seq[:pos-1] + alt + seq[end:]



import re
from functools import lru_cache
from typing import Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModel

# ============================================================
# Ankh model registry
# ============================================================
# Registry of available Ankh protein language models on HuggingFace.
# Each entry maps a human-readable model name to its HF repository.
ANKH_MODELS: Dict[str, Dict[str, str]] = {

    # ------------------------------------------------------------
    # Ankh v1 models
    # ------------------------------------------------------------
    # Characteristics:
    # - Requires `sentencepiece`
    "Ankh-Base": {
        "hf_name": "ElnaggarLab/ankh-base",
    },

    "Ankh-Large": {
        "hf_name": "ElnaggarLab/ankh-large",
    },

    # ------------------------------------------------------------
    # Ankh v2 / v3 models
    # ------------------------------------------------------------
    # Improved architecture and efficiency
    "Ankh2-Large": {
        "hf_name": "ElnaggarLab/ankh2-large",
    },

    "Ankh3-Large": {
        "hf_name": "ElnaggarLab/ankh3-large",
    },

    "Ankh3-XL": {
        "hf_name": "ElnaggarLab/ankh3-xl",
    },
}


# ============================================================
# Load encoder (cached)
# ============================================================
@lru_cache(maxsize=8)
def _load_ankh_encoder(model_id: str, device: str):
    """
    Load an Ankh encoder model and tokenizer.

    Ankh models use an encoder-only representation, therefore
    `T5EncoderModel` is used instead of the full T5 model to
    avoid `decoder_input_ids` related errors.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier.

    device : str
        Device to load the model on ("cpu" or "cuda").

    Returns
    -------
    tokenizer : T5Tokenizer
    encoder : T5EncoderModel
    """

    tokenizer = T5Tokenizer.from_pretrained(
        model_id,
        use_fast=False,  # safer default; can be switched to True if stable
    )

    encoder = T5EncoderModel.from_pretrained(model_id)
    encoder = encoder.to(device).eval()

    return tokenizer, encoder


# ============================================================
# Embedding extraction
# ============================================================
def embed_Ankh(
    sequence: str,
    model_id: str,
    device: str | None = None,
    prefix: str = "[NLU]",
) -> torch.Tensor:
    """
    Generate protein embeddings using an Ankh model.

    Parameters
    ----------
    sequence : str
        Amino acid sequence.

    model_id : str
        HuggingFace model identifier.

    device : str, optional
        Device used for inference ("cpu" or "cuda").
        Automatically selected if not provided.

    prefix : str
        Prefix recommended in the Ankh model card
        (default: "[NLU]").

    Returns
    -------
    torch.Tensor
        Residue-level embeddings with shape (L, D),
        where L is the sequence length and D is the embedding dimension.
    """

    # Select device automatically if not provided
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer, encoder = _load_ankh_encoder(model_id, device)

    # Model card recommends adding an NLU prefix
    input_sequence = prefix + sequence

    encoded = tokenizer(
        input_sequence,
        add_special_tokens=True,
        return_tensors="pt",
        is_split_into_words=False,
    )

    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = encoder(**encoded)

    # last_hidden_state shape: (1, L, D)
    # Remove special tokens ([CLS]/[SEP] equivalent)
    hidden = outputs.last_hidden_state[:, 1:-1, :]

    return hidden.cpu()




# ============================================================
# ProtTrans model registry
# ============================================================

PROTTRANS_MODELS: Dict[str, Dict[str, str]] = {
    "ProtT5": {  # alias
        "hf_name": "Rostlab/prot_t5_xl_half_uniref50-enc",
    },
    "ProtT5-XL-U50": {
        "hf_name": "Rostlab/prot_t5_xl_half_uniref50-enc",
    },
    "ProtT5-XL-BFD": {
        "hf_name": "Rostlab/prot_t5_xl_bfd",
    },
    "ProtT5-XXL-U50": {
        "hf_name": "Rostlab/prot_t5_xxl_uniref50",
    },
    "ProtT5-XXL-BFD": {
        "hf_name": "Rostlab/prot_t5_xxl_bfd",
    },
    "ProtBert-BFD": {
        "hf_name": "Rostlab/prot_bert_bfd",
    },
    "ProtBert": {
        "hf_name": "Rostlab/prot_bert",
    },
    "ProtAlbert": {
        "hf_name": "Rostlab/prot_albert",
    },
    "ProtElectra-Generator-BFD": {
        "hf_name": "Rostlab/prot_electra_generator_bfd",
    },
    "ProtElectra-Discriminator-BFD": {
        "hf_name": "Rostlab/prot_electra_discriminator_bfd",
    },
}


# ============================================================
# Sequence preprocessing
# ============================================================

def _prep_protein_for_prottrans(sequence: str) -> str:
    """
    Prepare protein sequence for ProtTrans models.

    ProtTrans tokenization expects:
    - Rare amino acids U/Z/O/B replaced with X
    - Space-separated amino acids

    Example
    -------
    Input:
        "MTEYKLV"

    Output:
        "M T E Y K L V"
    """
    seq = sequence.strip().upper()
    seq = re.sub(r"[UZOB]", "X", seq)
    return " ".join(list(seq))


# ============================================================
# Model loader (cached)
# ============================================================

@lru_cache(maxsize=16)
def _load_prottrans_model(
    embedding_model_name: str,
    device: str | None = None,
):
    """
    Load a ProtTrans model and tokenizer.

    Parameters
    ----------
    embedding_model_name : str
        Key from PROTTRANS_MODELS.

    device : str, optional
        Device for inference.

    Returns
    -------
    tokenizer
    model
    device
    """

    if embedding_model_name not in PROTTRANS_MODELS:
        raise ValueError(f"Unsupported ProtTrans model: {embedding_model_name}")

    hf_name = PROTTRANS_MODELS[embedding_model_name]["hf_name"]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Detect ProtT5 family
    is_prot_t5 = embedding_model_name.startswith("ProtT5")

    if is_prot_t5:
        # Use encoder-only T5 model
        tokenizer = T5Tokenizer.from_pretrained(
            hf_name,
            use_fast=False,
        )

        model = T5EncoderModel.from_pretrained(hf_name)

    else:
        tokenizer = AutoTokenizer.from_pretrained(
            hf_name,
            do_lower_case=False,
            use_fast=False,
        )

        model = AutoModel.from_pretrained(hf_name)

    model = model.to(device).eval()

    return tokenizer, model, device


# ============================================================
# Special token removal
# ============================================================

def _strip_special_tokens(
    hidden: torch.Tensor,
    input_ids: torch.Tensor,
    tokenizer,
) -> torch.Tensor:
    """
    Remove special tokens (CLS/SEP/PAD/BOS/EOS) from embeddings.

    Parameters
    ----------
    hidden : torch.Tensor
        Shape (1, T, D)

    input_ids : torch.Tensor
        Shape (1, T)

    tokenizer
        HuggingFace tokenizer instance

    Returns
    -------
    torch.Tensor
        Residue-only embeddings with shape (1, L, D)
    """

    assert hidden.size(0) == 1
    assert input_ids.size(0) == 1

    ids = input_ids[0]
    reps = hidden[0]

    # Collect tokenizer special token ids
    special_ids = set()

    for attr in [
        "pad_token_id",
        "cls_token_id",
        "sep_token_id",
        "bos_token_id",
        "eos_token_id",
    ]:
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            special_ids.add(tid)

    if not special_ids:
        return hidden

    mask = torch.ones_like(ids, dtype=torch.bool)

    for tid in special_ids:
        mask &= (ids != tid)

    idx = mask.nonzero(as_tuple=True)[0]

    residue_reps = reps[idx]

    return residue_reps.unsqueeze(0)


# ============================================================
# Embedding extraction
# ============================================================

def embed_ProtTrans(
    sequence: str,
    *,
    embedding_model_name: str = "ProtT5",
    per_residue: bool = True,
    device: str | None = None,
) -> torch.Tensor:
    """
    Generate protein embeddings using ProtTrans models.

    Parameters
    ----------
    sequence : str
        Amino acid sequence.

    embedding_model_name : str
        Model key defined in PROTTRANS_MODELS.

    per_residue : bool
        If True, return residue-level embeddings.
        If False, return pooled sequence embedding.

    device : str, optional
        Device for inference.

    Returns
    -------
    torch.Tensor

        per_residue=True
            (1, L, D)

        per_residue=False
            (1, D)
    """

    tokenizer, model, device = _load_prottrans_model(
        embedding_model_name,
        device=device,
    )

    prepped = _prep_protein_for_prottrans(sequence)

    encoded = tokenizer(
        prepped,
        add_special_tokens=True,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    hidden = outputs.last_hidden_state

    residue_hidden = _strip_special_tokens(
        hidden,
        input_ids,
        tokenizer,
    )

    if not per_residue:
        return residue_hidden.mean(dim=1, keepdim=True)

    return residue_hidden






# ============================================================
# xTrimoPGLM embedding
# ============================================================

from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForMaskedLM

# ------------------------------------------------------------
# Model loader (cached)
# ------------------------------------------------------------
@lru_cache(maxsize=4)
def _load_xtrimo_mlm(
    model_id: str,
    device: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
):
    """
    Load xTrimoPGLM / ProteinGLM MLM model and tokenizer.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier.
        Example:
            "biomap-research/xtrimopglm-1b-mlm"
            "biomap-research/proteinglm-1b-mlm"

    device : str, optional
        Device used for inference.

    dtype : torch.dtype
        Model weight precision.

    Returns
    -------
    tokenizer
    model
    device
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=True,
    )

    model = AutoModelForMaskedLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)

    model.eval()

    return tokenizer, model, device


# ------------------------------------------------------------
# Embedding extraction
# ------------------------------------------------------------
def embed_xTrimoPGLM(
    sequence: str,
    *,
    model_id: str = "biomap-research/xtrimopglm-1b-mlm",
    device: str | None = None,
) -> torch.Tensor:
    """
    Extract residue-level embeddings from xTrimoPGLM / ProteinGLM.

    Parameters
    ----------
    sequence : str
        Amino acid sequence.

    model_id : str
        HuggingFace model identifier.

    device : str, optional
        Device used for inference.

    Returns
    -------
    torch.Tensor
        Residue-level embeddings with shape:

            (1, L, D)
    """

    tokenizer, model, device = _load_xtrimo_mlm(
        model_id,
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=torch.bfloat16,
    )

    encoded = tokenizer(
        sequence,
        add_special_tokens=True,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_last_hidden_state=True,
        )

    hidden_states = outputs.hidden_states[:-1, 0].unsqueeze(0)

    return hidden_states


# ============================================================
# A3M utilities
# ============================================================

def _parse_a3m_to_list(a3m_text: str):
    """
    Parse A3M text into list of (name, sequence).
    """

    names, seqs = [], []
    cur_name, buf = None, []

    for line in a3m_text.splitlines():

        if not line:
            continue

        if line.startswith(">"):
            if cur_name is not None:
                seqs.append("".join(buf))

            cur_name = line[1:].strip()
            names.append(cur_name)
            buf = []

        else:
            buf.append(line.strip())

    if cur_name is not None:
        seqs.append("".join(buf))

    return list(zip(names, seqs))


def _format_a3m(names_seqs, wrap: int = 80) -> str:
    """
    Convert (name, seq) list back to A3M format.
    """

    out = []

    for name, seq in names_seqs:

        out.append(f">{name}")

        for i in range(0, len(seq), wrap):
            out.append(seq[i : i + wrap])

    return "\n".join(out) + "\n"


def _find_column_for_master_pos(aligned_master: str, pos_1based: int) -> int:
    """
    Find alignment column corresponding to a residue position
    in the master sequence (ignoring gaps).
    """

    cnt = 0

    for col, ch in enumerate(aligned_master):

        if ch != "-":
            cnt += 1

            if cnt == pos_1based:
                return col

    raise IndexError(f"Master position {pos_1based} not found.")

# ------------------------------------------------------------
# Apply variant to A3M alignment
# ------------------------------------------------------------
def apply_variant_on_a3m(
    a3m_text: str,
    pos: int,
    ref: str,
    alt: str,
    *,
    stop_policy: str = "gap",
) -> str:
    """
    Apply variant to the master sequence of an A3M alignment.

    Parameters
    ----------
    pos : int
        1-based residue position.

    ref : str
        Reference amino acid.

    alt : str
        Alternate amino acid or '*'.

    stop_policy : str
        Behavior when encountering stop mutation.

        "gap"
            Replace stop position and all following residues with gaps.

        "star"
            Mark stop position with '*' and mask the rest with gaps.

    Returns
    -------
    str
        Updated A3M text.
    """

    names_seqs = _parse_a3m_to_list(a3m_text)

    if not names_seqs:
        raise ValueError("Empty A3M.")

    master_name, master = names_seqs[0]

    col = _find_column_for_master_pos(master, pos)

    master_ref = master[col]

    if master_ref == "-":
        raise ValueError(f"Alignment column {col} is gap.")

    if master_ref.upper() != ref.upper():
        raise ValueError(
            f"Reference mismatch at pos={pos}: expected {ref}, found {master_ref}"
        )

    master_list = list(master)

    if alt == "*":

        if stop_policy == "star":
            master_list[col] = "*"

        else:
            master_list[col] = "-"

        for j in range(col + 1, len(master_list)):
            if master_list[j] != "-":
                master_list[j] = "-"

    else:

        master_list[col] = alt.upper()

    names_seqs[0] = (master_name, "".join(master_list))

    return _format_a3m(names_seqs)

# ============================================================
# ESM-MSA embedding
# ============================================================

@lru_cache(maxsize=1)
def _load_esm_msa_t12(device: str):

    model, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()

    model = model.to(device).eval()

    batch_converter = alphabet.get_batch_converter()

    return model, batch_converter


def embed_msa_from_a3m_text(
    a3m_text: str,
    layer: int = 12,
    top_n: int | None = None,
    truncate_to: int | None = None,
    device: str | None = None,
):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, batch_converter = _load_esm_msa_t12(device)

    msa = _parse_a3m_to_list(a3m_text)

    if not msa:
        raise ValueError("Empty MSA.")

    names, seqs = zip(*msa)

    names, seqs = list(names), list(seqs)

    if top_n is not None:
        names, seqs = names[:top_n], seqs[:top_n]

    if truncate_to is not None:
        seqs = [s[:truncate_to] for s in seqs]

    lengths = {len(s) for s in seqs}

    if len(lengths) != 1:
        raise ValueError(f"MSA sequence lengths differ: {lengths}")

    batch = list(zip(names, seqs))

    _, _, tokens = batch_converter(batch)

    tokens = tokens.to(device)

    with torch.no_grad():

        out = model(tokens, repr_layers=[layer], return_contacts=False)

    rep = out["representations"][layer][..., 1:, :]

    return rep.cpu()

 

@lru_cache(maxsize=1)
def _load_esm2_t33(device: str):
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    return model, batch_converter


def embed_esm2_t33_650M_UR50D(
    sequence: str,
    layer: int = 33,
    device: str | None = None,
):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, batch_converter = _load_esm2_t33(device)

    data = [("protein1", sequence)]

    _, _, tokens = batch_converter(data)

    tokens = tokens.to(device)

    with torch.no_grad():

        out = model(tokens, repr_layers=[layer], return_contacts=False)

    return out["representations"][layer].cpu()[:, 1:-1, :]

def collect_data(
    gene_name,
    variant,
    embedding_model_name,
    mutation_type,
    database_path=os.path.join(DATA_DIR, "sequence_embedding.zarr"),
    force: bool = False,
):
    """
    Collect sequence data, apply variant, compute embeddings, and store results in a Zarr database.

    Pipeline
    --------
    1. Retrieve UniProt ID from gene name
    2. Load sequence or MSA data
    3. Apply the specified variant
    4. Generate embeddings using the selected PLM
    5. Store the embedding into the Zarr database

    Parameters
    ----------
    gene_name : str
        Gene symbol (e.g., "TP53").

    variant : str
        Variant notation (e.g., "Y220C").

    embedding_model_name : str
        Name of the embedding model.

    mutation_type : str
        Type of mutation input:
            - "amino"
            - "dna"
            - "aminoMSA"

    database_path : str
        Path to the Zarr database.

    force : bool
        If True, overwrite existing embeddings.
    """

    try:

        # ------------------------------------------------------------
        # Retrieve UniProt ID
        # ------------------------------------------------------------
        uniprot_id = get_uniprot_id(gene_name)

        if not uniprot_id:
            print(f"UniProt ID not found for gene: {gene_name}")
            print(f'Suggest adding manual mapping: add_manual_mapping("{gene_name}", "???")')
            return

        root = zarr.open(database_path, mode="a")

        embeddings_path = f"{uniprot_id}/embeddings/{mutation_type}/{variant}/{embedding_model_name}"

        if embeddings_path in root and not force:
            print(f"Embedding already exists: {embeddings_path}")
            return


        # ------------------------------------------------------------
        # Metadata
        # ------------------------------------------------------------
        gene_group = root.require_group(uniprot_id)
        metadata = gene_group.require_group("metadata")

        metadata.attrs["gene_name"] = gene_name
        metadata.attrs["uniprot_id"] = uniprot_id


        # ------------------------------------------------------------
        # Load sequence or MSA
        # ------------------------------------------------------------
        if mutation_type in ("dna", "amino"):

            seq_ds_path = f"{uniprot_id}/sequences/{mutation_type}"

            if zhas(root, seq_ds_path):

                input_text = zread_text(root, seq_ds_path)

            else:

                time.sleep(1.5)  # avoid API rate limit

                input_text = (
                    get_protein_sequence(uniprot_id)
                    if mutation_type == "amino"
                    else get_protein_sequence(uniprot_id) # replace if you need other sequence
                )

                if not input_text:
                    print(f"Failed to retrieve sequence: {uniprot_id} ({mutation_type})")
                    return

                zwrite_text(root, seq_ds_path, input_text, chunks=(4096,))

            input_seq = input_text

        elif mutation_type == "aminoMSA":

            a3m_ds_path = f"{uniprot_id}/sequences/aminoMSA/a3m"

            if zhas(root, a3m_ds_path):
                a3m_text = zread_text(root, a3m_ds_path)
            else:
                raise ValueError(f"MSA data missing: {mutation_type}")

        else:
            raise ValueError(f"Unsupported mutation_type: {mutation_type}")


        # ------------------------------------------------------------
        # Apply variant
        # ------------------------------------------------------------
        if variant != "REF":

            try:

                level, ref, pos, alt = parse_variant(variant)

                if mutation_type == "aminoMSA":

                    if level != "amino":
                        raise ValueError(f"MSA only supports amino variants: {variant}")

                    input_seq = apply_variant_on_a3m(a3m_text, pos, ref, alt)

                    print(f"[MSA] Variant applied: {variant}")

                elif mutation_type == "amino":

                    if level != "amino":
                        raise ValueError(f"Amino variant expected but got {level}: {variant}")

                    input_seq = apply_variant_amino(input_text, pos, ref, alt)

                    print(f"[AA ] Variant applied: {variant}")

                elif mutation_type == "dna":

                    if level != "dna":
                        raise ValueError(f"DNA variant expected but got {level}: {variant}")

                    input_seq = apply_variant_dna(input_text, pos, ref, alt)

                    print(f"[DNA] Variant applied: {variant}")

            except Exception as e:

                print(f"Variant processing error: {e}")
                print(f"Sequence preview: {str(input_seq)[:120]}...")


        # ------------------------------------------------------------
        # Generate embedding
        # ------------------------------------------------------------

        # ===== xTrimoPGLM =====
        if embedding_model_name.startswith("xTrimoPGLM"):

            if mutation_type != "amino":
                raise ValueError("xTrimoPGLM requires amino sequence input.")

            xtrimo_hf_map = {
                "xTrimoPGLM-1B-MLM": "biomap-research/xtrimopglm-1b-mlm",
                "xTrimoPGLM-3B-MLM": "biomap-research/proteinglm-3b-mlm",
                "xTrimoPGLM-10B-MLM": "biomap-research/proteinglm-10b-mlm",
            }

            model_id = xtrimo_hf_map.get(embedding_model_name)

            if model_id is None:
                raise ValueError(f"xTrimoPGLM model mapping missing: {embedding_model_name}")

            embedding = embed_xTrimoPGLM(input_seq, model_id=model_id)


        # ===== ESM2 =====
        elif embedding_model_name == "esm2_t33_650M_UR50D":

            embedding = embed_esm2_t33_650M_UR50D(input_seq)


        # ===== ESM-MSA =====
        elif embedding_model_name == "esm_msa1_t12_100M_UR50S":

            if mutation_type != "aminoMSA":
                raise ValueError("MSA model requires aminoMSA input.")

            embedding = embed_msa_from_a3m_text(
                input_seq,
                layer=12,
                top_n=64,
                truncate_to=512,
            )[:, 0]


        # ===== Ankh =====
        elif embedding_model_name in ANKH_MODELS:

            if mutation_type != "amino":
                raise ValueError("Ankh models require amino sequence input.")

            model_id = ANKH_MODELS[embedding_model_name]["hf_name"]

            embedding = embed_Ankh(input_seq, model_id=model_id)


        # ===== ProtTrans =====
        elif embedding_model_name in PROTTRANS_MODELS:

            if mutation_type != "amino":
                raise ValueError(f"{embedding_model_name} requires amino sequence input.")

            embedding = embed_ProtTrans(
                input_seq,
                embedding_model_name=embedding_model_name,
                per_residue=True,
            )

        else:

            print(f"Unsupported embedding model: {embedding_model_name}")
            return


        # ------------------------------------------------------------
        # Convert tensor → numpy
        # ------------------------------------------------------------
        if isinstance(embedding, torch.Tensor):

            if embedding.dtype == torch.bfloat16:
                embedding = embedding.to(torch.float32)

            embedding = embedding.detach().cpu().numpy()


        # ------------------------------------------------------------
        # Store embedding
        # ------------------------------------------------------------
        embeddings_group = gene_group.require_group("embeddings")

        variant_group = (
            embeddings_group
            .require_group(mutation_type)
            .require_group(variant)
        )

        if embedding_model_name in variant_group:

            if force:

                del variant_group[embedding_model_name]

                print(f"Existing embedding deleted: {uniprot_id}, {variant}, {embedding_model_name}")

            else:

                print(f"Embedding already exists: {uniprot_id}, {variant}, {embedding_model_name}")
                return


        variant_group.create_dataset(
            embedding_model_name,
            shape=embedding.shape,
            dtype=np.float32,
            data=embedding,
        )

        print(
            f"Saved embedding: {uniprot_id}, {variant}, {embedding_model_name}, shape={embedding.shape}"
        )

    except Exception:

        logger.exception(
            f"[collect_data failed] {gene_name}, {variant}, {embedding_model_name}, {mutation_type}"
        )



def load_data(
    gene_name,
    variant,
    embedding_model_name,
    mutation_type,
    database_path=os.path.join(DATA_DIR, "sequence_embedding.zarr"),
    visible=True,
):
    """
    Load precomputed embedding data from a Zarr database.

    The function retrieves the embedding corresponding to a given
    gene, variant, embedding model, and mutation type. If the data
    does not exist in the database, the function returns None.

    Parameters
    ----------
    gene_name : str
        Gene symbol (e.g., "TP53").

    variant : str
        Variant identifier (e.g., "Y220C" or "REF").

    embedding_model_name : str
        Name of the embedding model used during preprocessing.

    mutation_type : str
        Type of mutation input. Supported values typically include:
            - "amino"
            - "dna"
            - "aminoMSA"

    database_path : str
        Path to the Zarr database storing embeddings.

    visible : bool
        If True, log information about the loaded embedding.

    Returns
    -------
    np.ndarray or None
        Loaded embedding array if present, otherwise None.
    """

    try:

        # ------------------------------------------------------------
        # Open Zarr database (read-only)
        # ------------------------------------------------------------
        root = zarr.open(database_path, mode="r")

        # ------------------------------------------------------------
        # Resolve gene name → UniProt ID
        # ------------------------------------------------------------
        uniprot_id = get_uniprot_id(gene_name, use_manual_mapping=True)

        if not uniprot_id:
            logger.error(f"UniProt ID not found for gene: {gene_name}")
            logger.info(f'Suggest adding manual mapping: add_manual_mapping("{gene_name}", "???")')
            return None

        # ------------------------------------------------------------
        # Check if gene exists in the database
        # ------------------------------------------------------------
        if uniprot_id not in root:
            print(f"No data found for gene: {uniprot_id}")
            return None

        gene_group = root[uniprot_id]

        # ------------------------------------------------------------
        # Construct embedding path
        # ------------------------------------------------------------
        embeddings_path = f"embeddings/{mutation_type}/{variant}/{embedding_model_name}"

        if embeddings_path not in gene_group:
            logger.warning(
                f"[Embedding missing] {uniprot_id}, {variant}, {embedding_model_name}, {mutation_type}"
            )
            return None

        # ------------------------------------------------------------
        # Load embedding array
        # ------------------------------------------------------------
        embedding = gene_group[embeddings_path][:]

        if visible:
            logger.info(
                f"[Loaded] {uniprot_id}, {variant}, {embedding_model_name}, {mutation_type}"
            )
            logger.info(f"[Embedding shape] {embedding.shape}")

        return embedding

    except Exception as e:

        logger.exception(
            f"[load_data failed] {gene_name}, {variant}, {embedding_model_name}, {mutation_type}"
        )
        logger.exception(f"[Error message] {e}")

        return None



def preload_embeddings(
    variant_list,
    embedding_model_name,
    mutation_type,
    database_path=os.path.join(DATA_DIR, "sequence_embedding.zarr"),
):
    """
    Preload embeddings for a list of gene–variant conditions from a Zarr database.

    This function loads embeddings into memory to avoid repeated disk access
    during downstream analysis or model inference.

    Parameters
    ----------
    variant_list : list[str]
        List of condition identifiers.
        Example:
            ["KRAS~G12D+ctrl", "KRAS~G13D+ctrl", ..., "ctrl"]

    embedding_model_name : str
        Name of the embedding model used to generate embeddings.

    mutation_type : str
        Mutation input type. Typically one of:
            - "amino"
            - "dna"
            - "aminoMSA"

    database_path : str
        Path to the Zarr embedding database.

    Returns
    -------
    dict
        Dictionary mapping (gene, variant) → embedding array.

        Example:
            {
                ("TP53", "Y220C"): np.ndarray,
                ("TP53", "REF"): np.ndarray
            }

    Notes
    -----
    - 'ctrl' entries are mapped to a fixed zero vector.
    - REF embeddings are also cached when available.
    """

    embedding_cache = {}

    # ------------------------------------------------------------
    # Open Zarr database
    # ------------------------------------------------------------
    try:
        root = zarr.open(database_path, mode="r")
    except Exception:
        logger.exception(f"[Zarr error] Failed to open database: {database_path}")
        return {}

    # ------------------------------------------------------------
    # Iterate over variant list
    # ------------------------------------------------------------
    for gene_var_ in variant_list:

        # --------------------------------------------------------
        # Handle 'ctrl' condition
        # --------------------------------------------------------
        if gene_var_ == "ctrl":

            # WARNING:
            # Different models may have different embedding dimensions.
            # The current implementation assumes a fixed size.
            embedding_cache[("ctrl", "REF")] = np.zeros((1280, 1))

            continue

        # Remove "+ctrl" suffix if present
        if "+" in gene_var_:
            gene_var_ = gene_var_.split("+")[0]

        # --------------------------------------------------------
        # Parse gene and variant
        # --------------------------------------------------------
        if "~" in gene_var_:
            gene, variant = gene_var_.split("~")
        else:
            gene, variant = gene_var_, "REF"

        # --------------------------------------------------------
        # Map gene → UniProt ID
        # --------------------------------------------------------
        uniprot_id = get_uniprot_id(gene, use_manual_mapping=True)

        if not uniprot_id:
            logger.error(f"[UniProt ID missing] {gene}")
            logger.info(f'Suggest manual mapping: add_manual_mapping("{gene}", "???")')
            continue

        if uniprot_id not in root:
            logger.warning(f"[Zarr missing] {uniprot_id} not found in database")
            continue

        group = root[uniprot_id]

        # --------------------------------------------------------
        # Load variant embedding
        # --------------------------------------------------------
        path = f"embeddings/{mutation_type}/{variant}/{embedding_model_name}"

        if path in group:

            embedding = group[path][:]

            embedding_cache[(gene, variant)] = embedding

            logger.debug(f"[Loaded] {gene}~{variant}")

        else:

            logger.warning(f"[Embedding missing] {gene}~{variant}")

        # --------------------------------------------------------
        # Load REF embedding if not already cached
        # --------------------------------------------------------
        ref_path = f"embeddings/{mutation_type}/REF/{embedding_model_name}"

        if (gene, "REF") not in embedding_cache and ref_path in group:

            embedding_cache[(gene, "REF")] = group[ref_path][:]

            logger.debug(f"[REF loaded] {gene}~REF")

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------
    logger.info(f"Preload complete: {len(embedding_cache)} embeddings loaded")

    return embedding_cache

def get_cached_embedding(
    gene_name,
    variant,
    cache,
    representation: str = "DIFF",
    as_tensor: bool = False,
    device=None,
):
    """
    Retrieve an embedding corresponding to (gene_name, variant) from a preloaded cache.

    Parameters
    ----------
    gene_name : str
        Gene name, e.g. 'TP53', 'KRAS', or 'ctrl'.
        If 'ctrl', a special handling rule is applied.

    variant : str or None
        Variant notation, e.g. 'Y220C', 'G12D'.
        For ctrl, None or 'REF' may appear in inputs, but the actual cache key
        always uses ('ctrl', 'REF').

    cache : dict
        Dictionary returned by `preload_embeddings`.
        Typical structure:

            cache[('TP53', 'Y220C')] = {
                'ALT':  np.ndarray(shape=(1280,)),
                'REF':  np.ndarray(shape=(1280,)),
                'DIFF': np.ndarray(shape=(1280,)),
            }

            cache[('ctrl', 'REF')] = np.ndarray(shape=(1280,))  # or dict

    representation : {"REF", "ALT", "DIFF", None}, default="DIFF"
        Specifies which embedding representation to retrieve.

        - "REF"  : reference embedding
        - "ALT"  : variant embedding
        - "DIFF" : difference embedding (typically REF − ALT)
        - None   : return the full dictionary stored in cache

    as_tensor : bool, default=False
        If True, convert the output to a torch.Tensor.

    device : torch.device or str or None, default=None
        Device for the returned tensor when as_tensor=True.
        Example: "cuda", "cuda:0", or "cpu".

    Returns
    -------
    np.ndarray or torch.Tensor or dict or None
        - Returns the requested embedding vector (or dict) if available.
        - Returns None if the key or representation does not exist (warning logged).

    Notes
    -----
    * If gene_name == "ctrl":
        - Always load from key ('ctrl', 'REF').
        - The REF embedding is returned regardless of the representation argument.
        - This acts as the control baseline embedding.
    """

    import numpy as np

    # Internal helper: convert numpy array to torch tensor if requested
    def _to_tensor(x):
        if not as_tensor:
            return x
        import torch

        t = torch.as_tensor(x)
        if device is not None:
            t = t.to(device)
        return t

    # ------------------------------
    # 1) Special handling for ctrl
    # ------------------------------
    if gene_name == "ctrl":
        ctrl_key = ("ctrl", "REF")
        if ctrl_key not in cache:
            logger.warning(f"[Cache miss - ctrl] {ctrl_key}")
            print(f"Cache miss for {ctrl_key}")
            return None

        ctrl_entry = cache[ctrl_key]

        # Handle both dict and raw vector formats
        if isinstance(ctrl_entry, dict):
            if "REF" not in ctrl_entry:
                logger.warning(f"[Cache miss - ctrl REF] {ctrl_key} has no 'REF' field")
                print(f"Cache miss for {ctrl_key}['REF']")
                return None
            vec = ctrl_entry["REF"]
        else:
            vec = ctrl_entry  # assume it is already a vector

        logger.debug(f"[Cache hit - ctrl] {ctrl_key}")
        return _to_tensor(vec)

    # ------------------------------
    # 2) General (gene, variant) case
    # ------------------------------
    key = (gene_name, variant)

    if key not in cache:
        logger.warning(f"[Cache miss] {key}")
        print(f"Cache miss for {key}")
        return None

    entry = cache[key]
    logger.debug(f"[Cache hit] {key}")

    # If representation=None, return the raw dict without tensor conversion
    if representation is None:
        return entry

    # If entry is a dict, select one of 'REF', 'ALT', 'DIFF'
    if isinstance(entry, dict):
        if representation not in entry:
            logger.warning(f"[Cache missing representation] {key} has no '{representation}' field")
            print(f"Cache missing representation '{representation}' for {key}")
            return None
        vec = entry[representation]
    else:
        # If entry is not a dict, treat it as a single vector
        vec = entry

    return _to_tensor(vec)


def save_embedding_cache(cache, filepath):
    """
    Save the embedding cache to a pickle file.

    Parameters
    ----------
    cache : dict
        Dictionary storing embeddings with structure {(gene, variant): embedding}.

    filepath : str
        File path where the pickle file will be saved.

    Notes
    -----
    This function writes the in-memory embedding cache dictionary to disk
    using pickle format. A success message is logged on completion,
    and exceptions are logged if an error occurs.
    """
    try:
        with open(filepath, 'wb') as f:  # open file in binary write mode
            pickle.dump(cache, f)  # serialize cache using pickle
        logger.info(f"✅ Embedding cache successfully saved: {filepath}")
    except Exception:
        logger.exception(f"[Error] Failed to save cache: {filepath}")


def load_embedding_cache(filepath):
    """
    Load an embedding cache from a pickle file.

    Parameters
    ----------
    filepath : str
        Path to the pickle file.

    Returns
    -------
    dict
        Dictionary containing {(gene, variant): embedding}.
        If loading fails, an empty dictionary is returned.

    Notes
    -----
    If the file does not exist or loading fails, the function logs the error
    and returns an empty dictionary.
    """
    try:
        with open(filepath, 'rb') as f:  # open file in binary read mode
            cache = pickle.load(f)  # deserialize cache using pickle
        logger.info(f"✅ Embedding cache successfully loaded: {filepath}")
        return cache
    except Exception:
        logger.exception(f"[Error] Failed to load cache: {filepath}")
        return {}


from torch.nn.functional import pad

def pad_to_length(embedding, target_len):
    """
    Pad or truncate an embedding tensor to match the target sequence length.

    Parameters
    ----------
    embedding : torch.Tensor
        Input tensor with shape (batch, seq_len, dim).

    target_len : int
        Desired sequence length.

    Returns
    -------
    torch.Tensor
        Tensor with sequence length adjusted to target_len.
        If shorter, zero-padding is added.
        If longer, the tensor is truncated.
    """
    pad_len = target_len - embedding.shape[1]  # calculate required padding length

    if pad_len > 0:
        # pad along the sequence length dimension
        return pad(embedding, (0, 0, 0, pad_len))
    else:
        # truncate if sequence is longer than target
        return embedding[:, :target_len, :]