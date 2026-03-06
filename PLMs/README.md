# Protein Variant Embedding Pipeline

Generate protein language model (PLM) embeddings for gene variants and
extract **variant‑position embeddings** for downstream models.

------------------------------------------------------------------------

## 1. Workflow

1.  Convert **gene name → UniProt ID**
2.  Retrieve **reference protein sequence**
3.  Apply **variant mutation**
4.  Generate **protein embeddings** using PLMs
5.  Extract **variant-position embeddings (ALT, DIFF)**

Embedding utilities are implemented in `embedding.py`.

------------------------------------------------------------------------

## 2. Supported Models

-   esm2_t33_650M_UR50D\
-   esm_msa1_t12_100M_UR50S\
-   ProtT5-XXL-U50\
-   xTrimoPGLM-10B-MLM\
-   Ankh3-Large

------------------------------------------------------------------------

## 3. Example

``` python
gene_name = "TP53"
variant = "Y220C"

uniprot_id = get_uniprot_id(gene_name)
ref_seq = get_protein_sequence(uniprot_id)

_, ref, pos, alt = parse_variant(variant)
alt_seq = apply_variant(ref_seq, pos, alt)

ref_emb = embed_esm2_t33_650M_UR50D(ref_seq)
alt_emb = embed_esm2_t33_650M_UR50D(alt_seq)
```

------------------------------------------------------------------------

## 4. Variant Representation

Two embeddings are used:

-   **ALT** : variant embedding\
-   **DIFF** : `REF - ALT`

------------------------------------------------------------------------

## 5. Generate Variant Embeddings

``` python
collect_data(
    gene_name="TP53",
    variant="Y220C",
    embedding_model_name="esm2_t33_650M_UR50D",
    mutation_type="amino"
)
```

Embeddings are stored in a **Zarr database**.

------------------------------------------------------------------------

## 6. Extract Variant Position Embedding

``` python
extract_variant_position_embeddings_parallel(
    input_pkl="embedding_cache_[model].pkl",
    output_pkl="embedding_cache_variant_position_[model].pkl",
    max_workers=80
)
```

Output format:

    {
     ('TP53','Y220C'):
       {
         "ALT": [vector],
         "DIFF": [vector]
       }
    }

------------------------------------------------------------------------

## 7. Dependencies

    torch
    transformers
    esm
    zarr
    mygene
    biopython
    numpy
