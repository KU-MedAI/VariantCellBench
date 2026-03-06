# PerturbNet Variant Perturbation Prediction

Implementation of **PerturbNet**, a conditional generative model for
predicting single‑cell transcriptomic responses to genetic variants.\
The model combines **PLM-derived variant embeddings**, **scVI latent
cell representations**, and **conditional normalizing flows (CINN)**.

------------------------------------------------------------------------

## Environment

``` bash
conda create -n perturbnet_env python=3.10
conda activate perturbnet_env
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

Key dependencies: `scanpy`, `anndata`, `scvi-tools`, `torch`, `numpy`,
`pandas`, `perturbnet`.

------------------------------------------------------------------------

## Data

### 1. Perturb-seq Data (`.h5ad`)

AnnData containing single‑cell expression.

Required metadata:

-   `condition` -- variant label\
-   `split` -- train/test split

Counts layer:

``` python
adata.layers["counts"]
```

### 2. Variant Embedding Cache (`.pkl`)

Dictionary format:

``` python
{(Gene, Variant): {"ALT": vector, "DIFF": vector}}
```

Supported PLMs:

`esm2`, `protT5`, `msa`, `pglm`, `ankh`

Example structure:

    data/
    ├── perturb_processed_metadata.h5ad
    ├── embedding/
    │   └── embedding_cache_variant_position_[PLM].pkl
    ├── scvi_models/
    └── cinn_models/

------------------------------------------------------------------------

## Training

Training consists of:

1.  **scVI** -- learn cell latent representation\
2.  **PerturbNet (CINN)** -- learn conditional mapping

``` bash
python train.py   --cell_line hct116   --plm esm2   --emb_type diff   --compression position_embedding   --epochs 25   --batch_size 128   --lr 4.5e-6
```

Key arguments:

  Argument          Description
  ----------------- ---------------------------------
  `--plm`           PLM used for variant embeddings
  `--emb_type`      `alt` or `diff` representation
  `--compression`   embedding extraction strategy

Output:

    checkpoint_dir/
    ├── model.pt
    ├── config.json
    └── config.pkl

------------------------------------------------------------------------

## Inference

Generate predicted expression for unseen variants.

``` bash
python test.py   --checkpoint /path/to/model   --plm esm2   --emb_type diff
```

Workflow:

1.  load variant embedding\
2.  sample library-size latent variables\
3.  generate cell latent states via CINN\
4.  decode expression using scVI

Outputs:

    pred_adata.h5ad
    truth_adata.h5ad

------------------------------------------------------------------------

## Model Overview

**Cell Representation**

    gene expression → scVI encoder → latent state (z)

**Variant Representation**

-   `ALT` : mutant embedding\
-   `DIFF` : mutant − WT embedding

**Conditional Flow**

The model learns:

    p(z_cell | variant_embedding)

**Generation**

    variant embedding
          ↓
    conditional flow
          ↓
    latent cell state
          ↓
    scVI decoder
          ↓
    predicted expression
