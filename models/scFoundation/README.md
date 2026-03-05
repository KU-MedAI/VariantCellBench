# scFoundation Variant Perturbation Prediction

Variant-aware single-cell perturbation prediction using **scFoundation**
integrated with **protein language model (PLM) embeddings**.

------------------------------------------------------------------------

## Environment

``` bash
conda create -n scfoundation_env python=3.10
conda activate scfoundation_env
pip install torch torchvision torchaudio
pip install pandas numpy anndata scanpy torch-geometric tqdm
```

------------------------------------------------------------------------

## Data

### Required Files

    dataset/
    ├── perturb_processed.h5ad
    ├── dataloader/
    │   ├── train.pt
    │   ├── valid.pt
    │   └── test.pt
    └── embedding/
        └── embedding_cache_variant_position_[MODEL].pkl

Additional file:

    OS_scRNA_gene_index.19264.tsv

Used to align dataset gene order with the scFoundation gene vocabulary.

------------------------------------------------------------------------

## Training

``` bash
python train.py
```

Outputs:

    log/{project_name}/
    ├── best.pt
    ├── last.pt
    ├── config.json
    └── model_config.json

------------------------------------------------------------------------

## Testing

``` bash
python test.py
```

Outputs:

    pred_adata.h5ad
    truth_adata.h5ad

------------------------------------------------------------------------

## Inference

``` bash
python inference.py
```

Runs predictions across multiple datasets (e.g. cell lines and folds).

Output:

    {timestamp}_{cell_line}_scFoundation_{fold}.h5ad

------------------------------------------------------------------------

## Model

**scFoundation (MaeAutobin)**\
Transformer-based masked autoencoder for gene expression modeling.

Inputs: - gene expression tokens - positional gene indices - variant
embeddings (ALT / DIFF)

Objective:

    Masked MSE reconstruction

------------------------------------------------------------------------

## Output

Predictions are exported as **AnnData (`.h5ad`)** files compatible with
Scanpy pipelines.
