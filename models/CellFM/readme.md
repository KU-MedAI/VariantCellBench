# CellFM Variant Perturbation Prediction (MindSpore)

This repository contains the **MindSpore** implementation of **CellFM**, a foundation model framework for single-cell gene expression prediction under genetic variant perturbations. It utilizes optimal transport-based flow matching logic to predict the transcriptomic response of cells to specific mutations.

---

## Environment Setting

### Conda Environment (Recommended)
```bash
# Create and activate environment
conda create -n cellfm_ms python=3.9
conda activate cellfm_ms

# Install MindSpore (GPU version recommended)
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.0/MindSpore/gpu/x86_64/cuda-11.6/mindspore_gpu-2.2.0-cp39-cp39-linux_x86_64.whl
```

### Required Packages
Install dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```
*Note: Ensure `mindspore` version is 2.2.0 or higher to maintain compatibility with the provided training wrappers.*

---

## Data

### Data Format

1. **Perturb-seq Data (`.h5ad`)**
   - **Training**: Paired AnnData containing `adata.layers['x']` (control state) and `adata.layers['y']` (perturbed state).
   - **Unseen**: AnnData with single-state expression, processed into chunks (e.g., `perturb_processed_[01].h5ad`).

2. **Variant Embeddings Cache (`.pkl`)**
   - Pre-computed embeddings from Protein Language Models (PLM) such as ESM2, ProtT5, or MSA.
   - Structured as a dictionary: `{ (Gene, Mutation): {'ALT': vector, 'DIFF': vector} }`.

3. **Model Configurations (`config.py`)**
   - Contains model hyperparameters for different scales (e.g., standard `Config` or `Config_80M`).

### Data Directory Structure
```text
/NFS_DATA/samsung/
├── CellFM/
│   ├── checkpoint_1217/         # Training checkpoints (main.py)
│   └── checkpointB_1217/        # Unseen inference checkpoints (inference.py)
└── database/
    ├── gears/
    │   ├── kim2023_hct116_[benchmark][1_3-fold]/
    │   └── embedding/           # PLM .pkl files
    └── benchmark_figure/        # Final evaluated AnnData results
```

---

## Implementation

### 1. Training & Evaluation (End-to-End)

Train the model on paired data and evaluate on the test set. Supports early stopping and periodic checkpoint saving.

```bash
python main.py \
    --mode single_pipeline \
    --data_name hct116 \
    --emb_name msa \
    --emb_type diff \
    --num_fold 1-3 \
    --date 1217 \
    --epoch 30 \
    --batch 8 \
    --npu 0
```

#### Key Parameters
- `--mode`: `train` (train only), `inference` (eval only), or `single_pipeline` (both).
- `--emb_name`: PLM type (`esm2`, `protT5`, `msa`, `pglm`, `ankh`).
- `--num_fold`: Cross-validation fold (e.g., `1-3`, `2-3`).
- `--scheduler`: Use `WarmCosineDecay` for learning rate scheduling.
- `--patience`: Early stopping patience (default: 2).

#### Output Files
- **Checkpoints**: Saved in `{modelpath}/checkpoint_{date}/`.
- **Loss Logs**: CSV and Loss Plot saved in `{workpath}/analyse_{date}/`.
- **Results**: Predicted AnnData objects (`_pred.h5ad`, `_cw.h5ad`) and Truth files.

---

### 2. Inference on Unseen Datasets

Run inference on unseen/out-of-vocabulary (OOV) datasets (e.g., OncoKB chunks) where ground truth is unavailable.

```bash
python inference.py \
    --mode inference \
    --date 1217 \
    --modelpath /path/to/checkpoints \
    --datapath /path/to/datasets \
    --npu 0
```

#### Key Parameters
- `--modelpath`: Root directory containing `checkpointB_{date}`.
- `--unseen_dir`: (Optional) Custom path to unseen `.h5ad` chunks. 
- `--batch_size`: Inference batch size (default: 120).

#### Output Files
- **Inference output**: `{date_time}_{data}_{plm}_{type}_gw_{fold}_{chunk}.h5ad` saved in the specified output directory.
- `gw`: Gene-wise predictions.
- `cw`: Cell-wise predictions.

---

## Model Architecture

### CellFM Transformer (MindSpore)

- **Input**: Gene expression values and high-dimensional PLM variant embeddings.
- **Backbone**: A heavy Transformer Encoder (`enc_dims=1536`, `enc_nlayers=40`) optimized for MindSpore `GRAPH_MODE`.
- **Variant Projector**: Maps protein-level embeddings to the gene expression latent space.
- **Objective**: Flow Matching loss for reconstructing the perturbed state from the control state.
- **TP53 Special Handling**: Includes specific positional indexing for TP53 to enhance mutation-specific sensitivity.