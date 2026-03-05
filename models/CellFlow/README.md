# CellFlow Variant Perturbation Prediction

This project utilizes **CellFlow**, an optimal transport-based flow matching model, to predict single-cell gene expression responses to genetic variant perturbations. It integrates protein language model (PLM) embeddings with scRNA-seq data to model the continuous trajectory of cell states under various mutational conditions.

---

## Environment Setting

### Conda Environment (Recommended)
```bash
# Create a new environment
conda create -n cellflow_env python=3.10
conda activate cellflow_env

# Install JAX with CUDA support (Adjust according to your CUDA version)
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Required Packages
Install dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```
*Note: This project requires a specific version of `cellflow-tools`. Ensure the git-based installation in requirements.txt is performed correctly.*

---

## Data

### Data Format

1. **Perturb-seq Data (`.h5ad`)**
   - **Training/Validation**: Requires AnnData objects with layers `x` (source/control state) and `y` (target/perturbed state).
   - **Unseen Data**: AnnData objects with standard expression matrices. Metadata must include perturbation/condition columns.

2. **Variant Embeddings Cache (`.pkl`)**
   - Dictionary format: `{ (Gene, Mutation): {'ALT': vector, 'DIFF': vector} }`.
   - Supports PLMs: `esm2`, `protT5`, `msa`, `pglm`, `ankh`.

3. **Master Metadata (`.h5ad`)**
   - A reference AnnData file used to align gene lists (`var_names`) and retrieve ground-truth control cells for final reconstruction.

### Data Directory Structure
```text
data_dir/
├── hct116_train_1-3.h5ad
├── hct116_valid_1-3.h5ad
├── hct116_test_1-3.h5ad
└── master/
    └── perturb_processed_hct116.h5ad

emb_dir/
└── embedding_cache_variant_position_[esm2_t33_650M_UR50D].pkl
```

---

## Implementation

### 1. Training & Evaluation (End-to-End)

The `main.py` script handles training, validation with early stopping, and inference on test sets (paired data).

```bash
python main.py \
    --mode pipeline \
    --data_dir ./datasets \
    --emb_dir ./embeddings \
    --output_dir ./output \
    --cell_line hct116 \
    --plm esm2 \
    --emb_type alt \
    --num_fold 1-3 \
    --epochs 20 \
    --batch_size 256 \
    --lr 1e-5
```

#### Key Parameters
- `--mode`: `train` (train only), `inference` (eval only), or `pipeline` (both).
- `--cell_line`: Targeted cell line (e.g., `hct116`, `u2os`).
- `--plm`: Protein Language Model used for embeddings (`esm2`, `msa`, etc.).
- `--emb_type`: Embedding mode (`alt` or `diff`).
- `--patience`: Early stopping patience (default: 10).

#### Output Files
- **Checkpoints**: Best model saved under `output_dir/checkpoints/{run_id}/`.
- **PCA Model**: `pca_model.pkl` (essential for reconstruction).
- **Results**: Predicted AnnData (`{run_id}_pred.h5ad`) saved in `output_dir/results/{run_id}/`.

---

### 2. Inference on Unseen Datasets

Use `inference.py` to predict effects on new datasets where Ground Truth (perturbed state) is unavailable.

```bash
python inference.py \
    --mode inference \
    --data_dir ./datasets \
    --emb_dir ./embeddings \
    --unseen_dir ./datasets/unseen_chunks \
    --train_result ./output/checkpoints \
    --date 20260304 \
    --cell_line u2os
```

#### Key Parameters
- `--unseen_dir`: Path containing unseen `.h5ad` chunks (e.g., OncoKB datasets).
- `--train_result`: Path to the directory containing trained model checkpoints.
- `--date`: Date tag for result versioning.

#### Output Files
- **Predictions**: `{timestamp}_{cell_line}_{plm}_{type}_{fold}_{chunk}_pred.h5ad` (includes predicted expression and original metadata).

---

## Model Architecture

### Optimal Transport Flow Matching (OTFM)

The model learns a conditional vector field that pushes the distribution of control cells toward the distribution of perturbed cells.

- **Encoder**: PCA-based dimensionality reduction (default: 50 components).
- **Conditioning**: High-dimensional PLM embeddings are projected via an MLP and integrated into the flow matching solver using **Attention Token Pooling** and concatenation.
- **Solver**: Uses `diffrax` for ODE integration and `ott-jax` for Optimal Transport-based coupling during training.
- **Reconstruction**: Predicted PCA components are mapped back to the high-dimensional gene space using the reference PCA components and gene means stored during training.