# GeneCompass Variant Perturbation Prediction

This repository provides the implementation for fine-tuning and running inference with **GeneCompass**, a large-scale biological language model (BERT-based), tailored for predicting single-cell transcriptomic responses to genetic variants. This version integrates prior biological knowledge and protein language model (PLM) embeddings specifically into gene tokens (e.g., TP53) to model mutational effects.

---

## Environment Settings

### Conda Environment (Recommended)
```bash
# Create and activate environments
conda create -n genecompass_env python=3.10
conda activate genecompass_env

# Install PyTorch with CUDA support (Match your CUDA version)
pip install torch torchvision torchaudio
```

### Required Packages
Install dependencies from the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```
*Note: This project requires the `genecompass` model library and specific versions of HuggingFace `transformers` (v4.30.0).*

---

## Data

### Data Format

1. **Perturb-seq Data (`.h5ad`)**
   - AnnData objects containing gene expression. Metadata must include a `variant` column for mutational labels.

2. **Token Dictionary (`.pickle`)**
   - A mapping of Ensembl IDs to token indices used by GeneCompass.

3. **Prior Knowledge Embeddings**
   - Pre-computed embeddings for gene relationships: Promoter similarity, Co-expression, Gene Family, PECA GRN, and Human-Mouse homology.

4. **Variant Embeddings Cache (`.pkl`)**
   - Dictionary: `{ (Gene, Mutation): {'ALT': vector, 'DIFF': vector} }`.
   - Supports PLMs: `esm2`, `protT5`, `msa`, `pglm`, `ankh`.

### Data Directory Structure
```text
data_path/
├── hct116_train_1-3.h5ad
├── hct116_valid_1-3.h5ad
├── hct116_test_1-3.h5ad
└── prior_knowledge/
    └── human_mouse_tokens.pickle

finetuned_model/
└── checkpoint_1217/           # Directory for saved model checkpoints
```

---

## Implementation

### 1. Training & Evaluation (End-to-End)

The `main.py` script performs fine-tuning using the HuggingFace `Trainer` API. It incorporates early stopping and saves the best model based on validation loss.

```bash
python main.py \
    --mode single_pipeline \
    --cell_line hct116 \
    --plm esm2 \
    --emb_type alt \
    --num_fold 1-3 \
    --date 1217 \
    --num_train_epochs 10 \
    --train_batch_size 4 \
    --learning_rate 1e-6 \
    --num_top_genes 2047 \
    --fp16
```

#### Key Parameters
- `--pretrained_model`: Path to the base GeneCompass weights (e.g., `genecompass_small`).
- `--num_top_genes`: Number of highly variable genes to include in the sequence (default: 2047).
- `--num_fold`: Dataset split index (e.g., `1-3`).
- `--emb_type`: Use `alt` (mutant sequence) or `diff` (mutant - wildtype) embeddings.

#### Output Files
- **Fine-tuned Weights**: Saved in `{finetuned_model}/checkpoint_{date}/{run_id}-{best_epoch}/`.
- **Loss Plots**: Training/Validation loss curves saved in `{train_result}/loss_plot_{date}/`.
- **Test Results**: Ground truth and predicted AnnData files saved in `{anndata_save}/eval_data_{date}/`.

---

### 2. Inference on Unseen Datasets

Use `inference.py` to run predictions on unseen variants (e.g., OncoKB chunks) where paired ground truth is not required.

```bash
python inference.py \
    --mode inference \
    --date 1217 \
    --finetuned_model /path/to/GeneCompass \
    --eval_batch_size 20 \
    --num_top_genes 2047
```

#### Key Parameters
- `--finetuned_model`: Root directory where the script will search for folders matching the `ep{N}-{BestEpoch}` pattern.
- `--date`: The date tag used to locate the specific checkpoint folder.

#### Output Files
- **Predictions**: `{timestamp}_{cell}_{plm}_{type}_{fold}_{chunk}_pred.h5ad` containing predicted expression values restored to the full gene space and aligned with master metadata.

---

## Model Architecture

### GeneCompass for Variants

- **Encoder**: BERT-style Transformer architecture pre-trained on massive single-cell datasets.
- **Knowledge Integration**: Infuses prior biological knowledge (GRNs, Homology) into the gene embeddings.
- **Variant Injection**: 
  - Specifically targets the **TP53 token** (via `tp53_token_id`).
  - High-dimensional PLM embeddings are projected and added/concatenated to the TP53 gene token representation before entering the Transformer layers.
- **Prediction Head**: Predicts normalized gene expression values for all input tokens.
- **Post-processing**: Automatically handles the mapping between Ensembl IDs, Gene Symbols, and Token IDs, ensuring output consistency with standard AnnData formats.