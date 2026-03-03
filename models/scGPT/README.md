# scGPT Variant Perturbation Prediction

## Environment Setting

### Required Packages

```bash
# PyTorch (CUDA support recommended)
pip install torch torchvision torchaudio

# Data Processing & Single Cell Analysis
pip install pandas numpy anndata scanpy

# Graph Neural Networks (for dataloader)
pip install torch-geometric

# Deep Learning Frameworks for Single Cell
pip install scgpt
pip install gears
```

## Data

### Data Format

The project uses the following data formats:

1. **DataLoader Dictionary** (`.pt` format)
   - PyTorch Geometric: `DataLoader` objects mapped by keys: `train_loader`,`val_loader`, `test_loader`
   - Node features are dynamically updated to a 2-column to a format: `[expression, pert_flag]` (발현값, 교란 정보)

2. **Perturb-seq Data** (`.h5ad` format)
   - Processed AnnData object containing the full gene expression and perturbation metadata.
   - Contains gene names (`adata.var_names`) and perturbation conditions (`adata.obs['condition']`).

3. **Variant Embeddings Cache** (`.pkl` format)
   - Pre-computed embeddings (e.g., ProtT5 representations) for each variant/condition.
   - Mapped to full condition names and projected to the model's embedding size (`embsize`)

4. **Vocabulary & Model Configs** (`.json` format)
   - `vocab.json`: scGPT standard gene vocabulary.
   - `args.json`: Pre-trained model configurations (e.g., `embsize`, `nheads`, `nlayers`)

### Data Directory Structure

```

save/scGPT_human/
├── best_model.pt
├── vocab.json
└── args.json
```

## Implementation

### 1. Training & Testing (End-to-End)

Train the model using the pre-trained scGPT weights, apply variant embeddings, and generate predictions on the test set.

```bash
python main.py \
    --dataloader_path /path/to/dataloader.pt \
    --adata_path /path/to/perturb_processed.h5ad \
    --embedding_cache_path /path/to/variant_embeddings.pkl \
    --model_type protT5 \
    --data_name kim2023_hct116 \
    --output_dir ./outputs \
    --amp
```

#### Key Parameters

- `--dataloader_path`: Path to the `.pt` file containing train/val/test loaders.
- `--adata_path`: Path to the full AnnData (.h5ad) file.
- `--embedding_cache_path`: Path to the .pkl cache file containing variant embeddings.
- `--model_type`: Identifier for the variant model used (e.g., protT5, ESM).
- `--epochs`: Maximum number of training epochs (default: 30).
- `--early_stop`: Patience for early stopping based on validation loss (default: 10).

#### Output files

Training Checkpoints: Saved dynamically with timestamp under `./save/train_{data_name}_{model_type}_{time}/best_model.pt`.

Prediction Results: Evaluates the test set and outputs two AnnData objects in the specified `--output_dir`.

   `predict_anndata_{model_type}.h5ad`: Model's predicted gene expression post-perturbation.

   `truth_anndata_{model_type}.h5ad`: Ground truth gene expression.

### 2. Inference

Generate molecules for new DEG data using the trained model.

```bash
python inference.py \
    --model_checkpoint ./checkpoints/DEG2MOL_best_model.pth \
    --data_type Perturb-seq \
    --num_samples 100 \
    --guidance_scale 3 \
```

#### Key Parameters

- `--data_type`: Data type (`KO`, `KD`, `Perturb-seq`)

#### Output Files

- **Training**: Checkpoint files are saved in `--save_dir`
- **Testing/Inference**: Generated molecule dictionary is saved as a `.pkl` file
  - Filename: `{data_type}_generated_molecules_dict_{guidance_scale}.pkl`
  - Format: `{sample_name}_{idx}: {'generated_mols': [list of Mol objects]}`

### Model Architecture

#### scGPT Transformer Generator

- **Input**: Mapped gene IDs, original gene values (expression), perturbation flags (binary vector), and variant embeddings.
- **Structure**: 
  - Gene embedding & Value embedding.
  - Transformer Encoder blocks (pre-loaded with scGPT human weights).
  - Variant Embedding Projector: A linear layer (`nn.Linear`) that maps high-dimensional variant embeddings (e.g., ProtT5) to the scGPT `embsize` (default: 512).
- **Objective**: Masked Language Modeling (MLM) using Masked Mean Squared Error (`masked_mse_loss`).
