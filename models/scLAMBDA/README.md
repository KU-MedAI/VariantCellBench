# scLAMBDA Variant Perturbation Prediction

## Environment Setting

### Required Packages

```bash
# PyTorch (CUDA support recommended)
pip install torch torchvision torchaudio

# Data Processing & Single Cell Analysis
pip install pandas numpy scipy anndata scanpy

# Experiment Tracking & Logging
pip install wandb

# Custom scLAMBDA Module
# Ensure the scLAMBDA directory is in your PYTHONPATH or correctly appended in the script.
```

## Data

### Data Format

The project uses the following data formats:

1. **Pre-split AnnData** (`.h5ad` format)
   - Contains single-cell gene expression data and metadata.
   - Required columns: -`obs['split']`: Must contain predefined data splits (`train`, `val`, `test`).
    - `obs['condition']`: Perturbation conditions (e.g., `ctrl`, specific variants).
    - `var['gene_name']`: Gene identifiers.
   - Used for both model training and ground truth evaluation.

2. **Gene/Variant Embeddings** (`.pkl` format)
   - Pre-computed embeddings for genetic variants (e.g., ESM representations).
   - Supports specific representation types based on the `--emb_type` argument.

### Data Directory Structure

```

save/scGPT_human/
├── best_model.pt
├── vocab.json
└── args.json
```

## Implementation

### 1. Training & Testing (End-to-End)

Run the script to train the scLAMBDA model, predict the test set perturbations, append control sample data, and merge metadata into the final AnnData objects.

```bash
python main.py \
    --gene_emb_path /NFS_DATA/.../embedding_cache_variant_position_[esm_msa1_t12_100M_UR50S].pkl \
    --adata_path /NFS_DATA/.../perturb_processed_metadata.h5ad \
    --model_path /NFS_DATA/.../hct116_MSA_ALT \
    --wandb_project "sclambda-project-hct116_MSA_ALT" \
    --wandb_entity "bandalcom-medai" \
    --emb_type "ALT" \
    --epochs 100 \
    --batch_size 16
```

#### Key Parameters

- `--gene_emb_path`: Path to the `.pkl` file containing pre-computed variant/gene embeddings.
- `--adata_path`: Path to the pre-split `.h5ad` AnnData file.
- `--model_path`: Base directory path to save the trained model weights and the final output files.
- `--emb_type`: Specify the embedding type to use.
- `--epochs`: Number of training epochs
- `--batch_size`: Training batch size
- `--wandb_project`: Name of the Weights & Biases project for logging.
- `--wandb_entity`: Your W&B username or team entity name.

#### Output files

Upon successful completion, the script saves the following files in the specified `--model_path` directory:

- Trained Model Weights: Saved natively by the sclambda.model.Model module.

- `pred.h5ad`: The final predicted AnnData object.

    - Contains the model's generated cell expressions (`pred`).

    - Concatenated with the original unperturbed control cells (`ctrl_input`).

    - Enriched with complete metadata (`cell_type`, `variant_count`, `dose_val`, etc.) for downstream visualization.

- `truth.h5ad`: The corresponding ground truth AnnData object.

    - Contains real single-cell expressions for the test conditions.

    - Similarly concatenated with control cells and complete metadata for direct benchmarking against the predictions.

### 2. Inference

Run inference on unseen datasets using the trained model checkpoint.

```bash
python inference.py \
    --emb_type DIFF \
    --gene_emb_path "/NFS_DATA/.../embedding_cache_variant_position_[esm_msa1_t12_100M_UR50S].pkl" \
    --adata_path "/NFS_DATA/.../perturb_processed_[01].h5ad" \
    --ckpt_path "/NFS_DATA/.../hct116_MSA_DIFF/ckpt.pth" \
    --cell_type hct116 \
    --model_name msa \
    --fold 1-3 \
    --data_id 01
```

#### Key Parameters

- `--ckpt_path`: Path to the strictly fine-tuned model weights (`ckpt.pth`) to be manually loaded into the `Net` module.
- `--adata_path`: Path to the target `.h5ad` dataset acting as the template.
- `--gene_emb_path`: Path to the variant embeddings cache.
- `--cell_type`: Target cell line.
- `--model_name`: Base model used for embeddings
- `--fold`: Cross-validation fold identifier
- `--data_id`: Specific dataset ID or benchmark identifier

#### Output Files

- **Condition Validation**: The script parses conditions (e.g., TP53~R273H+...) and validates them against the provided embedding keys.
- **Template Filling**: The original ctrl cells remain unchanged to provide a consistent baseline.
    - Perturbed cells (`condition != 'ctrl'`) are replaced row-by-row with the generated predictions. If the generated cell count doesn't perfectly match the needed count, the script automatically handles it via slicing or random resampling.
- **Saving Results**: * The final template-filled AnnData object is saved in the predefined `OUTPUT_DIR` (e.g., `/NFS_DATA/samsung/database/benchmark_figure/ann_dataset_oncoKB/scLAMBDA`).
    - Output File Format: `1127_2142_{cell_type}_{model_name}_{emb_type}_{fold}_{data_id}_pred.h5ad`

### Model Architecture

#### scLAMBDA Generative Network (sclambda.network.Net)

- **Input**: 
    - Cellular State (`x_dim`): Single-cell gene expression profiles (typically baseline or control cells).
    - Perturbation Embedding (`p_dim`): High-dimensional pre-computed variant embeddings (e.g., ESM, ProtT5). Depending on the `--emb_type` setting, this uses either the mutated sequence embedding directly (`ALT`) or the delta/difference between the reference and mutated embeddings (`DIFF`).
- **Structure**: 
  - Latent Space Mapping: Utilizes a deep neural network architecture configured with hidden layers (`hidden_dim`) to project inputs into a lower-dimensional latent bottleneck (`latent_dim`).
  - Conditional Generation: Integrates the variant embeddings into the latent space to condition the neural network, effectively teaching the model how specific mutations alter the transcriptomic landscape.
- **Objective**: To generate high-fidelity, post-perturbation single-cell expression profiles. The model learns the complex, non-linear transcriptomic response to genetic variants, allowing it to predict unseen mutant responses by simply swapping the input variant embeddings.