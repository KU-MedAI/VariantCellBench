# Variant-Cell-Eval Pipeline

## Description
This pipeline provides a comprehensive suite of metrics for evaluating the performance of models (e.g., CellFM, GeneCompass) that predict cellular responses to genetic variants at the single-cell level. 

Rather than running individual evaluation scripts manually, this pipeline orchestrates multiple evaluation frameworks—**GEARS, Systema, Gene-wise (PCC/RMSE), and AUPRC**—and aggregates their outputs into a single, standardized summary matrix.

## Installation
This pipeline relies on specific Conda environments to manage conflicting dependencies across different evaluation tools.

```bash
# 1. Create the main environment from the YAML file
conda env create -f cell-eval-dev.yml

# 2. Activate the main environment
conda activate cell-eval-dev

# Note: The AUPRC module requires an R-compatible environment. 
# Ensure 'r-cmap-env' is created and available on your system. 
# The script will automatically switch to this environment when running AUPRC.
```
## Usage
To get started, you'll need the predicted outputs and their corresponding ground truth data in `.h5ad` format. The script expects the data to be structured in specific benchmark directories.

### Prep (Data Filtering & Alignment)
Before metric calculation, the pipeline automatically prepares and aligns the predicted and real anndata objects. You do not need to run a separate preparation script. The pipeline natively handles:

1. **Condition Cleaning**: Automatically strips `+ctrl` suffixes from the condition column (e.g., `TP53~A276V+ctrl` → `TP53~A276V`).

2. **Gene Filtering**: Filters the predicted genes to strictly match the `exist` column mask in the truth anndata, ensuring perfectly aligned matrices for evaluation.

### Run
To run the full comprehensive evaluation, use the main python script.

The script will automatically scan the input directory for `_pred.h5ad` files, parse metadata (Cell Line, PLM, Embedding, Fold), match them with the ground truth, and execute the selected suite of metrics.

```bash
python result_summary.py \
    --mode gears_systema \
    --model CellFM \
    --date 260209
```

### Available Modes:
- `--mode gears_systema`: Runs the full comprehensive suite (GEARS, Systema, Gene-wise, and AUPRC) and automatically merges fold results (1-3, 2-3, 3-3) for Gene-wise evaluation.
- `--mode auprc`: Runs only the AUPRC standalone evaluation.
- `--mode cell_eval`: Runs the standard cell-eval original pipeline.

### Score & Aggregation
Unlike standard evaluation scripts that scatter results across multiple CSVs, this pipeline maps and aggregates all metrics into a single, standardized output score sheet.

Once the run is complete, the pipeline collects metrics from all tools, applies the METRIC_MAPPING (e.g., `pearson_top20_de` → `PCC_DE_20`), and generates a final comprehensive CSV.


```bash
# The final aggregated results will be saved automatically to:
/NFS_DATA/samsung/database/benchmark_figure/pred_dataset_NZ/<MODEL_NAME>_total.csv
```

## Pipeline Architecture
The `result_summary.py` script acts as the master orchestrator. It processes predictions through a highly structured 4-step pipeline, ensuring data alignment, isolated metric calculation, and standardized aggregation.

### Step 1: Data Parsing & Alignment
- **Metadata Extraction**: Automatically parses essential metadata (Cell Line, PLM, Embedding strategy, and Cross-validation Fold) directly from the `_pred.h5ad` filename.
- **Gene Masking**: Loads both the prediction and the corresponding ground truth `.h5ad` files. It strictly filters the predicted matrix using the `exist` column mask from the truth data to ensure identical gene spaces before any metric is calculated.

### Step 2: Pre-computation (Cross-Fold Merging)
- **Gene-wise Caching**: Because Gene-wise metrics require evaluating variants across the entire dataset, the pipeline first groups the individual fold files (e.g., Fold 1-3, 2-3, 3-3) by their configuration `(Cell Line, PLM, Embedding)`. 
- It concatenates these folds into a single temporary matrix, runs the `condition.py` script to compute global PCC and MSE, and caches these results in memory for later injection.

### Step 3: Multi-Environment Orchestration
The pipeline uses Python's `subprocess` module to orchestrate disparate evaluation tools simultaneously, isolating temporary outputs to prevent overwriting.
- **Native Execution**: Runs GEARS (`run_gears.py`), Systema (`run_systema.py`), and Gene-wise calculations in the primary `cell-eval-dev` Conda environment.
- **Environment Bridging**: For the AUPRC metric (`main.py` $\rightarrow$ `analysis.R`), the script dynamically sources the bash profile and injects the execution command into an isolated R-compatible environment (`r-cmap-env`) to resolve dependency conflicts.

### Step 4: Metric Mapping & Aggregation
- **Data Harvesting**: After all subprocesses finish, the script crawls through the temporary output directories and reads the individual CSV files (`gears_perturbations.csv`, `systema_pearson.csv`, `overlap_stats.csv`, etc.).
- **Standardization** (`METRIC_MAPPING`): It maps local, tool-specific metric names (e.g., Systema's `prediction` or GEARS's `mse_top20_non_dropout`) to the globally standardized names (e.g., `ACC_centroid`, `MSE_ND_DE_20`).
- **Final Output**: The script appends the cached Gene-wise metrics, compiles a comprehensive Pandas DataFrame, and exports the final `<MODEL_NAME>_total.csv` matrix. Temporary directories are then automatically cleaned up.

## Metrics Reference
The pipeline aggregates and standardizes metrics from different evaluation tools into a single summary file.

### Terminology
- DE (Differentially Expressed): Refers to the top 20 significant DEGs identified using the Wilcoxon rank-sum test (adjusted $p$-value $< 0.05$), ranked by the absolute value of the $\log_2$ fold change ($|\text{LFC}|$).
- ND (Non-Dropout): Excludes dropout genes where the control expression is positive but the true perturbed expression is zero ($x_{\text{ctrl},g} > 0$ and $x^k_{\text{true},g} = 0$), ensuring metrics only evaluate biologically expressed genes.

### 1. GEARS Metrics (Cell-wise Evaluation)
CW (Cell-wise) metrics measure the prediction accuracy of the average transcriptional response for each perturbation condition.

- $\text{MSE}_{\text{cw}}^{\text{DE}}$ (from `mse_top20_non_dropout`): Cell-wise Mean Squared Error. Calculates the prediction error for each condition as the mean squared difference between the predicted ($x^k_{\text{pred},g}$) and true ($x^k_{\text{true},g}$) averaged expression over the non-dropout DEGs.
- $\text{PCC}_{\text{cw}}^{\text{DE}}$ (from `pearson_top20_non_dropout`): Cell-wise Pearson Correlation Coefficient. Measures the correlation between the predicted and ground truth averaged expression vectors over the non-dropout DEGs.
- $\Delta\text{PCC}_{\text{cw}}^{\text{DE}}$ (from `pearson_delta_top20_non_dropout`): Cell-wise Delta Pearson Correlation Coefficient. Measures the perturbation-induced transcriptional response by computing the correlation of expression *changes* relative to the control cell ($x_{\text{ctrl},g}$) across the non-dropout DEGs.

### 2. Systema Metrics
Metrics derived from the Systema evaluation framework, focusing on isolating variant-specific differences.
- $\Delta\text{SPCC}_{\text{cw}}^{\text{DE}}$ (from `corr_nondropout_top20de_allpert`): Systema Delta Pearson Correlation Coefficient. Replaces the control mean with the average expression profile across *all* perturbed cells ($x^{\text{true}}_{\text{avg}, g}$). This strict evaluation reduces shared variation across variants and purely focuses on variant-specific directional effects.
- `C-Acc.` (from `prediction`): Centroid Accuracy. Evaluates whether a predicted profile ($\mathbf{x}^k_{\text{pred}}$) is more similar to its corresponding ground-truth centroid ($\mathbf{x}^k_{\text{true}}$) than to any other centroid based on Euclidean distance. It measures the model’s ability to discriminate between variant-specific gene expression patterns.

### 3. Gene-wise Metrics
Calculated by merging cross-validation folds (1-3, 2-3, 3-3). GW metrics evaluate whether a model correctly identifies the transcriptional changes specific to each variant for individual genes.
- $\text{PCC}_{\text{gw}}$ (from `mean_pcc`): Gene-wise Pearson Correlation Coefficient. Evaluates variant-specific gene expression changes by measuring the correlation between the predicted ($\mathbf{v}^g_{\text{pred}}$) and true ($\mathbf{v}^g_{\text{true}}$) expression vectors of individual genes across all $K$ variant conditions. The final score is averaged across all valid genes.
- $\text{MSE}_{\text{gw}}$ (from `mean_mse`): Gene-wise Mean Squared Error. Calculates the mean squared difference between predicted and true expression levels for individual genes across all variant conditions, averaged over all valid genes.

### 4. AUPRC
- `AUPRC` (from `overlap_stats.csv`): Area Under the Precision-Recall Curve. Treats DE identification as a binary classification task to evaluate the model's ability to identify DEGs against background noise.
    - Ground Truth labeling: Uses the `limma` package to estimate the ground truth logFC and adjusted $p$-value. A gene is labeled as a true DEG ($y^k_g = 1$) if the adjusted $p$-value $< 0.05$ and the absolute logFC $> 0.58$.
    - Prediction Scoring: The prediction score ($s^k_g$) is defined as the magnitude of the predicted logFC ($|\log_2(x^k_{\text{pred},g} / x_{\text{ctrl},g})|$). The metric evaluates the precision-recall relationship as this classification threshold varies, averaged across all conditions.