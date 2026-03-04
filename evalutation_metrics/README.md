# Variant-Cell-Eval Pipeline

## Description
This pipeline provides a comprehensive suite of metrics for evaluating the performance of models (e.g., CellFM, GeneCompass) that predict cellular responses to genetic variants at the single-cell level. 

Rather than running individual evaluation scripts manually, this pipeline orchestrates multiple evaluation frameworks—**GEARS, Systema, Gene-wise (PCC/RMSE), and AUPRC**—and aggregates their outputs into a single, standardized summary matrix.

## Installation
This pipeline relies on specific Conda environments to manage conflicting dependencies across different evaluation tools.

```bash
# Main environment for the pipeline (GEARS, Systema, Gene-wise, cell-eval)
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

Once the run is complete, the pipeline collects metrics from all tools, applies the METRIC_MAPPING (e.g., `pearson_top20_d`e → `PCC_DE_20`), and generates a final comprehensive CSV.


```bash
# The final aggregated results will be saved automatically to:
/NFS_DATA/samsung/database/benchmark_figure/pred_dataset_NZ/<MODEL_NAME>_total.csv
```

## Pipeline Architecture

## Metrics Reference
The pipeline aggregates and standardizes metrics from different evaluation tools into a single summary file.

### Terminology
- DE (Differentially Expressed): Refers to the top 20 significant DEGs identified using the Wilcoxon rank-sum test (adjusted $p$-value $< 0.05$), ranked by the absolute value of the $\log_2$ fold change ($|\text{LFC}|$).
- ND (Non-Dropout): Excludes dropout genes where the control expression is positive but the true perturbed expression is zero ($x_{\text{ctrl},g} > 0$ and $x^k_{\text{true},g} = 0$), ensuring metrics only evaluate biologically expressed genes.

### 1. GEARS Metrics (Cell-wise Evaluation)
CW (Cell-wise) metrics measure the prediction accuracy of the average transcriptional response for each perturbation condition.

- `MSE_ND_DE_20` (from `mse_top20_non_dropout`): Mean Squared Error (MSE) of the raw expression values for the Top 20 Non-Dropout DE genes.
- `PCC_ND_DE_20` (from `pearson_top20_non_dropout`): Pearson Correlation Coefficient (PCC) of the raw expression values for the Top 20 Non-Dropout DE genes.
- `delta_PCC_ND_DE_20` (from `pearson_delta_top20_non_dropout`): PCC of the delta expression for the Top 20 Non-Dropout DE genes.

### 2. Systema Metrics
Metrics derived from the Systema evaluation framework.
- `sys_delta_PCC_ND_DE_20` (from `corr_nondropout_top20de_allpert`): Systema's calculation of Pearson Correlation on the delta expression for the Top 20 Non-Dropout DE genes across all perturbations.
- `ACC_centroid` (from `prediction`): Euclidean distance-based prediction accuracy (centroid classification). Evaluates if the predicted perturbation state is closest to its true perturbation state compared to others.

### 3. Gene-wise Metrics
Calculated by merging cross-validation folds (1-3, 2-3, 3-3) to evaluate per-gene predictive performance across conditions.
- `GW_PCC` (from `mean_pcc`): The overall Gene-wise Pearson Correlation Coefficient, averaged across **all** valid genes. It measures how well the model predicts the up- and down-regulation trends of genes across various genetic perturbations.
- `GW_MSE` (from `mean_mse`): The overall Gene-wise Mean Squared Error, averaged across **all** valid genes. It measures the average squared deviation between the true and predicted pseudobulk expression levels of genes across different variants.

### 4. AUPRC
- `AUPRC` (from `overlap_stats.csv`): Area Under the Precision-Recall Curve. Evaluates the model's ability to correctly classify DE.
    - Ground Truth Definition: Uses the `limma` package to identify true Differentially Expressed Genes (DEGs) based on strict statistical thresholds (adjusted p-value < 0.05 and |logFC| > 0.58).
    - Prediction Scoring: Calculates the predicted |logFC| between the true control and the predicted variant state. This predicted magnitude serves as the confidence score for the PR curve.
    - Output Aggregation: Computes the mean AUPRC across all valid variants, providing a global classification performance metric.