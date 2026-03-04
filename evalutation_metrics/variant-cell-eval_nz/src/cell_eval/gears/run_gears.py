
import logging
import numpy as np
import pandas as pd
import scanpy as sc
import os
import argparse
import sys


from cell_eval.gears import utils as gears_utils 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def main(adata_pred_path: str, adata_truth_path: str, pert_col: str, control_pert: str, outdir: str = "./cell-eval-outdir"):

    logger.info("Loading AnnData objects...")
    adata_truth = sc.read_h5ad(adata_truth_path)
    adata_pred = sc.read_h5ad(adata_pred_path)
    logger.info("Data loading complete.")

    logger.info("\n--- adata_truth.uns 내용 확인 ---")
    logger.info(adata_truth.uns.keys())
    if 'rank_genes_groups_cov_all' in adata_truth.uns:
        logger.info("'rank_genes_groups_cov_all' 키가 adata_truth.uns에 존재합니다.")
    else:
        logger.error("'rank_genes_groups_cov_all' 키가 adata_truth.uns에 없습니다. 파일을 다시 확인하세요.")
        return

    logger.info("Preparing for analysis...")
    de_genes_dict = gears_utils.create_de_genes_dictionary(adata_truth)
    test_res = gears_utils.create_results_from_anndata(adata_pred, adata_truth, de_genes_dict)
    logger.info("Computing basic metrics...")
    basic_metrics, _ = gears_utils.compute_metrics(test_res)
    logger.info("Basic metrics computed successfully.")
    
    logger.info("\n[Overall Basic Metrics]")
    for k, v in basic_metrics.items():
        logger.info(f"  - {k}: {v:.6f}")
        value_to_print = v[0] if isinstance(v, tuple) else v
        print(f"{k}: {value_to_print:.6f}")
    ctrl_expression = gears_utils.get_control_expression(adata_truth)
    logger.info("Preparation complete.")

    logger.info("Starting deeper analysis...")
    deeper_res = gears_utils.deeper_analysis(adata_truth, test_res, ctrl_expression, control_pert)
    logger.info("Deeper analysis finished.")
    
    logger.info("Starting non-dropout analysis...")
    non_dropout_res = gears_utils.non_dropout_analysis(adata_truth, test_res, ctrl_expression, control_pert)
    logger.info("Non-dropout analysis finished.")

    logger.info("Starting subgroup analysis...")

    subgroups = adata_truth.uns.get("subgroups")

    if not subgroups:
        logger.info("No 'subgroups' found in adata.uns. Creating a default group with all perturbations.")
        all_perts = list(adata_truth.obs[pert_col].unique())
        if control_pert in all_perts:
            all_perts.remove(control_pert)
        
        subgroups = {
            'all_perts': all_perts
        } 
    metrics = ["pearson_delta", "pearson_delta_de"]

    metrics_non_dropout = [
        "pearson_delta_top20_de_non_dropout",
        "pearson_top20_de_non_dropout",
        "mse_top20_de_non_dropout",
        "mse_delta_top20_de_non_dropout",
    ]
    subgroup_analysis = {}

    for name, perts in subgroups.items():
        subgroup_analysis[name] = {m: [] for m in metrics + metrics_non_dropout}
        for pert in perts:
            if pert in deeper_res:
                for m in metrics:
                    subgroup_analysis[name][m].append(deeper_res[pert].get(m))
            if pert in non_dropout_res:
                for m in metrics_non_dropout:
                    value = non_dropout_res[pert].get(m)
                    if value is not None:
                        subgroup_analysis[name][m].append(value)

    results_data = []
    perturbation_results = []
    
    for name, result in subgroup_analysis.items():
        logger.info(f"--- Subgroup: {name} ---")
        for m, values in result.items():
            valid_values = [v for v in values if v is not None]
            if valid_values:
                mean_value = np.mean(valid_values)
                logger.info(f"test_{name}_{m}: {mean_value:.4f}")

                results_data.append({
                    'subgroup': name,
                    'metric': m,
                    'value': mean_value
                })
    
    for k, v in basic_metrics.items():
        value_to_save = v[0] if isinstance(v, tuple) else v
        results_data.append({
            'subgroup': 'overall',
            'metric': k,
            'value': value_to_save
        })
    for pert, metrics_dict in deeper_res.items():
        for metric, value in metrics_dict.items():
            if value is not None:
                perturbation_results.append({
                    'perturbation': pert,
                    'metric': metric,
                    'value': value,
                    'analysis_type': 'deeper'
                })
    
    for pert, metrics_dict in non_dropout_res.items():
        for metric, value in metrics_dict.items():
            if value is not None:
                perturbation_results.append({
                    'perturbation': pert,
                    'metric': metric,
                    'value': value,
                    'analysis_type': 'non_dropout'
                })
    
    logger.info("Subgroup analysis finished.")

    logger.info("Saving results to CSV...")
    os.makedirs(outdir, exist_ok=True)

    if results_data:
        results_df = pd.DataFrame(results_data)
        output_path = os.path.join(outdir, "gears.csv")
        results_df.to_csv(output_path, index=False)
        logger.info(f"Subgroup results saved to: {output_path}")

    if perturbation_results:
        pert_df = pd.DataFrame(perturbation_results)
        pert_output_path = os.path.join(outdir, "gears_perturbations.csv")
        pert_df.to_csv(pert_output_path, index=False)
        logger.info(f"Individual perturbation results saved to: {pert_output_path}")

        summary_data = []
        for name, result in subgroup_analysis.items():
            summary_row = {'subgroup': name}
            for m, values in result.items():
                valid_values = [v for v in values if v is not None]
                if valid_values:
                    summary_row[m] = np.mean(valid_values)
            summary_data.append(summary_row)

        deeper_summary = {}
        non_dropout_summary = {}

        if deeper_res:
            deeper_metrics = {}
            for pert, metrics_dict in deeper_res.items():
                for metric, value in metrics_dict.items():
                    if value is not None:
                        if metric not in deeper_metrics:
                            deeper_metrics[metric] = []
                        deeper_metrics[metric].append(value)
            
            for metric, values in deeper_metrics.items():
                deeper_summary[metric] = np.mean(values)

        if non_dropout_res:
            non_dropout_metrics = {}
            for pert, metrics_dict in non_dropout_res.items():
                for metric, value in metrics_dict.items():
                    if value is not None:
                        if metric not in non_dropout_metrics:
                            non_dropout_metrics[metric] = []
                        non_dropout_metrics[metric].append(value)
            
            for metric, values in non_dropout_metrics.items():
                non_dropout_summary[metric] = np.mean(values)

        if deeper_summary or non_dropout_summary:
            overall_summary = {'subgroup': 'overall'}
            overall_summary.update(deeper_summary)
            overall_summary.update(non_dropout_summary)
            summary_data.append(overall_summary)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_path = os.path.join(outdir, "gears_summary.csv")
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"Summary saved to: {summary_path}")
    else:
        logger.warning("No results to save.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run GEARS metric evaluation pipeline.")

    parser.add_argument('--pred', type=str, required=True, help='Path to Prediction AnnData (.h5ad)')
    parser.add_argument('--truth', type=str, required=True, help='Path to Truth AnnData (.h5ad)')
    
    parser.add_argument('--pert_col', type=str, default='condition', help='Column name for perturbation info (default: condition)')
    parser.add_argument('--control', type=str, default='ctrl', help='Name of control perturbation (default: ctrl)')
    parser.add_argument('--outdir', type=str, default='./gears-outdir', help='Output directory for results (default: ./cell-eval-outdir)')

    args = parser.parse_args()
    
    main(
        adata_pred_path=args.pred,
        adata_truth_path=args.truth,
        pert_col=args.pert_col,
        control_pert=args.control,
        outdir=args.outdir
    )