
import argparse
import os
import numpy as np
import pandas as pd
import anndata
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from utils2 import get_pseudobulk_matrix, compute_inter_variant_metrics
from scipy import sparse
from anndata import AnnData

def main():
    parser = argparse.ArgumentParser(description="Calculate Pseudo-bulk Gene Metrics (Inter-variant).")
    
    parser.add_argument('--truth', type=str, required=True, help='Path to ground truth .h5ad')
    parser.add_argument('--pred', type=str, required=True, help='Path to prediction .h5ad')
    parser.add_argument('--output', type=str, default='pseudobulk_metrics_results', help='Output folder name')
    parser.add_argument('--condition_col', type=str, default='condition', help='Column in obs to split by')

    args = parser.parse_args()
    
    print("Loading data...")
    truth_adata = anndata.read_h5ad(args.truth)
    pred_adata = anndata.read_h5ad(args.pred)
    os.makedirs(args.output, exist_ok=True)

    # 1. 공통 유전자 필터링
    common_genes = truth_adata.var_names.intersection(pred_adata.var_names)
    print(f"[Info] Using {len(common_genes)} common genes.")
    
    truth_adata = truth_adata[:, common_genes].copy()
    pred_adata = pred_adata[:, common_genes].copy()

    # 2. Pseudo-bulk (평균) 변환
    print("\n>>> Step 1: Aggregating Pseudo-bulk Means...")
    
    df_truth = get_pseudobulk_matrix(truth_adata, args.condition_col)
    df_pred = get_pseudobulk_matrix(pred_adata, args.condition_col)
    
    print(f"    - Truth Matrix Shape: {df_truth.shape}")
    print(f"    - Pred Matrix Shape: {df_pred.shape}")

    # 3. Inter-variant PCC/MSE 계산
    print("\n>>> Step 2: Calculating Gene-wise PCC & MSE across variants...")
    result_df = compute_inter_variant_metrics(df_truth, df_pred)

    if result_df.empty:
        print("[Error] No valid results generated.")
        return

    # 4. 결과 저장
    cols_order = ['gene_name', 'pcc', 'mse', 'p_value', 'mean_expr_truth', 'mean_expr_pred']
    result_df = result_df[cols_order]
    
    # (2) PCC 높은 순으로 정렬
    result_df = result_df.sort_values(by='pcc', ascending=False)
    
    # (3) 파일 저장
    save_filename = "pseudobulk_nonzero_gene_metrics.csv"
    save_path = os.path.join(args.output, save_filename)
    result_df.to_csv(save_path, index=False)
    
    # 요약 통계 저장
    summary = {
        'total_genes': len(result_df),
        'valid_pcc_genes': result_df['pcc'].notna().sum(),
        'mean_pcc': result_df['pcc'].mean(),
        'median_pcc': result_df['pcc'].median(),
        'mean_mse': result_df['mse'].mean()
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(args.output, "summary_stats.csv"), index=False)

    print(f"\n[Done] All gene metrics saved to: {save_path}")
    print(f"    - Mean PCC: {summary['mean_pcc']:.4f}")
    print(f"    - Mean MSE: {summary['mean_mse']:.4f}")

if __name__ == "__main__":
    main()