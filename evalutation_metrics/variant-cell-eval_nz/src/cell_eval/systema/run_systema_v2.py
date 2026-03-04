'''
251206 수정본
[수정사항]
person correlation 계산 시, 평균 계산 방식을 mean_nonzero로 변경
- pert별 평균발현량 계산 시, mean_nonzero 함수를 사용하도록 변경
- reference 계산 시, mean_nonzero 함수를 사용하도록 변경
python run_systema_v2.py \
  --adata_pred_path /NFS_DATA/samsung/database/benchmark_figure/ann_dataset/scLAMBDA/0108_0538_hct116_ankh_alt_1_3_pred.h5ad \
  --adata_truth_path /NFS_DATA/samsung/database/benchmark_figure/ann_dataset/truth/hct116_1-3.h5ad \
  --pert_col 'condition' \
  --control_pert 'ctrl' \
  --outdir './systema_results_260130'
'''


import argparse
import anndata
import pandas as pd
import numpy as np
import os
import sys

# 패키지 구조에 따라 import 경로를 유연하게 처리하기 위한 설정
try:
    # 패키지로 설치되었거나 구조가 잡혀있는 경우
    from . import utils_v2 as sys_utils
except ImportError:
    try:
        # 같은 폴더에 utils.py가 있는 경우
        import utils_v2 as sys_utils
    except ImportError:
        print("ERROR: Could not import 'utils'. Please ensure utils.py is in the same directory or properly installed.")
        sys.exit(1)

def main(
    adata_pred_path: str,
    adata_truth_path: str,
    pert_col: str,
    control_pert: str,
    outdir: str = "./systema-outdir"
):
    # --- 1. 데이터 로드 및 전처리 ---
    print(f"INFO: Loading AnnData files...")
    print(f"   - Pred: {adata_pred_path}")
    print(f"   - Truth: {adata_truth_path}")
    
    try:
        pred_adata = anndata.read_h5ad(adata_pred_path)
        truth_adata = anndata.read_h5ad(adata_truth_path)
    except Exception as e:
        print(f"ERROR: Failed to load AnnData files. {e}")
        return

    print("INFO: Converting AnnData to pandas DataFrame...")
    truth_df = pd.DataFrame(
        truth_adata.X.toarray() if hasattr(truth_adata.X, "toarray") else truth_adata.X,
        index=truth_adata.obs[pert_col],
        columns=truth_adata.var_names
    )
    # 예측 데이터에 'method' 열이 없으면 'prediction'으로 기본값 설정
    if 'method' not in pred_adata.obs.columns:
        pred_adata.obs['method'] = 'prediction'
    
    pred_df = pd.DataFrame(
        pred_adata.X.toarray() if hasattr(pred_adata.X, "toarray") else pred_adata.X,
        columns=pred_adata.var_names,
        index=pd.MultiIndex.from_frame(pred_adata.obs[[pert_col, 'method']], names=['condition', 'method'])
    )

    # 각 condition 내 여러 샘플들의 평균을 계산하여 대표 프로필(centroid) 생성
    print("INFO: Calculating centroids for each condition by taking the mean...")
    truth_centroids = truth_df.groupby(truth_df.index).mean()
    pred_centroids = pred_df.groupby(level=['condition', 'method']).mean()

    # --- 2. 메트릭 1: 유클리드 거리 기반 순위 정확도(centroid accuracy) ---
    print("\nINFO: Calculating Metric 1: Euclidean Distance Rank Accuracy...")
    euclidean_scores_df = sys_utils.calculate_centroid_accuracies(pred_centroids, truth_centroids)

    # --- 3. 메트릭 2: 피어슨 상관계수 (Control 대비 변화량) ---
    print("INFO: Calculating the average perturbation profile as the reference (using mean_nonzero)...")
    
    reference = sys_utils.calculate_average_perturbation(
        truth_adata, 
        pert_col=pert_col, 
        control_pert=control_pert
    )
    print(f"DEBUG: Reference shape: {reference.shape}, has NaN: {np.isnan(reference).any()}")


    # --- 2. 메트릭: 평균 교란 대비 피어슨 상관계수 --- 
    print("INFO: Calculating Pearson Correlation on Average-Perturbation-Normalized Data...")

    # 결과를 저장할 딕셔너리
    results = {}

    # 모든 perturbation 목록 생성 (Control 제외)
    perturbations = truth_adata.obs[pert_col].unique().tolist()
    if control_pert in perturbations:
        perturbations.remove(control_pert)

    # 각 퍼터베이션에 대해 루프 실행
    for pert in perturbations:
        print(f"Processing perturbation: {pert}")

        # [수정됨 2] 1. X_true (실제값) 계산 - mean_nonzero 사용
        adata_pert = truth_adata[truth_adata.obs[pert_col] == pert]
        # 기존: X_true = np.array(adata_pert.X.mean(axis=0)).flatten()
        X_true = sys_utils.mean_nonzero(adata_pert.X, axis=0)
        
        # [수정됨 3] 2. X_pred (예측값) 계산 - mean_nonzero 사용
        adata_pert_pred = pred_adata[pred_adata.obs[pert_col] == pert]
        # 기존: X_pred = np.array(adata_pert_pred.X.mean(axis=0)).flatten()
        X_pred = sys_utils.mean_nonzero(adata_pert_pred.X, axis=0)

        print(f"DEBUG: X_true shape: {X_true.shape}, has NaN: {np.isnan(X_true).any()}")


        # condition_name을 사용해서 uns에서 정보 가져오기
        # pert에 해당하는 condition_name 찾기
        pert_condition_names = truth_adata[truth_adata.obs[pert_col] == pert].obs['condition_name'].unique()
        if len(pert_condition_names) == 0:
            print(f"경고: '{pert}'에 해당하는 condition_name을 찾을 수 없습니다. 이 퍼터베이션을 건너뜁니다.")
            continue
        
        full_key_name = pert_condition_names[0]
        print(f"DEBUG: Using condition_name: {full_key_name}")
        
        non_dropout_idxs = np.array([]) 
        
        # 'top_non_dropout' 키가 있는지 확인
        if 'top_non_dropout' in truth_adata.uns:
            available_keys = list(truth_adata.uns['top_non_dropout'].keys())
            
            if full_key_name not in available_keys:
                print(f"WARNING: {full_key_name} not found in 'top_non_dropout' keys")
                
                # 폴백 로직: 비슷한 키 검색
                pert_core = pert.replace('+ctrl', '').replace('~', '~') 
                print(f"DEBUG: Looking for key containing: {pert_core}")
                
                found_key = None
                for key in available_keys:
                    if pert_core in key:
                        found_key = key
                        print(f"DEBUG: Using similar key: {found_key}")
                        break
                
                if not found_key:
                    print(f"ERROR: No matching key found for {pert_core} in available keys")
                else:
                    key_to_use = found_key
            else:
                key_to_use = full_key_name

            if 'key_to_use' in locals():
                non_dropout_gene_names_data = truth_adata.uns['top_non_dropout'].get(key_to_use)

                if non_dropout_gene_names_data is not None:
                    if isinstance(non_dropout_gene_names_data, np.ndarray):
                        non_dropout_gene_names = non_dropout_gene_names_data.tolist()
                    else:
                        non_dropout_gene_names = list(non_dropout_gene_names_data)
                    
                    # 유전자 이름을 현재 anndata의 인덱스로 변환
                    all_indices = truth_adata.var_names.get_indexer(non_dropout_gene_names)
                    non_dropout_idxs = all_indices[all_indices != -1]
                    
                    print(f"DEBUG: Matched {len(non_dropout_idxs)} non-dropout genes to current adata.var_names.")
                else:
                    print(f"DEBUG: No non-dropout data found for key '{key_to_use}' in 'top_non_dropout'.")
        else:
            print(f"WARNING: 'top_non_dropout' key not found in truth_adata.uns")

        # --- 2. 전체 유전자용 top20_de_names 가져오기 ---
        if 'rank_genes_groups_cov_all' in truth_adata.uns:
            all_genes_top20_names = truth_adata.uns['rank_genes_groups_cov_all'].get(full_key_name, [])
            if not isinstance(all_genes_top20_names, list):
                all_genes_top20_names = list(all_genes_top20_names)
                
            if not all_genes_top20_names:
                print(f"경고: 'rank_genes_groups_cov_all'에서 {full_key_name}을 찾을 수 없습니다.")
            all_genes_top20_names = all_genes_top20_names[:20]
        else:
            all_genes_top20_names = []
            print(f"경고: 'rank_genes_groups_cov_all'을 찾을 수 없습니다.")

        # --- 3. non_dropout 유전자용 top20_de_names 가져오기 ---
        if 'top_non_dropout_de_20' in truth_adata.uns:
            non_dropout_top20_names = truth_adata.uns['top_non_dropout_de_20'].get(full_key_name, [])
            if not isinstance(non_dropout_top20_names, list):
                non_dropout_top20_names = list(non_dropout_top20_names)

            if not non_dropout_top20_names:
                print(f"경고: 'top_non_dropout_de_20'에서 {full_key_name}을 찾을 수 없습니다.")
        else:
            non_dropout_top20_names = []
            print(f"경고: 'top_non_dropout_de_20'을 찾을 수 없습니다.")

        # --- 4. top20_de_idxs 준비 ---
        all_genes_top20_idxs_raw = truth_adata.var_names.get_indexer(all_genes_top20_names)
        all_genes_top20_idxs = all_genes_top20_idxs_raw[all_genes_top20_idxs_raw != -1]

        non_dropout_top20_idxs_raw = truth_adata.var_names.get_indexer(non_dropout_top20_names)
        non_dropout_top20_idxs = non_dropout_top20_idxs_raw[non_dropout_top20_idxs_raw != -1]

        # --- 5. 메트릭 함수 호출 (이미 mean_nonzero가 적용된 값들 전달) ---
        # 1. 전체 메트릭
        metrics_all = sys_utils.pearson_delta_reference_metrics(
            X_true=X_true,
            X_pred=X_pred,
            reference=reference, 
            top20_de_idxs=all_genes_top20_idxs,
            non_dropout_idxs=non_dropout_idxs
        )

        # 2. Top20 DE 메트릭
        metrics_top20_de = sys_utils.pearson_delta_reference_metrics_top20_de(
            X_true=X_true,
            X_pred=X_pred,
            reference=reference,
            top20_de_idxs=all_genes_top20_idxs
        )

        # metrics_mse_all = sys_utils.mse_delta_reference_metrics(
        #     X_true, X_pred, reference, all_genes_top20_idxs, non_dropout_idxs
        # )

        # 3. Non-dropout 메트릭
        metrics_non_dropout = {}
        if len(non_dropout_idxs) > 0:
            metrics_non_dropout = sys_utils.pearson_delta_reference_metrics_non_dropout(
                X_true=X_true,
                X_pred=X_pred,
                reference=reference,
                non_dropout_idxs=non_dropout_idxs
            )
            
            metrics_non_dropout_top20de = sys_utils.pearson_delta_reference_metrics_non_dropout_top20de(
                X_true=X_true,
                X_pred=X_pred,
                reference=reference,
                non_dropout_idxs=non_dropout_idxs,
                top20_de_idxs=non_dropout_top20_idxs
            )
            # m_mse_nd_top20 = sys_utils.mse_delta_reference_metrics_non_dropout_top20de(
            #     X_true, X_pred, reference, non_dropout_idxs, non_dropout_top20_idxs
            # )
            metrics_non_dropout.update(metrics_non_dropout_top20de)
            #metrics_non_dropout.update(m_mse_nd_top20)
        else:
            print(f"DEBUG: No *matched* non-dropout genes found for {pert}")

        combined_metrics = {}
        combined_metrics.update(metrics_all)
        combined_metrics.update(metrics_top20_de)
        combined_metrics.update(metrics_non_dropout)
        #combined_metrics.update(metrics_mse_all)

        results[pert] = combined_metrics

    # 최종 결과 확인
    for pert, scores in results.items():
        print(f"\n--- {pert} ---")
        print(scores)

    # DataFrame 변환 및 결과 출력
    results_df = pd.DataFrame.from_dict(results, orient='index')

    nd_results = {}
    for pert, metrics in results.items():
        nd_metrics = {k: v for k, v in metrics.items() if 'nondropout' in k}
        if nd_metrics:
            nd_results[pert] = nd_metrics
    
    nd_results_df = pd.DataFrame.from_dict(nd_results, orient='index')
    average_metrics = results_df.mean()
    nd_average_metrics = nd_results_df.mean() if not nd_results_df.empty else pd.Series()

    # --- 최종 보고서 출력 ---
    print("\n\n" + "="*60)
    print("            Final Evaluation Results (Systema - Mean Nonzero)")
    print("="*60)

    print("\n--- [Metric 1] Euclidean Distance Rank Accuracy ---")
    print(euclidean_scores_df)
    scores_without_ctrl = euclidean_scores_df.drop(control_pert, errors='ignore')
    print(f"Mean Score (excluding {control_pert}): {scores_without_ctrl['prediction'].mean()}")

    print("\n" + "-"*60)
    print("\n--- [Metric 2] Pearson Correlation (vs. Avg. Perturbation) ---")
    print(results_df)
    print("\n--- Mean Score ---")
    print(average_metrics)
    
    if not nd_results_df.empty:
        print("\n--- [Non-dropout Genes Results] ---")
        print(nd_results_df)
        print("\n--- [Non-dropout Mean Score] ---")
        print(nd_average_metrics)

    print("\n\n" + "="*60)
    
    # CSV 저장
    print("\nINFO: Saving results to CSV files...")
    os.makedirs(outdir, exist_ok=True)
    
    euclidean_output_path = os.path.join(outdir, "systema_euclidean.csv")
    euclidean_scores_df.to_csv(euclidean_output_path)
    print(f"Euclidean distance results saved to: {euclidean_output_path}")
    
    pearson_output_path = os.path.join(outdir, "systema_pearson.csv")
    results_df.to_csv(pearson_output_path)
    print(f"Pearson correlation results saved to: {pearson_output_path}")
    
    if not nd_results_df.empty:
        nd_output_path = os.path.join(outdir, "systema_ND.csv")
        nd_results_df.to_csv(nd_output_path)
        print(f"Non-dropout genes results saved to: {nd_output_path}")
    
    # Summary 저장
    summary_data = []
    
    euclidean_mean = scores_without_ctrl['prediction'].mean()
    summary_data.append({
        'metric_type': 'euclidean_distance',
        'metric_name': 'mean_rank_accuracy',
        'value': euclidean_mean,
        'description': 'Euclidean Distance Rank Accuracy'
    })
    
    for metric_name, value in average_metrics.items():
        summary_data.append({
            'metric_type': 'pearson_correlation',
            'metric_name': metric_name,
            'value': value,
            'description': f'Pearson Correlation - {metric_name}'
        })
    
    if not nd_average_metrics.empty:
        for metric_name, value in nd_average_metrics.items():
            summary_data.append({
                'metric_type': 'pearson_correlation_nd',
                'metric_name': metric_name,
                'value': value,
                'description': f'Pearson Correlation (Non-dropout) - {metric_name}'
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_output_path = os.path.join(outdir, "systema_summary.csv")
    summary_df.to_csv(summary_output_path, index=False)
    print(f"Summary results saved to: {summary_output_path}")
    
    print("INFO: All systema results saved successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Systema metric evaluation pipeline.")
    
    # 필수 인자
    parser.add_argument('--adata_pred_path', type=str, required=True, help='Path to Prediction AnnData (.h5ad)')
    parser.add_argument('--adata_truth_path', type=str, required=True, help='Path to Truth AnnData (.h5ad)')
    parser.add_argument('--pert_col', type=str, required=True, help='Column name for perturbation info')
    parser.add_argument('--control_pert', type=str, required=True, help='Name of control perturbation')
    
    # 선택 인자
    parser.add_argument('--outdir', type=str, default='./cell-eval-outdir', help='Output directory for results')

    args = parser.parse_args()

    # Main 함수 실행
    main(
        adata_pred_path=args.adata_pred_path,
        adata_truth_path=args.adata_truth_path,
        pert_col=args.pert_col,
        control_pert=args.control_pert,
        outdir=args.outdir
    )