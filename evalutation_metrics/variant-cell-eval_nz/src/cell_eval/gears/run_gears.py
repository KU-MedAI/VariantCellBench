# /home/tech/variantseq/eugenie/variant_metric/cell-eval/src/cell_eval/gears/run_gears.py

'''

conda activate cell-eval-dev

python run_gears.py \
  --pred /NFS_DATA/samsung/database/benchmark_figure/ann_dataset/scLAMBDA/0108_0538_hct116_ankh_alt_1_3_pred.h5ad \
  --truth /NFS_DATA/samsung/database/benchmark_figure/ann_dataset/truth/hct116_1-3.h5ad \
  --pert_col "condition" \
  --control "ctrl" \
  --outdir ./gears_test_260130_1641
'''
import logging
import numpy as np
import pandas as pd
import scanpy as sc
import os
import argparse
import sys

# 같은 폴더의 utils.py에서 필요한 함수들을 모두 import 합니다.
from cell_eval.gears import utils as gears_utils    # 고치기

# 로거 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def main(adata_pred_path: str, adata_truth_path: str, pert_col: str, control_pert: str, outdir: str = "./cell-eval-outdir"):
    """
    두 개의 AnnData 파일 경로를 받아 전체 분석 파이프라인을 실행합니다.
    
    Args:
        adata_pred_path: 예측 데이터 AnnData 파일 경로
        adata_truth_path: 실제 데이터 AnnData 파일 경로
        pert_col: perturbation 컬럼명
        control_pert: control perturbation 이름
        outdir: 결과를 저장할 출력 디렉터리 경로
    """
    # 1. 데이터 로딩
    logger.info("Loading AnnData objects...")
    adata_truth = sc.read_h5ad(adata_truth_path)
    adata_pred = sc.read_h5ad(adata_pred_path)
    logger.info("Data loading complete.")

    # ⚠️ 디버깅 코드 추가: adata_truth의 .uns 내용 확인
    logger.info("\n--- adata_truth.uns 내용 확인 ---")
    logger.info(adata_truth.uns.keys())
    if 'rank_genes_groups_cov_all' in adata_truth.uns:
        logger.info("'rank_genes_groups_cov_all' 키가 adata_truth.uns에 존재합니다.")
    else:
        logger.error("'rank_genes_groups_cov_all' 키가 adata_truth.uns에 없습니다. 파일을 다시 확인하세요.")
        return # 키가 없으므로 함수를 종료합니다.

    # 2. 분석 준비
    logger.info("Preparing for analysis...")
    de_genes_dict = gears_utils.create_de_genes_dictionary(adata_truth)
    test_res = gears_utils.create_results_from_anndata(adata_pred, adata_truth, de_genes_dict)
    logger.info("Computing basic metrics...")
    basic_metrics, _ = gears_utils.compute_metrics(test_res)
    logger.info("Basic metrics computed successfully.")
    
    # 기본 메트릭 결과 출력
    logger.info("\n[Overall Basic Metrics]")
    for k, v in basic_metrics.items():
        logger.info(f"  - {k}: {v:.6f}")
        # pearsonr 함수의 결과가 튜플일 경우를 대비 (안정성)
        value_to_print = v[0] if isinstance(v, tuple) else v
        print(f"{k}: {value_to_print:.6f}")
    ctrl_expression = gears_utils.get_control_expression(adata_truth)
    logger.info("Preparation complete.")
    
    # 3. 심층 분석 실행
    logger.info("Starting deeper analysis...")
    deeper_res = gears_utils.deeper_analysis(adata_truth, test_res, ctrl_expression, control_pert)
    logger.info("Deeper analysis finished.")
    
    logger.info("Starting non-dropout analysis...")
    non_dropout_res = gears_utils.non_dropout_analysis(adata_truth, test_res, ctrl_expression, control_pert)
    logger.info("Non-dropout analysis finished.")

    # 4. 서브그룹 분석 및 결과 출력
    logger.info("Starting subgroup analysis...")
    # subgroup 정보는 adata_truth.uns에 저장되어 있다고 가정
    subgroups = adata_truth.uns.get("subgroups") # .get()은 키가 없으면 None을 반환

    # .uns에 'subgroups'가 없거나 비어있는 경우, 컨트롤을 제외한 모든 pert를 단일 그룹으로 생성
    if not subgroups:
        logger.info("No 'subgroups' found in adata.uns. Creating a default group with all perturbations.")
        all_perts = list(adata_truth.obs[pert_col].unique())
        if control_pert in all_perts:
            all_perts.remove(control_pert)
        
        subgroups = {
            'all_perts': all_perts
        } 
    metrics = ["pearson_delta", "pearson_delta_de"]
    # non_dropout_analysis에서 실제로 계산되는 모든 메트릭들
    metrics_non_dropout = [
        # Top20 DE non-dropout genes에 대한 결과
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

    # 결과를 DataFrame으로 저장하기 위한 리스트
    results_data = []
    perturbation_results = []
    
    for name, result in subgroup_analysis.items():
        logger.info(f"--- Subgroup: {name} ---")
        for m, values in result.items():
            valid_values = [v for v in values if v is not None]
            if valid_values:
                mean_value = np.mean(valid_values)
                logger.info(f"test_{name}_{m}: {mean_value:.4f}")
                # 결과를 리스트에 추가
                results_data.append({
                    'subgroup': name,
                    'metric': m,
                    'value': mean_value
                })
    
    # 기본 메트릭도 결과에 추가
    for k, v in basic_metrics.items():
        value_to_save = v[0] if isinstance(v, tuple) else v
        results_data.append({
            'subgroup': 'overall',
            'metric': k,
            'value': value_to_save
        })
    
    # 개별 perturbation별 결과도 저장
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
    
    # 5. 결과를 CSV 파일로 저장
    logger.info("Saving results to CSV...")
    os.makedirs(outdir, exist_ok=True)
    
    # 서브그룹별 평균 결과 저장
    if results_data:
        results_df = pd.DataFrame(results_data)
        output_path = os.path.join(outdir, "gears.csv")
        results_df.to_csv(output_path, index=False)
        logger.info(f"Subgroup results saved to: {output_path}")
    
    # 개별 perturbation별 결과 저장
    if perturbation_results:
        pert_df = pd.DataFrame(perturbation_results)
        pert_output_path = os.path.join(outdir, "gears_perturbations.csv")
        pert_df.to_csv(pert_output_path, index=False)
        logger.info(f"Individual perturbation results saved to: {pert_output_path}")
        
        # 요약 정보도 저장 (pivot 형태)
        summary_data = []
        for name, result in subgroup_analysis.items():
            summary_row = {'subgroup': name}
            for m, values in result.items():
                valid_values = [v for v in values if v is not None]
                if valid_values:
                    summary_row[m] = np.mean(valid_values)
            summary_data.append(summary_row)
        
        # deeper_res와 non_dropout_res에서도 평균값 계산
        deeper_summary = {}
        non_dropout_summary = {}
        
        # deeper_res 평균값 계산
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
        
        # non_dropout_res 평균값 계산
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
            
        
        # 전체 평균 요약 추가
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
    
    # 필수 인자
    parser.add_argument('--pred', type=str, required=True, help='Path to Prediction AnnData (.h5ad)')
    parser.add_argument('--truth', type=str, required=True, help='Path to Truth AnnData (.h5ad)')
    
    # 선택 인자 (기본값 설정됨)
    parser.add_argument('--pert_col', type=str, default='condition', help='Column name for perturbation info (default: condition)')
    parser.add_argument('--control', type=str, default='ctrl', help='Name of control perturbation (default: ctrl)')
    parser.add_argument('--outdir', type=str, default='./gears-outdir', help='Output directory for results (default: ./cell-eval-outdir)')

    args = parser.parse_args()

    # Main 함수 실행
    main(
        adata_pred_path=args.pred,
        adata_truth_path=args.truth,
        pert_col=args.pert_col,
        control_pert=args.control,
        outdir=args.outdir
    )