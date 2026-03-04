import pandas as pd
import os
import re
import sys
import argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import subprocess
import shutil
import anndata as ad
import glob

'''
variant result 집계 방식 수정 for result plot
fold3-3, cell-eval-dev, r-cmap-env 버전 (원래 cellfm의 result_summary와 다름, from genecompass)
benchmark용 gears, systema, gene wise, auprc 통합용
gw 수정, metric 집계 acc-centroid 추가
v6: NZ 적용 안한 버전
'''

'''
conda activate cell-eval-dev

/home/tech/anaconda3/envs/cellflow/bin/python /home/tech/variantseq/suji/result_summary6.py \
--mode gears_systema \
--model CellFM \
--date 260209


'''

TRUTH_DIR = "/NFS_DATA/samsung/database/benchmark_figure/ann_dataset/truth/"

METRIC_MAPPING = {
    'GEARS': {
        'mse_top20_non_dropout': 'MSE_ND_DE_20',
        'mse_delta_top20_non_dropout': 'delta_MSE_ND_DE_20',
        'pearson_top20_non_dropout': 'PCC_ND_DE_20',
        'pearson_delta_top20_non_dropout': 'delta_PCC_ND_DE_20',
        'mse_top20_de': 'MSE_DE_20', # gears_summary.csv
        'pearson_top20_de': 'PCC_DE_20', # gears_summary.csv
        'pearson_delta_de': 'delta_PCC_DE_20' # gears_summary.csv
    },
    'systema': {
        'corr_nondropout_top20de_allpert': 'sys_delta_PCC_ND_DE_20',
        'corr_20de_allpert': 'sys_delta_PCC_DE_20', # systema_summary.csv
        'prediction' : 'ACC_centroid'
    },
    'gene_wise': {
        'mean_pcc': 'GW_PCC_ND_DE_20',
        'mean_mse': 'GW_MSE_ND_DE_20'
    },
    'AUPRC': {
        'AUPRC': 'AUPRC'
    },
}

# =============================================================================
# 2. Parsing Helper Functions
# =============================================================================

def clean_condition_column(adata):
    """
    adata.obs['condition']에서 '+ctrl' 접미사를 제거합니다.
    예: 'TP53~A276V+ctrl' -> 'TP53~A276V'
    'ctrl'은 그대로 유지됩니다.
    """
    if 'condition' in adata.obs.columns:
        # Categorical 타입일 수 있으므로 string으로 변환 후 처리
        conditions = adata.obs['condition'].astype(str)
        
        # 정규표현식으로 끝에 있는 '+ctrl'만 제거
        # GeneCompass 데이터(이미 깔끔한 경우)는 변화 없음
        conditions = conditions.str.replace(r'\+ctrl$', '', regex=True)
        
        adata.obs['condition'] = conditions
        # print("  -> Condition column cleaned (+ctrl removed).")

def parse_systema_summary_value(value_str):
    match = re.search(r'[\d\.]+', str(value_str))
    return float(match.group(0)) if match else None

def get_mapped_metric(category, original_metric):
    if category in METRIC_MAPPING and original_metric in METRIC_MAPPING[category]:
        return METRIC_MAPPING[category][original_metric]
    return None

def parse_run_name(run_name):
    """실행 이름에서 메타데이터를 추출합니다."""
    plm_map = {
        'protT5': 'ProtT5', 'esm2': 'ESM2', 'msa': 'MSA-Transformer',
        'pglm': 'xTrimoPGLM', 'ankh': 'Ankh'
    }
    embedding_map = {
        'alt': 'ALT', 'diff': 'REF-ALT'
    }

    match = re.search(r'\d{4}_\d{4}_(.*)', run_name)
    if not match:
        return {'cell_line': None, 'plm': None, 'embedding': None, 'fold': None}
    
    parts = match.group(1).split('_')
    
    raw_cell_line = parts[0] if len(parts) > 0 else None
    raw_plm = parts[1] if len(parts) > 1 else None
    raw_embedding = parts[2] if len(parts) > 2 else None
    # raw_fold = parts[3] if len(parts) > 3 else None
    if len(parts) > 3:
        raw_fold = '-'.join(parts[3:])
    else:
        raw_fold = None

    cell_line = raw_cell_line.upper() if raw_cell_line else None
    plm = plm_map.get(raw_plm, raw_plm)
    embedding = embedding_map.get(raw_embedding, raw_embedding)
    fold = raw_fold

    return {
        'cell_line': cell_line,
        'plm': plm,
        'embedding': embedding,
        'fold': fold,
        'raw_parts':parts
    }

# =============================================================================
# 3. Result Processing Logic
# =============================================================================

def process_run_results(results_dir, run_name, gene_wise_cache):
    """
    GEARS, Systema, Gene-wise, AUPRC 결과를 모두 읽어 통합합니다.
    """
    if not os.path.isdir(results_dir):
        print(f"[오류] 대상 폴더를 찾을 수 없습니다: {results_dir}")
        return []

    # 1. 메타데이터 추출
    metadata = parse_run_name(run_name)
    # metadata['run_dir'] = os.path.join(OUTPUT_DIR, run_name)

    # 2. 결과를 variant 기준으로 수집할 딕셔너리
    variant_metrics = defaultdict(dict)
    mean_metrics = {}

    # -------------------------------------------------------------------------
    # [1] GEARS Parsing
    # -------------------------------------------------------------------------
    gears_pert_csv_path = os.path.join(results_dir, 'gears_perturbations.csv')
    if os.path.exists(gears_pert_csv_path):
        try:
            df = pd.read_csv(gears_pert_csv_path)
            for _, row in df.iterrows():
                variant = row['perturbation']
                if 'ctrl' in str(variant): continue
                mapped_metric = get_mapped_metric('GEARS', row['metric'])
                if mapped_metric:
                    variant_metrics[variant][mapped_metric] = row['value']
        except Exception as e:
            print(f"Warning: Error parsing GEARS pert csv: {e}")

    # GEARS Mean
    gears_summary_path = os.path.join(results_dir, 'gears_summary.csv')
    if os.path.exists(gears_summary_path):
        try:
            df_gears = pd.read_csv(gears_summary_path)
            overall_row = df_gears[df_gears['subgroup'] == 'overall']
            if not overall_row.empty:
                for metric, value in overall_row.iloc[0].items():
                    mapped = get_mapped_metric('GEARS', metric)
                    if mapped: mean_metrics[mapped] = value
        except Exception as e:
            print(f"Warning: Error parsing GEARS summary csv: {e}")

    # -------------------------------------------------------------------------
    # [2] Systema Parsing
    # -------------------------------------------------------------------------
    sys_p_path = os.path.join(results_dir, 'systema_pearson.csv')
    if os.path.exists(sys_p_path):
        try:
            df = pd.read_csv(sys_p_path, index_col=0)
            for variant, row in df.iterrows():
                if 'ctrl' in str(variant): continue
                for metric, value in row.items():
                    mapped_metric = get_mapped_metric('systema', metric)
                    if mapped_metric:
                        variant_metrics[variant][mapped_metric] = value
        except Exception as e:
            print(f"Warning: Error parsing Systema pearson csv: {e}")

    sys_e_path = os.path.join(results_dir, 'systema_euclidean.csv')
    if os.path.exists(sys_e_path):
        try:
            df = pd.read_csv(sys_e_path)
            acc_values = []
            mapped_metric = get_mapped_metric('systema', 'prediction') 
            if mapped_metric:
                for _, row in df.iterrows():
                    variant = row['condition']
                    if 'ctrl' in str(variant): continue
                    variant_metrics[variant][mapped_metric] = row['prediction']
                    acc_values.append(row['prediction'])
                if acc_values:
                    mean_metrics[mapped_metric] = np.mean(acc_values)
        except Exception as e:
            print(f"Warning: Error parsing Systema euclidean csv: {e}")

    # Systema Mean
    sys_summary_path = os.path.join(results_dir, 'systema_summary.csv')
    if os.path.exists(sys_summary_path):
        try:
            df_sys_s = pd.read_csv(sys_summary_path)   
            for _, row in df_sys_s.iterrows():
                metric_orig = 'accuracy_sys' if row['metric_name'] == 'mean_rank_accuracy' else row['metric_name']
                mapped = get_mapped_metric('systema', metric_orig)
                if mapped:
                    value = parse_systema_summary_value(row['value']) if metric_orig == 'accuracy_sys' else row['value']
                    mean_metrics[mapped] = value
        except Exception as e:
            print(f"Warning: Error parsing Systema summary csv: {e}")

    if gene_wise_cache:
        # 현재 run의 식별키 생성
        config_key = (metadata['cell_line'], metadata['plm'], metadata['embedding'])
        
        if config_key in gene_wise_cache:
            gw_results = gene_wise_cache[config_key]
            
            # 1. Variant 별 매칭
            for variant in variant_metrics.keys():
                if variant in gw_results:
                    variant_metrics[variant].update(gw_results[variant])
            
            # 2. Mean 값 매칭 ('TOTAL_AVERAGE' -> 'mean')
            if 'TOTAL_AVERAGE' in gw_results:
                mean_metrics.update(gw_results['TOTAL_AVERAGE'])

    # -------------------------------------------------------------------------
    # [4] AUPRC Parsing
    # -------------------------------------------------------------------------
    auprc_path = os.path.join(results_dir, 'overlap_stats.csv')
    if os.path.exists(auprc_path):
        try:
            df_auprc = pd.read_csv(auprc_path)
            # AUPRC csv format: columns=[Condition, ..., AUPRC], last row Condition='mean'
            for _, row in df_auprc.iterrows():
                cond = row['Condition']
                
                # Metric은 AUPRC 하나만 가져옴
                val = row['AUPRC']
                mapped = get_mapped_metric('AUPRC', 'AUPRC')
                
                if mapped:
                    if cond == 'mean':
                        mean_metrics[mapped] = val
                    else:
                        if 'ctrl' in str(cond): continue
                        variant_metrics[cond][mapped] = val
        except Exception as e:
            print(f"Warning: Error parsing AUPRC stats csv: {e}")


    # -------------------------------------------------------------------------
    # 3. Final Format Conversion
    # -------------------------------------------------------------------------
    output_rows = []

    def clean_metadata(meta_dict):
        d = meta_dict.copy()
        if 'raw_parts' in d:
            del d['raw_parts']
        return d
    
    for variant, metrics in sorted(variant_metrics.items()):
        row_data = clean_metadata(metadata)
        row_data['variant'] = variant
        row_data.update(metrics)
        output_rows.append(row_data)

    if mean_metrics:
        mean_row_data = clean_metadata(metadata)
        mean_row_data['variant'] = 'mean'
        mean_row_data.update(mean_metrics)
        output_rows.append(mean_row_data)

    return output_rows

# =============================================================================
# 4. Command Execution Helpers
# =============================================================================

def run_command(cmd_list, description):
    """현재 환경에서 subprocess 실행"""
    print(f"  - {description} 실행 중...")
    try:
        subprocess.run(cmd_list, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print(f"  - ✔️ {description} 완료.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  - [오류] {description} 실패! STDERR:\n{e.stderr.decode('utf-8')}")
        return False

def run_command_with_env(cmd_str, env_name, description):
    """
    다른 Conda 환경에서 명령어 실행
    bash의 source 기능을 이용해 환경을 활성화한 후 명령어를 실행합니다.
    cmd_str: 실행할 명령어 전체 문자열 (예: "python script.py --arg val")
    """
    print(f"  - {description} 실행 중 (Environment: {env_name})...")
    
    # conda initialize 경로를 찾습니다 (일반적인 Linux 서버 기준)
    # 만약 conda 경로가 다르다면 수정이 필요할 수 있습니다.
    conda_sh_path = os.popen("conda info --base").read().strip() + "/etc/profile.d/conda.sh"
    
    # bash 명령어 구성: conda.sh 로딩 -> env 활성화 -> 명령어 실행
    full_cmd = f"source {conda_sh_path} && conda activate {env_name} && {cmd_str}"
    
    try:
        # shell=True 대신 /bin/bash -c 를 명시적으로 사용하여 실행
        subprocess.run(['/bin/bash', '-c', full_cmd], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print(f"  - ✔️ {description} 완료.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  - [오류] {description} 실패! STDERR:\n{e.stderr.decode('utf-8')}")
        return False


def run_merged_genewise_evaluation(input_dir, output_root, script_path):
    """
    모든 파일을 스캔하여 Cell+PLM+Emb 별로 1-3, 2-3, 3-3을 합친 후 Gene-wise 실행
    결과를 Dictionary로 반환
    """
    print("\n" + "="*60)
    print("🚀 [Step 1] Gene-wise Merged Evaluation 시작...")
    print("="*60)

    # 1. 파일 그룹화
    # Key: (CellLine, PLM, Embedding), Value: list of {'pred': path, 'fold': '3-3'}
    groups = defaultdict(list)
    
    pred_files = sorted([f for f in os.listdir(input_dir) if f.endswith("_pred.h5ad")])
    for f in pred_files:
        run_name = f.replace('_pred.h5ad', '')
        meta = parse_run_name(run_name)
        if not meta['cell_line']: continue
        
        # 그룹 식별 키 (Fold 제외)
        # meta['raw_parts'] 사용: ['hct116', 'protT5', 'alt'] -> tuple
        # 실제 Cell/PLM/Emb 값은 매핑된 값일 수 있으므로 raw 값이나 파싱된 값 사용
        # 파싱된 값(Standardized)을 사용하는 것이 안전
        group_key = (meta['cell_line'], meta['plm'], meta['embedding'])
        
        groups[group_key].append({
            'pred_path': os.path.join(input_dir, f),
            'fold': meta['fold'],
            'run_name': run_name,
            'raw_cell': meta['raw_parts'][0] # truth 파일 찾기용
        })

    # 결과 캐시: { (Cell, PLM, Emb): { variant: metrics... } }
    gw_cache = {}
    temp_merge_dir = os.path.join(output_root, 'merged_temp')
    if os.path.exists(temp_merge_dir): shutil.rmtree(temp_merge_dir)
    os.makedirs(temp_merge_dir)

    for group_key, file_list in tqdm(groups.items(), desc="Gene-wise Merged Groups"):
        cell_line, plm, emb = group_key
        # print(f"  Processing Group: {cell_line} | {plm} | {emb} (Files: {len(file_list)})")

        ad_preds = []
        ad_truths = []
        
        valid_merge = True
        
        # 2. 데이터 로드 및 병합 준비
        for file_info in file_list:
            fold = file_info['fold']
            raw_cell = file_info['raw_cell'] # e.g., hct116
            
            # Truth Path
            truth_fname = f"{raw_cell.lower()}_{fold}.h5ad"
            truth_path = os.path.join(TRUTH_DIR, truth_fname)
            
            if not os.path.exists(truth_path):
                print(f"    [Skip] Truth not found for {file_info['run_name']}")
                valid_merge = False; break
                
            try:
                # Load Pred
                curr_pred = ad.read_h5ad(file_info['pred_path'])
                clean_condition_column(curr_pred)
                
                # Load Truth
                curr_truth = ad.read_h5ad(truth_path)
                clean_condition_column(curr_truth)
                
                # Gene Filter (Truth 기준 exist=1)
                if 'exist' in curr_truth.var.columns:
                    mask = curr_truth.var['exist'] == 1
                    curr_truth = curr_truth[:, mask].copy()
                    curr_pred = curr_pred[:, mask].copy()
                
                ad_preds.append(curr_pred)
                ad_truths.append(curr_truth)
                
            except Exception as e:
                print(f"    [Error] Loading {file_info['run_name']}: {e}")
                valid_merge = False; break
        
        if not valid_merge or not ad_preds:
            continue

        # 3. Concat (Merge)
        try:
            merged_pred = ad.concat(ad_preds)
            merged_truth = ad.concat(ad_truths)
        except Exception as e:
            print(f"    [Error] Merge failed for {group_key}: {e}")
            continue
            
        # 4. 임시 저장
        group_id_str = f"{cell_line}_{plm}_{emb}".replace(" ", "")
        merged_pred_path = os.path.join(temp_merge_dir, f"{group_id_str}_merged_pred.h5ad")
        merged_truth_path = os.path.join(temp_merge_dir, f"{group_id_str}_merged_truth.h5ad")
        merged_out_dir = os.path.join(temp_merge_dir, f"{group_id_str}_out")
        os.makedirs(merged_out_dir, exist_ok=True)
        
        merged_pred.write_h5ad(merged_pred_path)
        merged_truth.write_h5ad(merged_truth_path)

        # 5. Gene-wise Script 실행
        cmd_gw = [
            'python', script_path,
            '--truth', merged_truth_path,
            '--pred', merged_pred_path,
            '--condition_col', 'condition',
            '--output', merged_out_dir
        ]
        
        try:
            subprocess.run(cmd_gw, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"    [Error] Gene-wise script failed for {group_key}")
            continue

        # 6. 결과 파싱 및 캐싱
        summary_csv = os.path.join(merged_out_dir, 'summary_stats.csv')
        if os.path.exists(summary_csv):
            try:
                df_stats = pd.read_csv(summary_csv)
                if not df_stats.empty:
                    row = df_stats.iloc[0]
                    metrics_dict = {}
                    
                    # mean_pcc -> GW_PCC...
                    mapped_pcc = get_mapped_metric('gene_wise', 'mean_pcc')
                    if mapped_pcc and 'mean_pcc' in row:
                        metrics_dict[mapped_pcc] = row['mean_pcc']
                    
                    # mean_mse -> GW_MSE...
                    mapped_mse = get_mapped_metric('gene_wise', 'mean_mse')
                    if mapped_mse and 'mean_mse' in row:
                        metrics_dict[mapped_mse] = row['mean_mse']

                    gw_cache[group_key] = {'TOTAL_AVERAGE': metrics_dict}
            except Exception as e:
                print(f"    [Error] Parsing summary_stats.csv: {e}")
    # 임시 파일 정리
    if os.path.exists(temp_merge_dir): shutil.rmtree(temp_merge_dir)
    print("✅ Gene-wise Merged Evaluation 완료.")
    return gw_cache

# =============================================================================
# 5. Main Execution
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="GeneCompass Comprehensive Evaluation Script")
    parser.add_argument('--mode', type=str, required=True, choices=['cell_eval', 'gears_systema', 'auprc'])
    parser.add_argument('--model', type=str, required=True,
                        help='자기 모델 이름 대소문자 구분해서 정확히 적어야함')
    parser.add_argument('--date', type=int, required=True)
    args = parser.parse_args()

    # 경로 설정
    # INPUT_DIR = f"/NFS_DATA/samsung/{args.model}/temp_eval_{args.date}/"
    INPUT_DIR = f"/NFS_DATA/samsung/database/benchmark_figure/ann_dataset/{args.model}/"
    OUTPUT_DIR = f"/NFS_DATA/samsung/database/benchmark_figure/eval_result_NZ/{args.model}/{args.date}/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 각 스크립트의 절대 경로
    SCRIPT_PATHS = {
        'gears': "/home/tech/variantseq/eugenie/cell-eval/variant-cell-eval_nz/src/cell_eval/gears/run_gears.py",
        'systema': "/home/tech/variantseq/eugenie/cell-eval/variant-cell-eval_nz/src/cell_eval/systema/run_systema_v2.py",
        'gene_wise': "/home/tech/variantseq/eugenie/cell-eval/variant-cell-eval_nz/src/cell_eval/gene_wise_pcc/condition_v4.py",
        'auprc': "/home/tech/variantseq/eugenie/cell-eval/variant-cell-eval_nz/src/cell_eval/AUPRC_paper/main.py"
    }

    # 임시 폴더
    TEMP_OUTDIR = f"/NFS_DATA/samsung/temp_260128_1005/{args.model}/cell-eval-outdir" 
    TEMP_FILTERED_DIR = f"/NFS_DATA/samsung/temp_260128_1005/{args.model}/temp_filtered_h5ads"
    #/NFS_DATA/samsung/temp_260128

    print("="*60)
    print(f"입력 폴더: {INPUT_DIR}")
    print(f"출력 폴더: {OUTPUT_DIR}")
    print("="*60)

    gene_wise_cache = {}
    if args.mode == 'gears_systema':
        gene_wise_cache = run_merged_genewise_evaluation(
            INPUT_DIR, 
            f"/NFS_DATA/samsung/temp_260128_1005/{args.model}", 
            SCRIPT_PATHS['gene_wise']
        )

    # 초기화
    if os.path.exists(TEMP_FILTERED_DIR): shutil.rmtree(TEMP_FILTERED_DIR)
    os.makedirs(TEMP_FILTERED_DIR)
    if os.path.exists(TEMP_OUTDIR): shutil.rmtree(TEMP_OUTDIR)
    os.makedirs(TEMP_OUTDIR)

    # 파일 탐색: _pred.h5ad 만 찾음
    pred_files = []
    for filename in sorted(os.listdir(INPUT_DIR)):
        if filename.endswith("_pred.h5ad"):
             pred_files.append(filename)

    all_results_data = []
    total_runs = 0

    try:
        for pred_filename in tqdm(pred_files, desc=f"Run: {args.mode}"):
            pred_path = os.path.join(INPUT_DIR, pred_filename)
            
            # run_name 추출: "1104_..._pred.h5ad" -> "1104_..." (접미사 제거)
            # 파일명 규칙에 따라 _pred.h5ad 앞부분이 run_name이라고 가정
            run_name = pred_filename.replace('_pred.h5ad', '')
            
            total_runs += 1
            print(f"\n--- 작업 #{total_runs}: [{run_name}] ---")

            # 1. 메타데이터 파싱 및 Truth 경로 결정
            metadata = parse_run_name(run_name)
            cell_line, fold = metadata['cell_line'], metadata['fold']

            if not cell_line or not fold:
                print(f"  [Skip] Cell Line 또는 Fold 정보를 파싱할 수 없습니다. (Run: {run_name})")
                continue
            
            truth_filename = f"{cell_line.lower()}_{fold}.h5ad"
            raw_truth_path = os.path.join(TRUTH_DIR, truth_filename)
            print(f"  -> Detected Cell Line: {cell_line}, Using Truth: {raw_truth_path}")

            if not os.path.exists(raw_truth_path):
                print(f"  [Skip] 해당 Truth 파일이 존재하지 않습니다: {raw_truth_path}")
                continue

            # 2. 데이터 로딩 및 필터링
            # (1) Truth 로드 및 Clean
            try:
                adata_truth = ad.read_h5ad(raw_truth_path)
                clean_condition_column(adata_truth)
            except Exception as e:
                print(f"  [Error] Truth 파일을 읽을 수 없습니다: {e}")
                continue

            # (2) Pred 로드 및 Clean
            try:
                adata_pred_full = ad.read_h5ad(pred_path)
                clean_condition_column(adata_pred_full)
            except Exception as e:
                print(f"  [Error] Pred 파일을 읽을 수 없습니다: {e}")
                continue

            # (3) Gene Filtering (Truth의 'exist' 컬럼 기준)
            if 'exist' in adata_truth.var.columns:
                gene_mask = adata_truth.var['exist'] == 1
                adata_truth_filtered = adata_truth[:, gene_mask].copy()
                adata_pred_filtered = adata_pred_full[:, gene_mask].copy()
            else:
                adata_truth_filtered = adata_truth.copy()
                adata_pred_filtered = adata_pred_full.copy()

            filtered_truth_path = os.path.join(TEMP_FILTERED_DIR, f"{run_name}_truth.h5ad")
            filtered_pred_path = os.path.join(TEMP_FILTERED_DIR, f"{run_name}_pred.h5ad")
            adata_truth_filtered.write_h5ad(filtered_truth_path)
            adata_pred_filtered.write_h5ad(filtered_pred_path)
            
            final_dest_path = os.path.join(OUTPUT_DIR, run_name)

            # -----------------------------------------------------------------
            # Mode: Comprehensive Evaluation (GEARS + Systema + Gene-wise + AUPRC)
            # -----------------------------------------------------------------
            if args.mode == 'gears_systema': # 사용자가 요청한 모드 이름 유지 (내부적으로는 풀코스 실행)
                
                # 각 툴별 임시 출력 폴더
                gears_out = os.path.join(TEMP_OUTDIR, 'gears')
                sys_out = os.path.join(TEMP_OUTDIR, 'systema')
                gw_out = os.path.join(TEMP_OUTDIR, 'gene_wise')
                auprc_out = os.path.join(TEMP_OUTDIR, 'auprc')
                
                for d in [gears_out, sys_out, auprc_out]:
                    if os.path.exists(d): shutil.rmtree(d)
                    os.makedirs(d, exist_ok=True)

                # --- 1. GEARS (Current Env) ---
                cmd_gears = [
                    'python', SCRIPT_PATHS['gears'],
                    '--pred', filtered_pred_path,
                    '--truth', filtered_truth_path,
                    '--pert_col', 'condition',
                    '--control', 'ctrl',
                    '--outdir', gears_out
                ]
                run_command(cmd_gears, "GEARS")

                # --- 2. Systema (Current Env) ---
                cmd_sys = [
                    'python', SCRIPT_PATHS['systema'],
                    '--adata_pred_path', filtered_pred_path,
                    '--adata_truth_path', filtered_truth_path,
                    '--pert_col', 'condition',
                    '--control_pert', 'ctrl',
                    '--outdir', sys_out
                ]
                run_command(cmd_sys, "Systema")

                # --- 3. Gene-wise (Current Env) ---
                # python condition_v4.py --truth ... --pred ... --condition_col condition --output ...
                cmd_gw = [
                    'python', SCRIPT_PATHS['gene_wise'],
                    '--truth', filtered_truth_path,
                    '--pred', filtered_pred_path,
                    '--condition_col', 'condition',
                    '--output', gw_out
                ]
                run_command(cmd_gw, "Gene-wise PCC/RMSE")

                # --- 4. AUPRC (Different Env: r-cmap-env) ---
                # python main.py --truth ... --pred ... --output ...
                # 문자열로 명령어 구성 (경로에 공백이 없다고 가정)
                auprc_cmd_str = f"python {SCRIPT_PATHS['auprc']} --truth {filtered_truth_path} --pred {filtered_pred_path} --output {auprc_out}"
                run_command_with_env(auprc_cmd_str, "r-cmap-env", "AUPRC")

                # --- 결과 통합 ---
                combined_temp = os.path.join(TEMP_OUTDIR, 'combined')
                if os.path.exists(combined_temp): shutil.rmtree(combined_temp)
                os.makedirs(combined_temp)

                # 4개 폴더의 모든 파일을 combined_temp로 복사
                for src_dir in [gears_out, sys_out, gw_out, auprc_out]:
                    if os.path.exists(src_dir):
                        for f in os.listdir(src_dir):
                            # 중복 파일명 방지를 위해 덮어쓰기 (서로 다른 csv 이름을 쓴다고 가정)
                            shutil.copy(os.path.join(src_dir, f), os.path.join(combined_temp, f))
                
                # CSV 파싱 및 집계
                run_results = process_run_results(combined_temp, run_name, gene_wise_cache)
                all_results_data.extend(run_results)
                
                # 결과 폴더 백업
                if os.path.exists(final_dest_path): shutil.rmtree(final_dest_path)
                shutil.copytree(combined_temp, final_dest_path)
                print(f"    -> 통합 결과 저장 완료: {final_dest_path}")

            # -----------------------------------------------------------------
            # Mode: AUPRC ONLY (Standalone)
            # -----------------------------------------------------------------
            elif args.mode == 'auprc':
                # 독립 실행 모드 (기존 유지, 필요한 경우 사용)
                if not os.path.exists(final_dest_path): os.makedirs(final_dest_path)
                auprc_cmd_str = f"python {SCRIPT_PATHS['auprc']} --truth {filtered_truth_path} --pred {filtered_pred_path} --output {final_dest_path}"
                success = run_command_with_env(auprc_cmd_str, "r-cmap-env", "AUPRC (Standalone)")
                
                # Standalone 모드일 때도 CSV 집계를 원하면 아래 주석 해제하여 로직 추가 가능
                # 하지만 보통 Plot만 보려는 용도라면 유지
                if success:
                    print(f"    -> AUPRC 결과 저장 완료: {final_dest_path}")

            # -----------------------------------------------------------------
            # Mode: Cell-Eval Original
            # -----------------------------------------------------------------
            elif args.mode == 'cell_eval':
                default_out = "cell-eval-outdir"
                if os.path.exists(default_out): shutil.rmtree(default_out)
                
                cmd_base = [
                    'cell-eval', 'run', 
                    '-ar', filtered_truth_path, 
                    '-ap', filtered_pred_path,
                    '--control-pert', 'ctrl', 
                    '--pert-col', 'condition'
                ]
                if run_command(cmd_base, "Cell-Eval"):
                    run_results = process_run_results(default_out, run_name, None) # cell_eval 모드는 gw cache 사용 안 함
                    all_results_data.extend(run_results)

    finally:
        if os.path.exists(TEMP_FILTERED_DIR):
            shutil.rmtree(TEMP_FILTERED_DIR)
            print(f"\n임시 필터링 폴더 삭제 완료.")

    # ⭐️ 최종 통합 CSV 파일 생성
    if args.mode != 'auprc' and all_results_data: # AUPRC 단독 모드가 아니면 집계
        print("\n" + "="*60)
        print("모든 실행 완료. 최종 통합 CSV 파일을 생성합니다...")

        final_df = pd.DataFrame(all_results_data)

        # 컬럼 순서 정리 (식별자 + 메트릭 알파벳순)
        id_cols = ['cell_line', 'plm', 'embedding', 'fold', 'variant']
        metric_cols = sorted([col for col in final_df.columns if col not in id_cols])
        final_cols = id_cols + metric_cols
        
        # DataFrame에 없는 컬럼이 final_cols에 있을 경우를 대비
        final_cols_exist = [col for col in final_cols if col in final_df.columns]
        final_df = final_df[final_cols_exist]

        output_csv_path = os.path.join('/NFS_DATA/samsung/database/benchmark_figure/pred_dataset_NZ/', f"{args.model}_total.csv")
        final_df.to_csv(output_csv_path, index=False)
        print(f"✔️ 최종 통합 CSV 저장 완료: {output_csv_path}")
    
    print("\n" + "="*60)
    print(f"총 {total_runs}개의 평가 작업을 완료했습니다.")
    print("="*60)

if __name__ == "__main__":
    main()