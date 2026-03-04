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

def clean_condition_column(adata):
    if 'condition' in adata.obs.columns:
        conditions = adata.obs['condition'].astype(str)
        conditions = conditions.str.replace(r'\+ctrl$', '', regex=True)
        adata.obs['condition'] = conditions

def parse_systema_summary_value(value_str):
    match = re.search(r'[\d\.]+', str(value_str))
    return float(match.group(0)) if match else None

def get_mapped_metric(category, original_metric):
    if category in METRIC_MAPPING and original_metric in METRIC_MAPPING[category]:
        return METRIC_MAPPING[category][original_metric]
    return None

def parse_run_name(run_name):
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

def process_run_results(results_dir, run_name, gene_wise_cache):
    """
    GEARS, Systema, Gene-wise, AUPRC 결과를 모두 읽어 통합합니다.
    """
    if not os.path.isdir(results_dir):
        print(f"[오류] 대상 폴더를 찾을 수 없습니다: {results_dir}")
        return []

    metadata = parse_run_name(run_name)
    variant_metrics = defaultdict(dict)
    mean_metrics = {}
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
        config_key = (metadata['cell_line'], metadata['plm'], metadata['embedding'])
        
        if config_key in gene_wise_cache:
            gw_results = gene_wise_cache[config_key]
            
            for variant in variant_metrics.keys():
                if variant in gw_results:
                    variant_metrics[variant].update(gw_results[variant])
            if 'TOTAL_AVERAGE' in gw_results:
                mean_metrics.update(gw_results['TOTAL_AVERAGE'])

    auprc_path = os.path.join(results_dir, 'overlap_stats.csv')
    if os.path.exists(auprc_path):
        try:
            df_auprc = pd.read_csv(auprc_path)
            for _, row in df_auprc.iterrows():
                cond = row['Condition']
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
    print(f"  - {description} 실행 중 (Environment: {env_name})...")
    
    conda_sh_path = os.popen("conda info --base").read().strip() + "/etc/profile.d/conda.sh"

    full_cmd = f"source {conda_sh_path} && conda activate {env_name} && {cmd_str}"
    
    try:
        subprocess.run(['/bin/bash', '-c', full_cmd], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print(f"  - ✔️ {description} 완료.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  - [오류] {description} 실패! STDERR:\n{e.stderr.decode('utf-8')}")
        return False


def run_merged_genewise_evaluation(input_dir, output_root, script_path):
    print("\n" + "="*60)
    print("🚀 [Step 1] Gene-wise Merged Evaluation 시작...")
    print("="*60)

    groups = defaultdict(list)
    
    pred_files = sorted([f for f in os.listdir(input_dir) if f.endswith("_pred.h5ad")])
    for f in pred_files:
        run_name = f.replace('_pred.h5ad', '')
        meta = parse_run_name(run_name)
        if not meta['cell_line']: continue

        group_key = (meta['cell_line'], meta['plm'], meta['embedding'])
        
        groups[group_key].append({
            'pred_path': os.path.join(input_dir, f),
            'fold': meta['fold'],
            'run_name': run_name,
            'raw_cell': meta['raw_parts'][0]
        })
    gw_cache = {}
    temp_merge_dir = os.path.join(output_root, 'merged_temp')
    if os.path.exists(temp_merge_dir): shutil.rmtree(temp_merge_dir)
    os.makedirs(temp_merge_dir)

    for group_key, file_list in tqdm(groups.items(), desc="Gene-wise Merged Groups"):
        cell_line, plm, emb = group_key

        ad_preds = []
        ad_truths = []
        
        valid_merge = True
        
        for file_info in file_list:
            fold = file_info['fold']
            raw_cell = file_info['raw_cell'] 

            truth_fname = f"{raw_cell.lower()}_{fold}.h5ad"
            truth_path = os.path.join(TRUTH_DIR, truth_fname)
            
            if not os.path.exists(truth_path):
                print(f"    [Skip] Truth not found for {file_info['run_name']}")
                valid_merge = False; break
                
            try:

                curr_pred = ad.read_h5ad(file_info['pred_path'])
                clean_condition_column(curr_pred)
                
                curr_truth = ad.read_h5ad(truth_path)
                clean_condition_column(curr_truth)

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

        try:
            merged_pred = ad.concat(ad_preds)
            merged_truth = ad.concat(ad_truths)
        except Exception as e:
            print(f"    [Error] Merge failed for {group_key}: {e}")
            continue

        group_id_str = f"{cell_line}_{plm}_{emb}".replace(" ", "")
        merged_pred_path = os.path.join(temp_merge_dir, f"{group_id_str}_merged_pred.h5ad")
        merged_truth_path = os.path.join(temp_merge_dir, f"{group_id_str}_merged_truth.h5ad")
        merged_out_dir = os.path.join(temp_merge_dir, f"{group_id_str}_out")
        os.makedirs(merged_out_dir, exist_ok=True)
        
        merged_pred.write_h5ad(merged_pred_path)
        merged_truth.write_h5ad(merged_truth_path)

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
        summary_csv = os.path.join(merged_out_dir, 'summary_stats.csv')
        if os.path.exists(summary_csv):
            try:
                df_stats = pd.read_csv(summary_csv)
                if not df_stats.empty:
                    row = df_stats.iloc[0]
                    metrics_dict = {}

                    mapped_pcc = get_mapped_metric('gene_wise', 'mean_pcc')
                    if mapped_pcc and 'mean_pcc' in row:
                        metrics_dict[mapped_pcc] = row['mean_pcc']

                    mapped_mse = get_mapped_metric('gene_wise', 'mean_mse')
                    if mapped_mse and 'mean_mse' in row:
                        metrics_dict[mapped_mse] = row['mean_mse']

                    gw_cache[group_key] = {'TOTAL_AVERAGE': metrics_dict}
            except Exception as e:
                print(f"    [Error] Parsing summary_stats.csv: {e}")
    if os.path.exists(temp_merge_dir): shutil.rmtree(temp_merge_dir)
    print("✅ Gene-wise Merged Evaluation 완료.")
    return gw_cache

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
        'systema': "/home/tech/variantseq/eugenie/cell-eval/variant-cell-eval_nz/src/cell_eval/systema/run_systema.py",
        'gene_wise': "/home/tech/variantseq/eugenie/cell-eval/variant-cell-eval_nz/src/cell_eval/gene_wise_pcc/condition.py",
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
                cmd_gw = [
                    'python', SCRIPT_PATHS['gene_wise'],
                    '--truth', filtered_truth_path,
                    '--pred', filtered_pred_path,
                    '--condition_col', 'condition',
                    '--output', gw_out
                ]
                run_command(cmd_gw, "Gene-wise PCC/RMSE")
                auprc_cmd_str = f"python {SCRIPT_PATHS['auprc']} --truth {filtered_truth_path} --pred {filtered_pred_path} --output {auprc_out}"
                run_command_with_env(auprc_cmd_str, "r-cmap-env", "AUPRC")

                combined_temp = os.path.join(TEMP_OUTDIR, 'combined')
                if os.path.exists(combined_temp): shutil.rmtree(combined_temp)
                os.makedirs(combined_temp)

                for src_dir in [gears_out, sys_out, gw_out, auprc_out]:
                    if os.path.exists(src_dir):
                        for f in os.listdir(src_dir):

                            shutil.copy(os.path.join(src_dir, f), os.path.join(combined_temp, f))
                
                run_results = process_run_results(combined_temp, run_name, gene_wise_cache)
                all_results_data.extend(run_results)
                
                if os.path.exists(final_dest_path): shutil.rmtree(final_dest_path)
                shutil.copytree(combined_temp, final_dest_path)
                print(f"    -> 통합 결과 저장 완료: {final_dest_path}")
            elif args.mode == 'auprc':

                if not os.path.exists(final_dest_path): os.makedirs(final_dest_path)
                auprc_cmd_str = f"python {SCRIPT_PATHS['auprc']} --truth {filtered_truth_path} --pred {filtered_pred_path} --output {final_dest_path}"
                success = run_command_with_env(auprc_cmd_str, "r-cmap-env", "AUPRC (Standalone)")

                if success:
                    print(f"    -> AUPRC 결과 저장 완료: {final_dest_path}")

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
                    run_results = process_run_results(default_out, run_name, None)
                    all_results_data.extend(run_results)

    finally:
        if os.path.exists(TEMP_FILTERED_DIR):
            shutil.rmtree(TEMP_FILTERED_DIR)
            print(f"\n임시 필터링 폴더 삭제 완료.")

    if args.mode != 'auprc' and all_results_data: 
        print("\n" + "="*60)
        print("모든 실행 완료. 최종 통합 CSV 파일을 생성합니다...")

        final_df = pd.DataFrame(all_results_data)
        id_cols = ['cell_line', 'plm', 'embedding', 'fold', 'variant']
        metric_cols = sorted([col for col in final_df.columns if col not in id_cols])
        final_cols = id_cols + metric_cols

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