
import anndata as ad
import numpy as np
import scanpy as sc
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error as mse
from typing import Dict, List
from scipy.sparse import issparse

def mean_nonzero(data, axis=0):
    """
    Compute standard mean including zero values.
    (Function name kept for compatibility, but logic computes standard mean)
    
    Args:
        data: numpy array or sparse matrix
        axis: axis along which to compute the mean (0 for columns, 1 for rows)
        
    Returns:
        numpy array with mean values (zeros are included in the calculation)
    """
    # 희소 행렬(sparse matrix)인 경우 밀집 행렬(dense array)로 변환
    if issparse(data):
        data = data.toarray()
    
    # 0을 제외하는 마스킹 로직을 제거하고, 전체 데이터에 대해 단순 평균 계산
    result = np.mean(data, axis=axis)
    
    return result

def create_de_genes_dictionary(
    adata: ad.AnnData, 
    n_top_genes: int = 20,
    uns_key: str = 'rank_genes_groups_cov_all' # AnnData에 저장된 실제 키 이름 사용
) -> Dict:
    print(f"INFO: Creating DE genes dictionary from 'adata.uns[{uns_key}]'...")
    if uns_key not in adata.uns:
        raise KeyError(f"'{uns_key}' 키를 adata.uns에서 찾을 수 없습니다.")

    # 유전자 이름을 인덱스로 변환하기 위한 맵 생성
    gene_to_idx = {gene: i for i, gene in enumerate(adata.var_names)}
    
    # anndata에 저장된 DEG 결과 딕셔너리를 직접 가져옵니다.
    deg_results = adata.uns[uns_key]
    de_genes_dict = {}

    # deg_results의 키(group)와 값(top_gene_names)을 직접 순회합니다.
    for group, top_gene_names in deg_results.items():
        # 이미 정렬된 유전자 이름 리스트이므로 상위 n개만 선택합니다.
        selected_genes = top_gene_names[:n_top_genes]
        
        # 유전자 이름을 인덱스로 변환합니다.
        gene_indices = [gene_to_idx[gene] for gene in selected_genes if gene in gene_to_idx]
        
        de_genes_dict[group] = gene_indices
    
    print(f"INFO: Dictionary created for {len(de_genes_dict)} conditions.")
    return de_genes_dict

def create_results_from_anndata(
    adata_pred: ad.AnnData, 
    adata_truth: ad.AnnData,
    de_genes_dict: Dict
) -> Dict:
    """두 AnnData로부터 분석에 필요한 test_res 딕셔너리를 생성합니다 (eval_perturb 대체)."""

    adata_pred = adata_pred[adata_pred.obs.sort_values(by='condition', ascending=False).index].copy()
    adata_truth = adata_truth[adata_truth.obs.sort_values(by='condition', ascending=False).index].copy()
    
    pert2pert_full_id = dict(adata_truth.obs[['condition', 'condition_name']].values)
    
    pert_cat = adata_truth.obs['condition'].values
    pred_matrix = adata_pred.X.toarray() if hasattr(adata_pred.X, 'toarray') else adata_pred.X
    truth_matrix = adata_truth.X.toarray() if hasattr(adata_truth.X, 'toarray') else adata_truth.X

    pred_de_list, truth_de_list, de_pert_cat = [], [], []
    for pert in np.unique(pert_cat):
        full_pert_name = pert2pert_full_id.get(pert)
        if not full_pert_name: continue
            
        de_indices = de_genes_dict.get(full_pert_name)
        if not de_indices: continue
        
        cell_mask = (pert_cat == pert)
        if cell_mask.sum() > 0:
            num_cells_in_pert = cell_mask.sum()
            pred_de_list.append(pred_matrix[cell_mask][:, de_indices])
            truth_de_list.append(truth_matrix[cell_mask][:, de_indices])
            de_pert_cat.extend([pert] * num_cells_in_pert)

    results = {
        "pert_cat": pert_cat,
        "pred": pred_matrix,
        "truth": truth_matrix,
        "pred_de": np.concatenate(pred_de_list, axis=0) if pred_de_list else np.array([]),
        "truth_de": np.concatenate(truth_de_list, axis=0) if truth_de_list else np.array([]),
        "pert_cat_de": np.array(de_pert_cat) 
    }
    return results

def compute_metrics(results: Dict) -> tuple[Dict, Dict]:
    """
    주어진 results 딕셔너리로부터 기본 메트릭(MSE, Pearson)을 계산합니다.
    """
    print("INFO: Computing basic metrics (MSE, Pearson)...")
    metrics_pert = {}
    
    metric2fct = {
        'mse': mse,
        'pearson': lambda x, y: pearsonr(x, y)[0] if np.std(x) > 0 and np.std(y) > 0 else 0.0
    }
    
    metrics_agg = {m: [] for m in metric2fct.keys()}
    metrics_agg.update({m + '_de': [] for m in metric2fct.keys()})

    for pert in np.unique(results['pert_cat']):
        metrics_pert[pert] = {}
        
        # 전체 유전자에 대한 인덱스
        p_idx = np.where(results['pert_cat'] == pert)[0]
        if len(p_idx) == 0:
            continue

        pred_pert_mean = mean_nonzero(results['pred'][p_idx], axis=0)
        truth_pert_mean = mean_nonzero(results['truth'][p_idx], axis=0)

        # 전체 유전자에 대한 메트릭 계산
        for m, fct in metric2fct.items():
            val = fct(truth_pert_mean, pred_pert_mean)
            metrics_pert[pert][m] = val if not np.isnan(val) else 0
            metrics_agg[m].append(metrics_pert[pert][m])

        # DE 유전자에 대한 메트릭 계산
        if pert.lower() != 'control' and 'pert_cat_de' in results:
            p_idx_de = np.where(results['pert_cat_de'] == pert)[0]

            if len(p_idx_de) > 0:
                pred_de_pert_mean = mean_nonzero(results['pred_de'][p_idx_de], axis=0)
                truth_de_pert_mean = mean_nonzero(results['truth_de'][p_idx_de], axis=0)

                for m, fct in metric2fct.items():
                    val = fct(truth_de_pert_mean, pred_de_pert_mean)
                    metrics_pert[pert][m + '_de'] = val if not np.isnan(val) else 0
                    metrics_agg[m + '_de'].append(metrics_pert[pert][m + '_de'])
            else:
                for m in metric2fct.keys():
                    metrics_pert[pert][m + '_de'] = 0
        else:
            for m in metric2fct.keys():
                metrics_pert[pert][m + '_de'] = 0

    final_metrics = {key: np.mean(val) for key, val in metrics_agg.items() if val}
    
    print("INFO: Basic metrics computation complete.")
    return final_metrics, metrics_pert

def get_control_expression(adata: ad.AnnData) -> np.ndarray:
    control_name = 'ctrl'
    if control_name not in adata.obs['condition'].unique():
        return np.zeros(adata.n_vars)
    
    ctrl_adata = adata[adata.obs['condition'] == control_name]
    
    ctrl_expression = mean_nonzero(ctrl_adata.X, axis=0)
        
    return ctrl_expression
    
def deeper_analysis(adata: ad.AnnData, test_res: Dict, ctrl_expression: np.ndarray, control_pert: str, most_variable_genes: np.ndarray = None, uns_key: str = 'rank_genes_groups_cov_all') -> Dict:
    metric2fct = {
           'pearson': pearsonr,
           'mse': mse
    }

    pert_metric = {}

    obs_df = adata.obs[['condition', 'condition_name']].dropna()
    pert2pert_full_id = dict(obs_df.values)
    geneid2name = dict(zip(adata.var.index.values, adata.var['gene_name']))
    geneid2idx = dict(zip(adata.var.index.values, range(len(adata.var.index.values))))

    ctrl = ctrl_expression.reshape(1, -1)
    
    if most_variable_genes is None:
        unique_conditions = adata.obs.condition.unique()
        mean_expressions = np.array([mean_nonzero(adata[adata.obs.condition == c].X, axis=0) for c in unique_conditions])
        most_variable_genes = np.argsort(np.std(mean_expressions, axis=0))[-200:]
        
    gene_list = adata.var['gene_name'].values

    for pert in np.unique(test_res['pert_cat']):
        if pert == control_pert:
            continue
        pert_metric[pert] = {}
        full_pert_name = pert2pert_full_id[pert]
        if not full_pert_name:
            continue
        if full_pert_name not in adata.uns[uns_key]:
            print(f"WARNING: No DEG results found for group '{full_pert_name}'. Skipping.")
            continue
            
        de_genes_names = adata.uns[uns_key][full_pert_name]
        
        de_idx = [geneid2idx[g] for g in de_genes_names[:20] if g in geneid2idx]
        de_idx_50 = [geneid2idx[g] for g in de_genes_names[:50] if g in geneid2idx]
        de_idx_100 = [geneid2idx[g] for g in de_genes_names[:100] if g in geneid2idx]
        de_idx_200 = [geneid2idx[g] for g in de_genes_names[:200] if g in geneid2idx]

        if len(de_idx) == 0:
             print(f"WARNING: Group '{pert}' has 0 matched DEG indices. Skipping.")
             continue

        pert_idx = np.where(test_res['pert_cat'] == pert)[0]
        pert_idx_de = np.where(test_res['pert_cat_de'] == pert)[0]
        if len(pert_idx_de) == 0: continue
        pred_mean = mean_nonzero(test_res['pred_de'][pert_idx_de], axis=0).reshape(-1,)
        true_mean = mean_nonzero(test_res['truth_de'][pert_idx_de], axis=0).reshape(-1,)
        
        pred_pert_mean = mean_nonzero(test_res['pred'][pert_idx], axis=0)
        truth_pert_mean = mean_nonzero(test_res['truth'][pert_idx], axis=0)
        direc_change = np.abs(np.sign(pred_pert_mean - ctrl[0]) - np.sign(truth_pert_mean - ctrl[0]))            
        frac_correct_direction = len(np.where(direc_change == 0)[0])/len(geneid2name)
        pert_metric[pert]['frac_correct_direction_all'] = frac_correct_direction

        de_idx_map = {20: de_idx,
                      50: de_idx_50,
                      100: de_idx_100,
                      200: de_idx_200
                     }
        
        for val in [20, 50, 100, 200]:
            
            direc_change = np.abs(np.sign(pred_pert_mean[de_idx_map[val]] - ctrl[0][de_idx_map[val]]) - np.sign(truth_pert_mean[de_idx_map[val]] - ctrl[0][de_idx_map[val]]))            
            frac_correct_direction = len(np.where(direc_change == 0)[0])/val
            pert_metric[pert]['frac_correct_direction_' + str(val)] = frac_correct_direction

        mean = mean_nonzero(test_res['truth_de'][pert_idx_de], axis=0)
        std = np.std(test_res['truth_de'][pert_idx_de], axis = 0)
        min_ = np.min(test_res['truth_de'][pert_idx_de], axis = 0)
        max_ = np.max(test_res['truth_de'][pert_idx_de], axis = 0)
        q25 = np.quantile(test_res['truth_de'][pert_idx_de], 0.25, axis = 0)
        q75 = np.quantile(test_res['truth_de'][pert_idx_de], 0.75, axis = 0)
        q55 = np.quantile(test_res['truth_de'][pert_idx_de], 0.55, axis = 0)
        q45 = np.quantile(test_res['truth_de'][pert_idx_de], 0.45, axis = 0)
        q40 = np.quantile(test_res['truth_de'][pert_idx_de], 0.4, axis = 0)
        q60 = np.quantile(test_res['truth_de'][pert_idx_de], 0.6, axis = 0)

        zero_des = np.intersect1d(np.where(min_ == 0)[0], np.where(max_ == 0)[0])
        nonzero_des = np.setdiff1d(list(range(20)), zero_des)
        if len(nonzero_des) == 0:
            pass
        else:            
            
            direc_change = np.abs(np.sign(pred_mean[nonzero_des] - ctrl[0][de_idx][nonzero_des]) - np.sign(true_mean[nonzero_des] - ctrl[0][de_idx][nonzero_des]))            
            frac_correct_direction = len(np.where(direc_change == 0)[0])/len(nonzero_des)
            pert_metric[pert]['frac_correct_direction_20_nonzero'] = frac_correct_direction
            
            in_range = (pred_mean[nonzero_des] >= min_[nonzero_des]) & (pred_mean[nonzero_des] <= max_[nonzero_des])
            frac_in_range = sum(in_range)/len(nonzero_des)
            pert_metric[pert]['frac_in_range'] = frac_in_range

            in_range_5 = (pred_mean[nonzero_des] >= q45[nonzero_des]) & (pred_mean[nonzero_des] <= q55[nonzero_des])
            frac_in_range_45_55 = sum(in_range_5)/len(nonzero_des)
            pert_metric[pert]['frac_in_range_45_55'] = frac_in_range_45_55

            in_range_10 = (pred_mean[nonzero_des] >= q40[nonzero_des]) & (pred_mean[nonzero_des] <= q60[nonzero_des])
            frac_in_range_40_60 = sum(in_range_10)/len(nonzero_des)
            pert_metric[pert]['frac_in_range_40_60'] = frac_in_range_40_60

            in_range_25 = (pred_mean[nonzero_des] >= q25[nonzero_des]) & (pred_mean[nonzero_des] <= q75[nonzero_des])
            frac_in_range_25_75 = sum(in_range_25)/len(nonzero_des)
            pert_metric[pert]['frac_in_range_25_75'] = frac_in_range_25_75

            zero_idx = np.where(std > 0)[0]
            if len(zero_idx) > 0:
                sigma = (np.abs(pred_mean[zero_idx] - mean[zero_idx]))/(std[zero_idx])
                pert_metric[pert]['mean_sigma'] = np.mean(sigma)
                pert_metric[pert]['std_sigma'] = np.std(sigma)
                pert_metric[pert]['frac_sigma_below_1'] = 1 - len(np.where(sigma > 1)[0])/len(zero_idx)
                pert_metric[pert]['frac_sigma_below_2'] = 1 - len(np.where(sigma > 2)[0])/len(zero_idx)

        p_idx = np.where(test_res['pert_cat'] == pert)[0]

        for m, fct in metric2fct.items():
            if m != 'mse':
                val = fct(pred_pert_mean - ctrl[0], truth_pert_mean - ctrl[0])[0]
                if np.isnan(val):
                    val = 0

                pert_metric[pert][m + '_delta'] = val
                
                val = fct(pred_pert_mean[de_idx] - ctrl[0][de_idx], truth_pert_mean[de_idx] - ctrl[0][de_idx])[0]
                if np.isnan(val):
                    val = 0

                pert_metric[pert][m + '_delta_de'] = val
        pert_mean = mean_nonzero(test_res['truth'][p_idx], axis=0).reshape(-1,)

        fold_change = pert_mean/ctrl
        fold_change[np.isnan(fold_change)] = 0
        fold_change[np.isinf(fold_change)] = 0
        fold_change[0][np.where(pert_mean < 0.5)[0]] = 0

        o =  np.where(fold_change[0] > 0)[0]

        pred_fc = pred_pert_mean[o]
        true_fc = truth_pert_mean[o]
        ctrl_fc = ctrl[0][o]

        if len(o) > 0:
            pert_metric[pert]['fold_change_gap_all'] = np.mean(np.abs(pred_fc/ctrl_fc - true_fc/ctrl_fc))


        o = np.intersect1d(np.where(fold_change[0] <0.333)[0], np.where(fold_change[0] > 0)[0])

        pred_fc = pred_pert_mean[o]
        true_fc = truth_pert_mean[o]
        ctrl_fc = ctrl[0][o]

        if len(o) > 0:
            pert_metric[pert]['fold_change_gap_downreg_0.33'] = np.mean(np.abs(pred_fc/ctrl_fc - true_fc/ctrl_fc))


        o = np.intersect1d(np.where(fold_change[0] <0.1)[0], np.where(fold_change[0] > 0)[0])

        pred_fc = pred_pert_mean[o]
        true_fc = truth_pert_mean[o]
        ctrl_fc = ctrl[0][o]

        if len(o) > 0:
            pert_metric[pert]['fold_change_gap_downreg_0.1'] = np.mean(np.abs(pred_fc/ctrl_fc - true_fc/ctrl_fc))

        o = np.where(fold_change[0] > 3)[0]

        pred_fc = pred_pert_mean[o]
        true_fc = truth_pert_mean[o]
        ctrl_fc = ctrl[0][o]

        if len(o) > 0:
            pert_metric[pert]['fold_change_gap_upreg_3'] = np.mean(np.abs(pred_fc/ctrl_fc - true_fc/ctrl_fc))

        o = np.where(fold_change[0] > 10)[0]

        pred_fc = pred_pert_mean[o]
        true_fc = truth_pert_mean[o]
        ctrl_fc = ctrl[0][o]

        if len(o) > 0:
            pert_metric[pert]['fold_change_gap_upreg_10'] = np.mean(np.abs(pred_fc/ctrl_fc - true_fc/ctrl_fc))

        if len(most_variable_genes) >= 2:
            for m, fct in metric2fct.items():
                if m != 'mse':

                    val = fct(pred_pert_mean[most_variable_genes] - ctrl[0][most_variable_genes], truth_pert_mean[most_variable_genes] - ctrl[0][most_variable_genes])[0]
                    if np.isnan(val):
                        val = 0
                    pert_metric[pert][m + '_delta_top200_hvg'] = val

                    val = fct(pred_pert_mean[most_variable_genes], truth_pert_mean[most_variable_genes])[0]
                    if np.isnan(val):
                        val = 0
                    pert_metric[pert][m + '_top200_hvg'] = val
                else:

                    val = fct(pred_pert_mean[most_variable_genes], truth_pert_mean[most_variable_genes])
                    pert_metric[pert][m + '_top200_hvg'] = val
        else:
            for m in metric2fct.keys():
                pert_metric[pert][m + '_delta_top200_hvg'] = 0
                pert_metric[pert][m + '_top200_hvg'] = 0


        for m, fct in metric2fct.items():
            if m != 'mse':
                val = fct(pred_pert_mean[de_idx] - ctrl[0][de_idx], truth_pert_mean[de_idx] - ctrl[0][de_idx])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_delta_top20_de'] = val


                val = fct(pred_pert_mean[de_idx], truth_pert_mean[de_idx])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_top20_de'] = val
            else:
                val_delta = fct(pred_pert_mean[de_idx] - ctrl[0][de_idx], truth_pert_mean[de_idx] - ctrl[0][de_idx])
                pert_metric[pert][m + '_delta_top20_de'] = val_delta
                val_abs = fct(pred_pert_mean[de_idx], truth_pert_mean[de_idx])
                pert_metric[pert][m + '_top20_de'] = val_abs

        
        for m, fct in metric2fct.items():
            if m != 'mse':
                val = fct(test_res['pred'][p_idx].mean(0)[de_idx_200] - ctrl[0][de_idx_200], test_res['truth'][p_idx].mean(0)[de_idx_200]-ctrl[0][de_idx_200])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_delta_top200_de'] = val


                val = fct(test_res['pred'][p_idx].mean(0)[de_idx_200], test_res['truth'][p_idx].mean(0)[de_idx_200])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_top200_de'] = val
            else:
                val_delta = fct(test_res['pred'][p_idx].mean(0)[de_idx_200] - ctrl[0][de_idx_200], test_res['truth'][p_idx].mean(0)[de_idx_200]-ctrl[0][de_idx_200])
                pert_metric[pert][m + '_delta_top200_de'] = val_delta
                val_abs = fct(test_res['pred'][p_idx].mean(0)[de_idx_200], test_res['truth'][p_idx].mean(0)[de_idx_200])
                pert_metric[pert][m + '_top200_de'] = val_abs

        for m, fct in metric2fct.items():
            if m != 'mse':

                val = fct(test_res['pred'][p_idx].mean(0)[de_idx_100] - ctrl[0][de_idx_100], test_res['truth'][p_idx].mean(0)[de_idx_100]-ctrl[0][de_idx_100])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_delta_top100_de'] = val


                val = fct(test_res['pred'][p_idx].mean(0)[de_idx_100], test_res['truth'][p_idx].mean(0)[de_idx_100])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_top100_de'] = val
            else:
                val_delta = fct(test_res['pred'][p_idx].mean(0)[de_idx_100] - ctrl[0][de_idx_100], test_res['truth'][p_idx].mean(0)[de_idx_100]-ctrl[0][de_idx_100])
                pert_metric[pert][m + '_delta_top100_de'] = val_delta
                val_abs = fct(test_res['pred'][p_idx].mean(0)[de_idx_100], test_res['truth'][p_idx].mean(0)[de_idx_100])
                pert_metric[pert][m + '_top100_de'] = val_abs

        for m, fct in metric2fct.items():
            if m != 'mse':

                val = fct(test_res['pred'][p_idx].mean(0)[de_idx_50] - ctrl[0][de_idx_50], test_res['truth'][p_idx].mean(0)[de_idx_50]-ctrl[0][de_idx_50])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_delta_top50_de'] = val


                val = fct(test_res['pred'][p_idx].mean(0)[de_idx_50], test_res['truth'][p_idx].mean(0)[de_idx_50])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_top50_de'] = val
            else:
                val_delta = fct(test_res['pred'][p_idx].mean(0)[de_idx_50] - ctrl[0][de_idx_50], test_res['truth'][p_idx].mean(0)[de_idx_50]-ctrl[0][de_idx_50])
                pert_metric[pert][m + '_delta_top50_de'] = val_delta
                val_abs = fct(test_res['pred'][p_idx].mean(0)[de_idx_50], test_res['truth'][p_idx].mean(0)[de_idx_50])
                pert_metric[pert][m + '_top50_de'] = val_abs



    return pert_metric

def non_dropout_analysis(adata: ad.AnnData, test_res: Dict, ctrl_expression: np.ndarray, control_pert: str,
    uns_key_de: str = 'top_non_dropout',    # non dropout DE 유전자 이름 리스트 (Ranked)
    uns_key_all_idx: str = 'non_dropout_gene_idx'   # 전체 Non-dropout 유전자 인덱스 리스트
)-> Dict:
    metric2fct = {
           'pearson': pearsonr,
           'mse': mse
    }

    pert_metric = {}
    
    obs_df = adata.obs[['condition', 'condition_name']].dropna()
    pert2pert_full_id = dict(obs_df.values)
    geneid2idx = dict(zip(adata.var.index.values, range(len(adata.var.index.values))))

    ctrl = ctrl_expression.reshape(1, -1)

    for pert in np.unique(test_res['pert_cat']):
        if pert == control_pert:
            continue
        pert_metric[pert] = {}

        full_pert_name = pert2pert_full_id.get(pert)
        if full_pert_name is None:
            continue

        pert_idx = np.where(test_res['pert_cat'] == pert)[0]  
        
        de_gene_names = adata.uns.get(uns_key_de, {}).get(full_pert_name, [])
        if isinstance(de_gene_names, np.ndarray):
            de_gene_names = de_gene_names.tolist()
            
        if not de_gene_names:
            print(f"Warning: '{uns_key_de}' not found for {full_pert_name}. Skipping DE metrics.")
            de_indices_full = []
        else:

            de_indices_full = [geneid2idx[g] for g in de_gene_names if g in geneid2idx]


             
        pred_pert_mean = mean_nonzero(test_res['pred'][pert_idx], axis=0)
        truth_pert_mean = mean_nonzero(test_res['truth'][pert_idx], axis=0)

        de_idx_map = {
            20: de_indices_full[:20],
            50: de_indices_full[:50],
            100: de_indices_full[:100],
            200: de_indices_full[:200]
        }

        for k, current_de_idx in de_idx_map.items():
            if len(current_de_idx) == 0:
                continue
                
            suffix = f'_top{k}_non_dropout' 


            direc_change = np.abs(np.sign(pred_pert_mean[current_de_idx] - ctrl[0][current_de_idx]) - 
                                  np.sign(truth_pert_mean[current_de_idx] - ctrl[0][current_de_idx]))            
            
            pert_metric[pert][f'frac_correct_direction{suffix}'] = len(np.where(direc_change == 0)[0]) / len(current_de_idx)
            pert_metric[pert][f'frac_opposite_direction{suffix}'] = len(np.where(direc_change == 2)[0]) / len(current_de_idx)
            pert_metric[pert][f'frac_0/1_direction{suffix}'] = len(np.where(direc_change == 1)[0]) / len(current_de_idx)

            for m, fct in metric2fct.items():
                val_raw = fct(pred_pert_mean[current_de_idx], truth_pert_mean[current_de_idx])
                
                if m == 'pearson':
                    val_raw = val_raw[0]
                if np.isnan(val_raw):
                    val_raw = 0
                
                key_raw = f'{m}{suffix}'
                pert_metric[pert][key_raw] = val_raw
                pred_delta = pred_pert_mean[current_de_idx] - ctrl[0][current_de_idx]
                truth_delta = truth_pert_mean[current_de_idx] - ctrl[0][current_de_idx]
                
                val_delta = fct(pred_delta, truth_delta)
                
                if m == 'pearson':
                    val_delta = val_delta[0]
                if np.isnan(val_delta):
                    val_delta = 0
                
                key_delta = f'{m}_delta{suffix}'
                pert_metric[pert][key_delta] = val_delta

    return pert_metric