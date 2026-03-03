import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy import sparse
import anndata
from sklearn.metrics import mean_squared_error

def get_pseudobulk_matrix(adata: anndata.AnnData, condition_col: str):
    """
    AnnData를 입력받아 변이(condition_col)별 유전자 평균 발현량을 계산하여 반환합니다.
    * [수정됨] 평균 계산 시 0인 값은 제외합니다. (Sum / Non-zero Count)
    Returns: pd.DataFrame (Rows: Variants, Cols: Genes)
    """
    if condition_col not in adata.obs.columns:
        raise ValueError(f"'{condition_col}' column not found in adata.obs")

    # 'ctrl'을 제외한 변이들만 추출
    conditions = [c for c in adata.obs[condition_col].unique() if c != 'ctrl']
    conditions = sorted(conditions) # 순서 고정
    
    means_list = []
    valid_conditions = []

    for cond in conditions:
        subset = adata[adata.obs[condition_col] == cond]
        
        if subset.n_obs == 0:
            continue
            
        X = subset.X
        
        if sparse.issparse(X):
            sum_vals = np.array(X.sum(axis=0)).flatten()
        else:
            sum_vals = np.sum(X, axis=0)

        # 분모: 전체 관측치 수 (Total Count) - 0인 값도 개수에 포함됨
        total_count = subset.n_obs

        # 평균 계산: 합계 / 전체 개수
        mean_val = sum_vals / total_count
        # ---------------------------------------------------------------

        means_list.append(mean_val)
        valid_conditions.append(cond)



    if not means_list:
        return pd.DataFrame()
        
    return pd.DataFrame(means_list, index=valid_conditions, columns=adata.var_names)
    
def compute_inter_variant_metrics(truth_df: pd.DataFrame, pred_df: pd.DataFrame):
    """
    변이별 평균값 DataFrame(Truth vs Pred)을 받아 유전자별 PCC/MSE를 계산합니다.
    """
    # 공통 변이만 남기기 (인덱스 교집합)
    common_variants = truth_df.index.intersection(pred_df.index)
    if len(common_variants) < 3:
        print(f"[Warning] Not enough variants for correlation (n={len(common_variants)}). Need at least 3.")
        return pd.DataFrame()

    truth_sub = truth_df.loc[common_variants]
    pred_sub = pred_df.loc[common_variants]
    genes = truth_df.columns
    
    results = []
    
    for gene in genes:
        vec_t = truth_sub[gene].values
        vec_p = pred_sub[gene].values
        
        # 분산 0 체크 (PCC 계산 불가 방지)
        if np.std(vec_t) == 0 or np.std(vec_p) == 0:
            pcc, pval = np.nan, np.nan
        else:
            pcc, pval = pearsonr(vec_t, vec_p)
            
        mse = mean_squared_error(vec_t, vec_p)
        
        results.append({
            'gene_name': gene,
            'pcc': pcc,
            'p_value': pval,
            'mse': mse,  # 'rmse' -> 'mse'
            'mean_expr_truth': np.mean(vec_t),
            'mean_expr_pred': np.mean(vec_p)
        })
        
    return pd.DataFrame(results)