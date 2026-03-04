import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy import sparse
import anndata
from sklearn.metrics import mean_squared_error

def get_pseudobulk_matrix(adata: anndata.AnnData, condition_col: str):

    if condition_col not in adata.obs.columns:
        raise ValueError(f"'{condition_col}' column not found in adata.obs")

    conditions = [c for c in adata.obs[condition_col].unique() if c != 'ctrl']
    conditions = sorted(conditions)
    
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

        total_count = subset.n_obs

        mean_val = sum_vals / total_count
        means_list.append(mean_val)
        valid_conditions.append(cond)



    if not means_list:
        return pd.DataFrame()
        
    return pd.DataFrame(means_list, index=valid_conditions, columns=adata.var_names)
    
def compute_inter_variant_metrics(truth_df: pd.DataFrame, pred_df: pd.DataFrame):
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

        if np.std(vec_t) == 0 or np.std(vec_p) == 0:
            pcc, pval = np.nan, np.nan
        else:
            pcc, pval = pearsonr(vec_t, vec_p)
            
        mse = mean_squared_error(vec_t, vec_p)
        
        results.append({
            'gene_name': gene,
            'pcc': pcc,
            'p_value': pval,
            'mse': mse,
            'mean_expr_truth': np.mean(vec_t),
            'mean_expr_pred': np.mean(vec_p)
        })
        
    return pd.DataFrame(results)