
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import anndata
from scipy.stats import pearsonr
from scipy.sparse import issparse

def mean_nonzero(data, axis=0):

    if issparse(data):
        data = data.toarray()
    result = np.mean(data, axis=axis)
    
    return result


def average_of_perturbation_centroids(
    adata: anndata.AnnData, 
    pert_col: str, 
    control_pert: str
) -> np.ndarray:
    pert_adata = adata[adata.obs[pert_col] != control_pert].copy()

    if pert_adata.n_obs == 0:
        return np.zeros(adata.n_vars)

    pert_means = []
    unique_conditions = pert_adata.obs[pert_col].unique()

    for cond in unique_conditions:
        adata_cond = pert_adata[pert_adata.obs[pert_col] == cond]
        
        pert_mean = mean_nonzero(adata_cond.X, axis=0)
        pert_means.append(pert_mean)
    
    pert_means = np.array(pert_means)

    final_reference = mean_nonzero(pert_means, axis=0)
    
    return final_reference


def calculate_centroid_accuracies(agg_post_pred_df, post_gt_all_df):

    common_genes = post_gt_all_df.columns.intersection(agg_post_pred_df.columns)
    agg_post_pred_df = agg_post_pred_df[common_genes]
    post_gt_all_df = post_gt_all_df[common_genes]

    distances = cdist(agg_post_pred_df.values, post_gt_all_df.values, metric='euclidean')
    dist_df = pd.DataFrame(distances, index=agg_post_pred_df.index, columns=post_gt_all_df.index)

    valid_conditions = [cond for cond in dist_df.index.get_level_values('condition').unique() if cond in dist_df.columns]
    
    self_distances_list = []
    for condition in valid_conditions:
        for method in dist_df.loc[condition].index:
            dist_val = dist_df.loc[(condition, method), condition]
            self_distances_list.append({
                'condition': condition,
                'method': method,
                'self_distance': dist_val
            })

    self_distances_df = pd.DataFrame(self_distances_list).set_index(['condition', 'method'])['self_distance']

    scores = {}
    methods = agg_post_pred_df.index.get_level_values('method').unique()
    
    for method in methods:

        method_dist_df = dist_df.xs(method, level='method').loc[valid_conditions]
        
        method_self_dist = self_distances_df.xs(method, level='method')

        scores[method] = (method_dist_df.gt(method_self_dist, axis=0)).sum(axis=1) / (method_dist_df.shape[1] - 1)
        
    scores_df = pd.DataFrame(scores)
    return scores_df

def calculate_average_perturbation(
    adata: anndata.AnnData,
    pert_col: str = 'condition',
    control_pert: str = 'ctrl'
) -> np.ndarray:

    pert_adata = adata[adata.obs[pert_col] != control_pert].copy()
    average_pert_profile = mean_nonzero(pert_adata.X, axis=0)
    return average_pert_profile


def pearson_delta_reference_metrics(X_true, X_pred, reference, top20_de_idxs, non_dropout_idxs):
    
    delta_true_allpert = (X_true - reference).flatten()
    delta_pred_allpert = (X_pred - reference).flatten()

    out = {
        'corr_all_allpert': pearsonr(delta_true_allpert, delta_pred_allpert)[0],
        'corr_20de_allpert': pearsonr(delta_true_allpert[top20_de_idxs], delta_pred_allpert[top20_de_idxs])[0],
    }
    if len(non_dropout_idxs) > 0:
        corr_nondropout = pearsonr(delta_true_allpert[non_dropout_idxs], delta_pred_allpert[non_dropout_idxs])[0]
        out['corr_nondropout_allpert'] = corr_nondropout
    return out

def pearson_delta_reference_metrics_top20_de(X_true, X_pred, reference, top20_de_idxs):

    delta_true_allpert = (X_true - reference).flatten()
    delta_pred_allpert = (X_pred - reference).flatten()

    return {
        'corr_20de_allpert': pearsonr(delta_true_allpert[top20_de_idxs], delta_pred_allpert[top20_de_idxs])[0],
    }

def pearson_delta_reference_metrics_non_dropout(X_true, X_pred, reference, non_dropout_idxs):

    delta_true_allpert = (X_true - reference).flatten()
    delta_pred_allpert = (X_pred - reference).flatten()

    return {
        'corr_nondropout_allpert': pearsonr(delta_true_allpert[non_dropout_idxs], delta_pred_allpert[non_dropout_idxs])[0],
    }

def pearson_delta_reference_metrics_non_dropout_top20de(X_true, X_pred, reference, non_dropout_idxs, top20_de_idxs):

    intersection_idxs = np.intersect1d(non_dropout_idxs, top20_de_idxs)
    
    if len(intersection_idxs) == 0:
        return {
            'corr_nondropout_top20de_allpert': np.nan,
        }
    
    delta_true_allpert = (X_true - reference).flatten()
    delta_pred_allpert = (X_pred - reference).flatten()

    return {
        'corr_nondropout_top20de_allpert': pearsonr(delta_true_allpert[intersection_idxs], delta_pred_allpert[intersection_idxs])[0],
    }
