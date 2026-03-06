
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import anndata
from scipy.stats import pearsonr
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
    if issparse(data):
        data = data.toarray()
    
    result = np.mean(data, axis=axis)
    
    return result


def average_of_perturbation_centroids(
    adata: anndata.AnnData, 
    pert_col: str, 
    control_pert: str
) -> np.ndarray:
    """
    [수정됨] 대조군을 제외한 모든 교란 조건들의 평균 프로필(Reference)을 계산합니다.
    
    User Logic:
    1. Control을 제외한 데이터를 추립니다.
    2. 각 Condition별로 0을 제외한 평균(mean_nonzero)을 구합니다.
    3. 구해진 Condition별 평균들을 모아서, 다시 0을 제외한 평균(mean_nonzero)을 구합니다.
    """
    pert_adata = adata[adata.obs[pert_col] != control_pert].copy()

    if pert_adata.n_obs == 0:
        print(f"경고: '{control_pert}' 외에 다른 교란 조건이 없습니다. 0으로 채워진 벡터를 반환합니다.")
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
    """
    Calculates the centroid accuracies for all given perturbations and methods.
    (This is the user-provided function based on Euclidean distance and ranking.)

    Arguments:
    * agg_post_pred_df: Pandas dataframe with a MultiIndex ('condition', 'method').
    * post_gt_all_df: Pandas dataframe with the ground truth profiles.

    Returns a Pandas DataFrame with the centroid accuracies of each method.
    """
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
        
        # For each prediction, count how many other ground truths are farther away
        # than the correct ground truth.
        scores[method] = (method_dist_df.gt(method_self_dist, axis=0)).sum(axis=1) / (method_dist_df.shape[1] - 1)
        
    scores_df = pd.DataFrame(scores)
    return scores_df
def calculate_pairwise_mean_accuracies(pred_df, truth_df):
    """
    m:n 샘플 간의 모든 유클리드 거리를 구한 뒤 평균을 내어 순위 정확도를 계산합니다.
    (서브샘플링 없이 모든 샘플을 전수 검사하여 계산합니다.)
    
    Arguments:
    * pred_df: 예측값 DataFrame (MultiIndex: 'condition', 'method'). 그룹화(mean)되지 않은 개별 샘플.
    * truth_df: 실제값 DataFrame (Index: 'condition'). 그룹화되지 않은 개별 샘플.
    """
    print("INFO: Calculating Mean of Pairwise Distances (m:n) using ALL samples...")
    
    common_genes = truth_df.columns.intersection(pred_df.columns)
    pred_df = pred_df[common_genes]
    truth_df = truth_df[common_genes]

    pred_conditions_methods = pred_df.index.unique()
    truth_conditions = truth_df.index.unique()

    dist_dict = {}
    
    for pm_idx in pred_conditions_methods:
        pred_cond, method = pm_idx

        pred_samples = pred_df.loc[pm_idx].values
        if pred_samples.ndim == 1:
            pred_samples = pred_samples.reshape(1, -1)

        dist_dict[pm_idx] = {}
        
        for truth_cond in truth_conditions:
            truth_samples = truth_df.loc[truth_cond].values
            if truth_samples.ndim == 1:
                truth_samples = truth_samples.reshape(1, -1)

            pairwise_distances = cdist(pred_samples, truth_samples, metric='euclidean')
            dist_dict[pm_idx][truth_cond] = np.mean(pairwise_distances)

    dist_df = pd.DataFrame.from_dict(dist_dict, orient='index')
    dist_df.index = pd.MultiIndex.from_tuples(dist_df.index, names=['condition', 'method'])

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
    methods = dist_df.index.get_level_values('method').unique()
    
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
    """
    대조군을 제외한 모든 교란 조건들의 평균 프로필(pert_mean)을 계산합니다.
    0값은 제외하고 평균을 계산합니다.
    아래는 [260124] 버전
    systema에서 사용하는 pert mean 개념을 가져와서 수정함.
    모든 perturbed cell들의 평균 발현량으로 바꿈.
    """
    pert_adata = adata[adata.obs[pert_col] != control_pert].copy()
    average_pert_profile = mean_nonzero(pert_adata.X, axis=0)
    return average_pert_profile


def pearson_delta_reference_metrics(X_true, X_pred, reference, top20_de_idxs, non_dropout_idxs):
    """
    Compute PearsonΔ and PearsonΔ20 metrics using a specific reference

    Arguments:
    * X_true: ground-truth post-perturbation profile. Shape: (n_genes,)
    * X_pred: predicted post-perturbation profile. Shape: (n_genes,)
    * reference: reference. Shape: (n_genes,)
    * top20_de_idxs: indices of top 20 DE genes
    * non_dropout_idxs: indices of non-dropout genes

    Returns a dictionary with metrics: corr_all_allpert (PearsonΔ), corr_20de_allpert (PearsonΔ20), corr_nondropout_allpert (PearsonΔ non-dropout)
    """
    print(f"DEBUG: X_true shape: {X_true.shape}, X_pred shape: {X_pred.shape}, reference shape: {reference.shape}")
    print(f"DEBUG: X_true has NaN: {np.isnan(X_true).any()}, X_pred has NaN: {np.isnan(X_pred).any()}, reference has NaN: {np.isnan(reference).any()}")
    print(f"DEBUG: top20_de_idxs length: {len(top20_de_idxs)}, non_dropout_idxs length: {len(non_dropout_idxs)}")
    
    delta_true_allpert = (X_true - reference).flatten()
    delta_pred_allpert = (X_pred - reference).flatten()
    
    print(f"DEBUG: delta_true_allpert has NaN: {np.isnan(delta_true_allpert).any()}, delta_pred_allpert has NaN: {np.isnan(delta_pred_allpert).any()}")
    print(f"DEBUG: delta_true_allpert std: {np.std(delta_true_allpert)}, delta_pred_allpert std: {np.std(delta_pred_allpert)}")

    out = {
        'corr_all_allpert': pearsonr(delta_true_allpert, delta_pred_allpert)[0],
        'corr_20de_allpert': pearsonr(delta_true_allpert[top20_de_idxs], delta_pred_allpert[top20_de_idxs])[0],
    }
    if len(non_dropout_idxs) > 0:
        corr_nondropout = pearsonr(delta_true_allpert[non_dropout_idxs], delta_pred_allpert[non_dropout_idxs])[0]
        out['corr_nondropout_allpert'] = corr_nondropout
    return out

def pearson_delta_reference_metrics_top20_de(X_true, X_pred, reference, top20_de_idxs):
    """
    Compute PearsonΔ20 metrics using a specific reference (top 20 DE genes only)
    
    Arguments:
    * X_true: ground-truth post-perturbation profile. Shape: (n_genes,)
    * X_pred: predicted post-perturbation profile. Shape: (n_genes,)
    * reference: reference. Shape: (n_genes,)
    * top20_de_idxs: indices of top 20 DE genes

    Returns a dictionary with corr_20de_allpert metric
    """
    delta_true_allpert = (X_true - reference).flatten()
    delta_pred_allpert = (X_pred - reference).flatten()

    return {
        'corr_20de_allpert': pearsonr(delta_true_allpert[top20_de_idxs], delta_pred_allpert[top20_de_idxs])[0],
    }

def pearson_delta_reference_metrics_non_dropout(X_true, X_pred, reference, non_dropout_idxs):
    """
    Compute PearsonΔ metrics using a specific reference (non-dropout genes only)
    
    Arguments:
    * X_true: ground-truth post-perturbation profile. Shape: (n_genes,)
    * X_pred: predicted post-perturbation profile. Shape: (n_genes,)
    * reference: reference. Shape: (n_genes,)
    * non_dropout_idxs: indices of non-dropout genes

    Returns a dictionary with corr_nondropout_allpert metric
    """
    delta_true_allpert = (X_true - reference).flatten()
    delta_pred_allpert = (X_pred - reference).flatten()

    return {
        'corr_nondropout_allpert': pearsonr(delta_true_allpert[non_dropout_idxs], delta_pred_allpert[non_dropout_idxs])[0],
    }

def pearson_delta_reference_metrics_non_dropout_top20de(X_true, X_pred, reference, non_dropout_idxs, top20_de_idxs):
    """
    Compute PearsonΔ metrics using a specific reference (non-dropout genes + top20 DE intersection)
    
    Arguments:
    * X_true: ground-truth post-perturbation profile. Shape: (n_genes,)
    * X_pred: predicted post-perturbation profile. Shape: (n_genes,)
    * reference: reference. Shape: (n_genes,)
    * non_dropout_idxs: indices of non-dropout genes
    * top20_de_idxs: indices of top20 DE genes

    Returns a dictionary with corr_nondropout_top20de_allpert metric
    """
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