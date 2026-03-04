# util2.py

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import anndata
from scipy.stats import pearsonr
from scipy.sparse import issparse
# non-zero 적용 버전
# def mean_nonzero(data, axis=0):
#     """
#     Compute mean excluding zero values.
    
#     Args:
#         data: numpy array or sparse matrix
#         axis: axis along which to compute the mean (0 for columns, 1 for rows)
        
#     Returns:
#         numpy array with mean values excluding zeros
#     """
#     if issparse(data):
#         data = data.toarray()
    
#     # Use numpy's masked array approach for efficiency
#     masked_data = np.ma.masked_where(data == 0, data)
    
#     # Compute mean along the specified axis, ignoring masked (zero) values
#     result = np.ma.mean(masked_data, axis=axis)
    
#     # Fill any invalid results (where all values were masked/zero) with 0
#     result = np.ma.filled(result, 0.0)
    
#     return result
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
    # 1. Control 제외
    pert_adata = adata[adata.obs[pert_col] != control_pert].copy()

    if pert_adata.n_obs == 0:
        print(f"경고: '{control_pert}' 외에 다른 교란 조건이 없습니다. 0으로 채워진 벡터를 반환합니다.")
        return np.zeros(adata.n_vars)

    pert_means = []
    unique_conditions = pert_adata.obs[pert_col].unique()

    # 2. 각 Condition별 평균 계산 (mean_nonzero 사용)
    for cond in unique_conditions:
        adata_cond = pert_adata[pert_adata.obs[pert_col] == cond]
        
        # 여기서 일반 .mean()이 아니라 mean_nonzero 사용
        pert_mean = mean_nonzero(adata_cond.X, axis=0)
        pert_means.append(pert_mean)
    
    pert_means = np.array(pert_means)
    
    # 3. 모인 평균들의 최종 평균 계산 (mean_nonzero 사용)
    # 일반 np.mean(pert_means, axis=0) 대신 사용
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
    # Ensure columns are in the same order
    common_genes = post_gt_all_df.columns.intersection(agg_post_pred_df.columns)
    agg_post_pred_df = agg_post_pred_df[common_genes]
    post_gt_all_df = post_gt_all_df[common_genes]

    distances = cdist(agg_post_pred_df.values, post_gt_all_df.values, metric='euclidean')
    dist_df = pd.DataFrame(distances, index=agg_post_pred_df.index, columns=post_gt_all_df.index)
    
    # Self distances (distance between a prediction and its corresponding ground truth)
    # Filter for conditions present in both prediction and truth
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
    
    # Calculate rank-based scores
    scores = {}
    methods = agg_post_pred_df.index.get_level_values('method').unique()
    
    for method in methods:
        # Filter distances for the current method
        method_dist_df = dist_df.xs(method, level='method').loc[valid_conditions]
        
        # Filter self-distances for the current method
        method_self_dist = self_distances_df.xs(method, level='method')
        
        # For each prediction, count how many other ground truths are farther away
        # than the correct ground truth.
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
    
    # # 각 교란 조건별로 0값 제외 평균 계산
    # condition_centroids = {}
    # for condition in pert_adata.obs[pert_col].unique():
    #     condition_data = pert_adata[pert_adata.obs[pert_col] == condition]
    #     # 0값 제외 평균 계산
    #     centroid = mean_nonzero(condition_data.X, axis=0)
    #     condition_centroids[condition] = centroid
    
    # # 모든 centroid들의 평균을 계산하여 최종 결과 도출
    # all_centroids = np.array(list(condition_centroids.values()))
    # average_pert_profile = mean_nonzero(all_centroids, axis=0)
    # return average_pert_profile
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
    # non-dropout 유전자와 top20 DE 유전자의 교집합 찾기
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

# --- [추가됨] MSE 관련 함수들 ---
# def mse_delta_reference_metrics(X_true, X_pred, reference, top20_de_idxs, non_dropout_idxs):
#     """
#     Compute MSE on Delta (Changes from reference).
#     """
#     delta_true = (X_true - reference).flatten()
#     delta_pred = (X_pred - reference).flatten()
    
#     # MSE 계산 함수 (내부용)
#     def calc_mse(arr1, arr2):
#         return np.mean((arr1 - arr2) ** 2)

#     out = {
#         'mse_all_allpert': calc_mse(delta_true, delta_pred),
#         'mse_20de_allpert': calc_mse(delta_true[top20_de_idxs], delta_pred[top20_de_idxs]),
#     }
    
#     if len(non_dropout_idxs) > 0:
#         out['mse_nondropout_allpert'] = calc_mse(delta_true[non_dropout_idxs], delta_pred[non_dropout_idxs])
        
#     return out

# def mse_delta_reference_metrics_non_dropout_top20de(X_true, X_pred, reference, non_dropout_idxs, top20_de_idxs):
#     intersection_idxs = np.intersect1d(non_dropout_idxs, top20_de_idxs)
#     if len(intersection_idxs) == 0:
#         return {'mse_nondropout_top20de_allpert': np.nan}
    
#     delta_true = (X_true - reference).flatten()
#     delta_pred = (X_pred - reference).flatten()
    
#     mse_val = np.mean((delta_true[intersection_idxs] - delta_pred[intersection_idxs]) ** 2)
#     return {'mse_nondropout_top20de_allpert': mse_val}