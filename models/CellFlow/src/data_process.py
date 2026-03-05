import anndata as ad
import scanpy as sc

from cellflow.preprocessing import centered_pca, project_pca, reconstruct_pca

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def unroll_and_align_data(h5ad_path, embedding_dict, pca_params=None, n_comps=50):
    logger.info(f"Processing {h5ad_path}...")
    adata_orig = sc.read_h5ad(h5ad_path)
    
    adata_ctrl = ad.AnnData(X=adata_orig.layers['x'].copy(), obs=adata_orig.obs.copy(), var=adata_orig.var.copy())
    adata_ctrl.obs['is_control'] = True
    adata_ctrl.obs['variant_input'] = 'ctrl' 


    adata_pert = ad.AnnData(X=adata_orig.layers['y'].copy(), obs=adata_orig.obs.copy(), var=adata_orig.var.copy())
    adata_pert.obs['is_control'] = False
    
    if 'variant' in adata_pert.obs.columns:
        variant_series = adata_pert.obs['variant'].astype(str)
        
        ter_mask = variant_series.str.contains("Ter")
        n_ter = ter_mask.sum()
        
        if n_ter > 0:
            logger.warning(f"Removing {n_ter} cells containing 'Ter' variants from {h5ad_path}")
            adata_pert = adata_pert[~ter_mask].copy()

        adata_pert.obs['variant_input'] = adata_pert.obs['variant'].astype(str)
    else:
        raise KeyError("'variant' column not found in obs.")

    adata_combined = ad.concat([adata_ctrl, adata_pert])
    adata_combined.obs_names_make_unique()
    
    if pca_params is None:
        obs_keys = set(adata_combined.obs['variant_input'].unique())
        emb_keys = set(embedding_dict.keys())
        missing = obs_keys - emb_keys
        if missing:
            logger.error(f"CRITICAL: {len(missing)} variants in obs are missing from embedding dict!")
            logger.error(f"Missing examples: {list(missing)[:5]}")
        else:
            logger.info("Validation Success: All obs variants have corresponding embeddings.")

    adata_combined.uns['variant_embeddings'] = embedding_dict

    if pca_params is None:
        logger.info("Computing Native PCA (Train)...")
        centered_pca(adata_combined, n_comps=n_comps)
        pca_mean = adata_combined.varm["X_mean"]
        pca_components = adata_combined.varm["PCs"]
        return adata_combined, (pca_mean, pca_components)
    else:
        logger.info("Projecting PCA (Val/Test)...")
        mean, comps = pca_params
        project_pca(adata_combined, ref_means=mean, ref_pcs=comps)
        return adata_combined, pca_params


def unroll_and_align_data_unseen(h5ad_path, embedding_dict, pca_params=None, n_comps=50):

    adata_orig = sc.read_h5ad(h5ad_path)  
    adata_combined = adata_orig.copy()
    
    if 'perturbation' in adata_combined.obs.columns:
        raw_series = adata_combined.obs['perturbation'].astype(str)
    elif 'condition' in adata_combined.obs.columns:
        raw_series = adata_combined.obs['condition'].astype(str)
    else:
        raise KeyError("Neither 'perturbation' nor 'condition' found in obs.")

    adata_combined.obs['variant_input'] = raw_series.str.split('+').str[0]
    adata_combined.obs['is_control'] = True 
    adata_combined.obs_names_make_unique()
    
    adata_combined.uns['variant_embeddings'] = embedding_dict
    
    if pca_params is not None:
        mean, comps = pca_params
        project_pca(adata_combined, ref_means=mean, ref_pcs=comps)
        return adata_combined, pca_params
    else:
        centered_pca(adata_combined, n_comps=n_comps)
        return adata_combined, (adata_combined.varm["X_mean"], adata_combined.varm["PCs"])