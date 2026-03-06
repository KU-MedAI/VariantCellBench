import numpy as np
from typing import Dict, Union, Tuple
import pickle
import anndata as ad
import scanpy as sc
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def format_variant_dict_strict(pkl_path, mode='ALT'):
    """
    Structure: { ('TP53', 'A276V'): {'ALT': [vec...], 'DIFF': [vec...]} }
    """

    logger.info(f"Loading pickle from {pkl_path}...")
    with open(pkl_path, "rb") as f:
        raw_dict = pickle.load(f)
    
    formatted_dict = {}

    inferred_dim = None
    for inner_dict in raw_dict.values():
        if mode in inner_dict:
            # 첫 번째 유효한 벡터를 찾아 차원을 확인
            sample_vec = inner_dict[mode]
            
            # Numpy 변환 로직
            if hasattr(sample_vec, 'detach'): 
                sample_vec = sample_vec.detach().cpu().numpy()
            elif hasattr(sample_vec, 'numpy'): 
                sample_vec = sample_vec.numpy()
            
            sample_vec = np.array(sample_vec, dtype=np.float32).flatten()
            
            inferred_dim = sample_vec.shape[0]
            logger.info(f"Detected embedding dimension from data: {inferred_dim}")
            break
    
    if inferred_dim is None:
        raise ValueError(f"Could not infer dimension! Mode '{mode}' not found in any dictionary entry.")
    
    
    for key_tuple, inner_dict in raw_dict.items():
        if not isinstance(key_tuple, tuple) or len(key_tuple) != 2:
            logger.warning(f"Skipping invalid key format: {key_tuple}")
            continue
            
        gene, mut = key_tuple
        
        if "Ter" in mut:
            continue
            
        key_str = f"{gene}~{mut}"

        if mode not in inner_dict:
            continue

        vec_obj = inner_dict[mode]
                    
        if hasattr(vec_obj, 'detach'): 
            vec_obj = vec_obj.detach().cpu().numpy()
        elif hasattr(vec_obj, 'numpy'): 
            vec_obj = vec_obj.numpy()
        
        vec_np = np.array(vec_obj, dtype=np.float32).flatten()
        
        if vec_np.shape[0] != inferred_dim:
            if vec_np.size == inferred_dim:
                vec_np = vec_np.reshape(inferred_dim)
            else:
                logger.warning(f"Dimension mismatch for {key_str}: {vec_np.shape}, expected ({inferred_dim},)")
                continue
                
        formatted_dict[key_str] = vec_np

        if mut == 'REF':
            formatted_dict['ctrl'] = vec_np

    if 'ctrl' not in formatted_dict:
        logger.warning("WARNING: No 'REF' variant found in pickle. Creating fallback zero vector for 'ctrl'.")
        formatted_dict['ctrl'] = np.zeros(inferred_dim, dtype=np.float32)
    
    logger.info(f"Variant dictionary formatted. Total keys: {len(formatted_dict)}")
    
    return formatted_dict



