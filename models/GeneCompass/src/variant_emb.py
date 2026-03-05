import torch
import numpy as np
import anndata as ad
from torch.utils.data import Dataset
from typing import Dict, Tuple
import logging
from typing import Union

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_variant_key(s: str) -> Tuple[str, str]:
    """
    'TP53~C135R+ctrl' 또는 'TP53~C135R' -> ('TP53','C135R')
    'TP53~REF' -> ('TP53','REF')
    """
    s = str(s)
    # if s == 'ctrl':
    #     return 'unknown', 'REF' # 특정 유전자를 지정할 수 없으므로 'unknown'
    s = s.split('+', 1)[0]
    if '~' in s:
        gene, mut = s.split('~', 1)
        return gene, mut
    return s, 'REF'


class VariantEmbeddingGenerator:
    """
    이미 계산된(Pre-computed) variant embedding dictionary에서
    단순히 값을 조회하여 Tensor로 반환하는 클래스.
    """
    def __init__(self, variant_dict: Dict, variant_dim: int):
        self.variant_dict = variant_dict
        self.variant_dim = variant_dim
        
        # 'REF' 임베딩을 찾기 위해 유전자별 아무 변이 키 하나를 미리 저장해두면 유용할 수 있음 (옵션)
        self.gene_to_any_variant = {}
        for (g, m) in self.variant_dict.keys():
            if g not in self.gene_to_any_variant:
                self.gene_to_any_variant[g] = m

    def get_embedding(self, variant_str: str, embedding_type: str = 'alt') -> torch.Tensor:
        """
        variant_str: "TP53~M1A" 형태
        embedding_type: 'alt' (또는 'ref'), 'diff'
        """
        gene, mut = parse_variant_key(variant_str)
        
        # 0. 초기화 (Zero vector)
        final_vec = np.zeros(self.variant_dim, dtype=np.float32)        

        # 1. REF일 경우 처리
        if mut == 'REF':
            return torch.tensor(final_vec, dtype=torch.float32)

        # 2. 일반 Variant 조회
        data = self.variant_dict.get((gene, mut))
        if data is not None:
            # 대소문자 호환성 체크 ('ALT', 'alt', 'DIFF', 'diff' 모두 확인)
            emb_key_upper = embedding_type.upper()
            emb_key_lower = embedding_type.lower()
            
            if emb_key_upper in data:
                final_vec = data[emb_key_upper]
            elif emb_key_lower in data:
                final_vec = data[emb_key_lower]
            else:
                # 키가 있지만 해당 타입의 임베딩이 없는 경우
                logger.warning(f"Embedding type '{embedding_type}' not found for ({gene}, {mut}). Available keys: {list(data.keys())}. Using zeros.")
        else:
            logger.warning(f"Variant ({gene}, {mut}) not found in dictionary. Using zeros.")

        total_emb = torch.tensor(final_vec, dtype=torch.float32)

        return total_emb
    

'''
class VariantEmbeddingGenerator:
    """
    주어진 variant_dict로부터 PyTorch 텐서 형태의 variant embedding을 생성하는 클래스.
    """
    def __init__(self, variant_dict: Dict, variant_dim: int, pool: str = 'mean'):
        self.variant_dict = variant_dict
        self.variant_dim = variant_dim
        self.pool = pool
        if self.pool not in ['mean', 'max']:
            raise ValueError(f"Unsupported pooling method: {self.pool}")
        
    def _get_raw_embedding(self, gene: str, mut: str) -> Union[np.ndarray, None]:
        """주어진 (gene, mut) 키에 해당하는 원본 임베딩 배열(L, variant_dim)을 반환"""
        arr = self.variant_dict.get((gene, mut))
        if arr is None:
            return None
        
        arr_np = np.asarray(arr)
        # 일반적인 임베딩 형태 (1, L, dim)을 가정
        if arr_np.ndim != 3 or arr_np.shape[0] != 1 or arr_np.shape[-1] != self.variant_dim:
            logger.warning(f"Unexpected variant embedding shape for ({gene}, {mut}): {arr_np.shape}. Expected (1, L, {self.variant_dim}). Skipping.")
            return None
        
        # (1, L, dim) -> (L, dim) 형태로 반환
        return arr_np[0]
    
    def _pool_embedding(self, embedding_arr: np.ndarray) -> np.ndarray:
        """(L, variant_dim) 배열을 받아 풀링하여 (variant_dim,) 벡터로 만듦"""
        if self.pool == 'mean':
            return embedding_arr.mean(axis=0)
        elif self.pool == 'max':
            return embedding_arr.max(axis=0)

    def _get_pooled_vector(self, gene: str, mut: str) -> Union[np.ndarray, None]:
        """주어진 (gene, mut) 키에 해당하는 임베딩을 풀링하여 numpy 배열로 반환"""
        arr = self.variant_dict.get((gene, mut))
        if arr is None:
            return None
        
        arr_np = np.asarray(arr)
        if arr_np.ndim != 3 or arr_np.shape[-1] != self.variant_dim:
            logger.warning(f"Unexpected variant embedding shape for ({gene}, {mut}): {arr_np.shape}. Skipping.")
            return None
        
        if self.pool == 'mean':
            return arr_np.mean(axis=1)[0]
        elif self.pool == 'max':
            return arr_np.max(axis=1)[0]

    def get_embedding(self, variant_str: str, embedding_type: str = 'alt') -> torch.Tensor:
        """하나의 variant 문자열에 대한 최종 임베딩 텐서를 생성"""
        gene, mut = parse_variant_key(variant_str)
        final_vec = None

        if embedding_type == 'alt':
            if mut == 'REF' or gene == 'unknown':
                final_vec = np.zeros(self.variant_dim, dtype=np.float32)
            else:
                alt_vec = self._get_raw_embedding(gene, mut)
                if alt_vec is not None:
                    final_vec = self._pool_embedding(alt_vec)
                else: 
                    logger.warning(f"ALT embedding for ('{gene}', '{mut}') not found. Falling back to REF.")
                    ref_vec = self._get_raw_embedding(gene, 'REF')
                    if ref_vec is not None:
                        final_vec = self._pool_embedding(ref_vec)
                    else:
                        logger.error(f"Fallback REF embedding for ('{gene}', 'REF') also not found. Using zero vector.")
                        final_vec = np.zeros(self.variant_dim, dtype=np.float32)

        elif embedding_type == 'diff':

            key_with_max_length = max(self.variant_dict, key=lambda k: self.variant_dict[k].shape[1])
            max_length = self.variant_dict[key_with_max_length].shape[1]

            if mut == 'REF' or gene == 'unknown':
                final_vec = np.zeros(self.variant_dim, dtype=np.float32)
            else:
                ref_vec = self._get_raw_embedding(gene, 'REF')
                alt_vec = self._get_raw_embedding(gene, mut)
                if ref_vec is not None and alt_vec is not None:
                    norm_ref = np.zeros((max_length, self.variant_dim), dtype=np.float32)
                    norm_alt = np.zeros((max_length, self.variant_dim), dtype=np.float32)

                    # 2. 원본 데이터의 길이를 확인합니다.
                    len_ref = ref_vec.shape[0]
                    len_alt = alt_vec.shape[0]
                    
                    # 3. 복사할 길이를 결정합니다 (393보다 길면 잘라내기).
                    copy_len_ref = min(len_ref, max_length)
                    copy_len_alt = min(len_alt, max_length)

                    # 4. 표준 크기 배열에 원본 데이터를 복사합니다.
                    norm_ref[:copy_len_ref, :] = ref_vec[:copy_len_ref, :]
                    norm_alt[:copy_len_alt, :] = alt_vec[:copy_len_alt, :]

                    # 5. 이제 모양이 보장된 배열로 뺄셈을 수행합니다.
                    diff_embedding = norm_alt - norm_ref
                    final_vec = self._pool_embedding(diff_embedding)
                else:
                    if ref_vec is None:
                        logger.warning(f"REF embedding for ('{gene}', 'REF') not found for diff calculation. Using zero vector.")
                    if alt_vec is None:
                        logger.warning(f"ALT embedding for ('{gene}', '{mut}') not found for diff calculation. Using zero vector.")
                    final_vec = np.zeros(self.variant_dim, dtype=np.float32)
        
        else:
            raise ValueError(f"Unsupported embedding_type: {embedding_type}")
            
        return torch.tensor(final_vec, dtype=torch.float32)
    '''

def infer_variant_dim_from_pkl(variant_dict: dict) -> int:
    """
    새로운 구조의 variant 딕셔너리에서 임베딩 차원을 추론합니다.
    e.g., {('TP53','C135R'): {'alt': [..], 'diff': [..]}}
    """
    if not variant_dict:
        raise ValueError("Variant dictionary is empty. Cannot infer dimension.")

    # 딕셔너리의 첫 번째 값 (내부 딕셔너리)을 가져옵니다.
    first_inner_dict = next(iter(variant_dict.values()))
    if not first_inner_dict:
        raise ValueError("First variant entry has an empty inner dictionary.")

    # 내부 딕셔너리의 첫 번째 값 (임베딩 벡터)을 가져옵니다.
    first_embedding_vec = next(iter(first_inner_dict.values()))
    
    arr = np.asarray(first_embedding_vec)
    
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D vector for embedding, but got shape: {arr.shape}")
        
    # 벡터의 길이가 바로 차원(dimension)입니다.
    dimension = arr.shape[0]
    
    return int(dimension)
