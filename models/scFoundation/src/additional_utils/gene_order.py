# #### Variant-seq gene order -> scFoundation gene order
# 데이터의 유전자 순서를 매핑하는 유틸 (핵심 베이스 코드)
# to_model_order(): perturb-seq을 모델 유전자 순서에 맞게 정렬
# to_data_order(): 정렬된 유전자를 원본 perturb-seq에 맞게 복원

from dataclasses import dataclass
from typing import Sequence, List, Optional, Tuple
import torch

# -------------------------
# 기존 GeneOrderMap 그대로 사용
# -------------------------
@dataclass
class GeneOrderMap:
    data_genes: List[str]
    model_genes: List[str]
    to_model_index: torch.LongTensor
    to_model_missing_mask: torch.BoolTensor
    to_data_index: torch.LongTensor
    to_data_missing_mask: torch.BoolTensor
    missing_in_data: List[str]
    missing_in_model: List[str]

def build_gene_order_map(
    data_genes: Sequence[str],
    model_genes: Sequence[str],
    device: Optional[torch.device] = None,
) -> GeneOrderMap:
    data_genes = list(data_genes)
    model_genes = list(model_genes)

    data_idx = {}
    for i, g in enumerate(data_genes):
        if g not in data_idx:
            data_idx[g] = i
    model_idx = {g: i for i, g in enumerate(model_genes)}

    to_model_idx, to_model_missing = [], []
    for g in model_genes:
        if g in data_idx:
            to_model_idx.append(data_idx[g]); to_model_missing.append(False)
        else:
            to_model_idx.append(0);          to_model_missing.append(True)

    to_data_idx, to_data_missing = [], []
    for g in data_genes:
        if g in model_idx:
            to_data_idx.append(model_idx[g]); to_data_missing.append(False)
        else:
            to_data_idx.append(0);            to_data_missing.append(True)

    to_model_index = torch.tensor(to_model_idx, dtype=torch.long, device=device)
    to_model_missing_mask = torch.tensor(to_model_missing, dtype=torch.bool, device=device)
    to_data_index = torch.tensor(to_data_idx, dtype=torch.long, device=device)
    to_data_missing_mask = torch.tensor(to_data_missing, dtype=torch.bool, device=device)

    missing_in_data = [g for g, m in zip(model_genes, to_model_missing) if m]
    missing_in_model = [g for g, m in zip(data_genes, to_data_missing) if m]

    return GeneOrderMap(
        data_genes=data_genes, model_genes=model_genes,
        to_model_index=to_model_index, to_model_missing_mask=to_model_missing_mask,
        to_data_index=to_data_index, to_data_missing_mask=to_data_missing_mask,
        missing_in_data=missing_in_data, missing_in_model=missing_in_model,
    )

# -------------------------
# 레이아웃 어댑터
# -------------------------
def _infer_B_from_total(total: int, G: int) -> int:
    assert total % G == 0, f"Total({total}) not divisible by G({G})"
    return total // G

def _to_BGF(x: torch.Tensor, layout: str, G: int) -> Tuple[torch.Tensor, int, int]:
    """
    입력 텐서를 (B, G, F) 표준형으로 변환.
    지원 레이아웃:
      - 'BG'        : (B, G)
      - 'BGF'       : (B, G, F)
      - 'B*G'       : (B*G,)
      - '1xB*G'     : (1, B*G)
      - 'GB'        : (G, B)
      - 'G'         : (G,)
      - 'GF'        : (G, F)     <-- 추가
    반환: (x_bgf, B, F)
    """
    if layout == "BG":
        B = x.size(0); assert x.size(1) == G
        return x.unsqueeze(-1), B, 1

    if layout == "BGF":
        B = x.size(0); assert x.size(1) == G
        F = x.size(2)
        return x, B, F

    if layout == "B*G":
        total = x.numel()
        B = _infer_B_from_total(total, G)
        return x.view(B, G, 1), B, 1

    if layout == "1xB*G":
        assert x.dim() == 2 and x.size(0) == 1
        total = x.size(1)
        B = _infer_B_from_total(total, G)
        return x.view(1, B*G).view(B, G, 1), B, 1

    if layout == "GB":
        assert x.size(0) == G
        B = x.size(1)
        return x.permute(1, 0).contiguous().unsqueeze(-1), B, 1

    if layout == "G":
        assert x.numel() == G
        return x.view(1, G, 1), 1, 1

    if layout == "GF":                      # <-- 추가
        assert x.size(0) == G
        F = x.size(1)
        return x.unsqueeze(0), 1, F

    raise ValueError(f"Unsupported input layout: {layout}")


def _from_BGF(x_bgf: torch.Tensor, layout: str, B: int, G: int, F: int) -> torch.Tensor:
    """
    (B, G, F) → 원하는 레이아웃으로 복구
    """
    assert x_bgf.shape == (B, G, F)

    if layout == "BG":
        assert F == 1
        return x_bgf.squeeze(-1)

    if layout == "BGF":
        return x_bgf

    if layout == "B*G":
        assert F == 1
        return x_bgf.view(B*G)

    if layout == "1xB*G":
        assert F == 1
        return x_bgf.view(1, B*G)

    if layout == "GB":
        assert F == 1
        return x_bgf.squeeze(-1).permute(1, 0).contiguous()

    if layout == "G":
        assert F == 1 and B == 1
        return x_bgf.view(G)

    if layout == "GF":                      # <-- 추가
        assert B == 1
        return x_bgf.squeeze(0)

    raise ValueError(f"Unsupported output layout: {layout}")


# -------------------------
# 재정렬 본체
# -------------------------
@torch.no_grad()
def to_model_order(
    x: torch.Tensor,
    gom: GeneOrderMap,
    in_layout: str,
    out_layout: str,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """
    데이터 유전자 순서(G_data) → 모델 유전자 순서(G_model)
    어떤 레이아웃이 와도 동작: in_layout/out_layout로 명시
    """
    # 데이터/모델 유전자 길이(Gd, Gm) 파악
    Gd = len(gom.data_genes)
    Gm = len(gom.model_genes)

    # 내부 표준형 (B, Gd, F) 로 변환.
    # B(배치 크기), F(특징 차원)
    x_bgf, B, F = _to_BGF(x, in_layout, Gd)         # (B, Gd, F)
    out = x_bgf.new_full((B, Gm, F), fill_value)    # (B, Gm, F)
    # 모델 유전자 길이 Gm로 만들고 fill_value로 초기화. 데이터에 없는 유전자는 fill_value값으로 패딩
    # 원 텐서의 속성(device/dtype) 을 자동 상속
    # out = torch.full((B,Gm,F), fill_value, dtype=x_bgf.dtype, device=x_bgf.device)


    # 모델에 필요한 유전자 중 실제 데이터에 존재하는 것만 골라서 복사
    present = ~gom.to_model_missing_mask            # (Gm,)
    if present.any():
        src_idx = gom.to_model_index[present]       # indices into Gd
        out[:, present, :] = x_bgf[:, src_idx, :]

    return _from_BGF(out, out_layout, B, Gm, F)

@torch.no_grad()
def to_data_order(
    x: torch.Tensor,
    gom: GeneOrderMap,
    in_layout: str,
    out_layout: str,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """
    모델 유전자 순서(G_model) → 데이터 유전자 순서(G_data)
    """
    Gd = len(gom.data_genes)
    Gm = len(gom.model_genes)

    x_bgf, B, F = _to_BGF(x, in_layout, Gm)         # (B, Gm, F)
    out = x_bgf.new_full((B, Gd, F), fill_value)    # (B, Gd, F)

    present = ~gom.to_data_missing_mask             # (Gd,)
    if present.any():
        src_idx = gom.to_data_index[present]        # indices into Gm
        out[:, present, :] = x_bgf[:, src_idx, :]

    return _from_BGF(out, out_layout, B, Gd, F)

@torch.no_grad()
def reorder_batch_x(
    x: torch.Tensor,
    gom: GeneOrderMap,
    fill_value: float = 0.0
) -> torch.Tensor:
    """
    PyG 입력 batch.x 변환: (B*Gd, 1) -> (B, Gm)
    """
    return to_model_order(
        x.squeeze(-1),   # (B*Gd,)
        gom,
        in_layout="B*G",
        out_layout="BG",
        fill_value=fill_value,
    )


@torch.no_grad()
def reorder_batch_y(
    y: torch.Tensor,
    gom: GeneOrderMap,
    fill_value: float = 0.0
) -> torch.Tensor:
    """
    PyG 출력 batch.y 변환: (B, Gd) -> (B, Gm)
    """
    return to_model_order(
        y,   # (B, Gd)
        gom,
        in_layout="BG",
        out_layout="BG",
        fill_value=fill_value,
    )

import torch
import numpy as np

def to_long_1d(x, device=None) -> torch.Tensor:
    """
    list/tuple/np.ndarray/torch.Tensor 모두 받아서 1D LongTensor로 반환.
    None이면 None 그대로 반환.
    """
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.to(dtype=torch.long, device=device).view(-1)
    # list/tuple/np.ndarray 처리
    return torch.as_tensor(x, dtype=torch.long, device=device).view(-1)


def build_data2model_pos(gom: GeneOrderMap) -> torch.LongTensor:
    """
    data 유전자 순서의 각 위치(0..Gd-1)에 대응하는 model 위치를 담은 벡터를 만든다.
    - 존재하지 않는(모델에 없는) data 위치는 -1로 표기.
    """
    Gd = len(gom.data_genes)
    d2m = torch.full((Gd,), -1, dtype=torch.long)  # 기본 -1
    # gom.to_model_index[k] = data에서의 위치, k = model 위치
    for k in range(len(gom.model_genes)):
        if not gom.to_model_missing_mask[k]:
            d_pos = int(gom.to_model_index[k])
            d2m[d_pos] = k
    return d2m


def map_index_from_data_to_model(idx_tensor: torch.Tensor,
                                 d2m: torch.LongTensor,
                                 drop_invalid: bool = True,
                                 unique: bool = False,
                                 sort: bool = False) -> torch.Tensor:
    """
    data 순서의 인덱스 텐서를 model 순서 인덱스로 변환.
    - idx_tensor: 1D long tensor (예: de_idx, de_non_dropout_idx, hvg_total_idx 등)
    """
    m = d2m[idx_tensor]  # 같은 길이, 일부는 -1
    if drop_invalid:
        m = m[m >= 0]
    if unique:
        m = torch.unique_consecutive(m) if sort is False else torch.unique(m)
    if sort:
        m, _ = torch.sort(m)
    return m

from torch.utils.data import Dataset
from copy import deepcopy

class ReorderedVariantSeqDataset(Dataset):
    """
    base_dataset의 각 sample(Data)을 모델 유전자 순서(Gm)로 미리 재정렬해 보관.
    x, y는 물론, de_idx / de_non_dropout_idx / hvg_total_idx 같은 'data 순서 인덱스'도
    model 순서 인덱스로 변환해 둔다.
    """
    def __init__(self, base_dataset, gom: GeneOrderMap, fill_value: float = 0.0, clone: bool = True):
        self.gom = gom
        self.fill_value = fill_value
        self.model_G = len(gom.model_genes)
        self.data2model_pos = build_data2model_pos(gom)  # (Gd,)-> model pos or -1

        self.data_list = []
        for i in range(len(base_dataset)):
            d = base_dataset[i]
            d = deepcopy(d) if clone else d

            # --- x: (Gd,1) 또는 (Gd,)
            x = d.x if isinstance(d.x, torch.Tensor) else torch.as_tensor(d.x)
            if x.dim() == 2 and x.size(1) == 1:
                x_gm = to_model_order(x.squeeze(-1), gom, "B*G", "BG", fill_value)#.unsqueeze(-1)  # (Gm,1)
            else:
                x_gm = to_model_order(x.view(-1), gom, "B*G", "BG", fill_value)#.unsqueeze(-1)     # (Gm,1)
            d.x = x_gm

            # --- y: (1,Gd) -> (1,Gm)
            if hasattr(d, "y") and d.y is not None:
                y = d.y if isinstance(d.y, torch.Tensor) else torch.as_tensor(d.y)
                assert y.dim() == 2 and y.size(0) == 1, f"y shape unexpected: {tuple(y.shape)}"
                d.y = to_model_order(y, gom, "BG", "BG", fill_value)  # (1, Gm)

            # --- 인덱스 필드들: list -> 1D long tensor 로 변환 후 매핑
            if hasattr(d, "de_idx") and d.de_idx is not None:
                de_idx_1d = to_long_1d(d.de_idx)  # <<< 변경
                d.de_idx = map_index_from_data_to_model(de_idx_1d, self.data2model_pos,
                                                        drop_invalid=True, unique=False, sort=False)

            if hasattr(d, "de_non_dropout_idx") and d.de_non_dropout_idx is not None:
                de_non_1d = to_long_1d(d.de_non_dropout_idx)  # <<< 변경
                d.de_non_dropout_idx = map_index_from_data_to_model(de_non_1d, self.data2model_pos,
                                                                    drop_invalid=True, unique=False, sort=False)

            if hasattr(d, "hvg_total_idx") and d.hvg_total_idx is not None:
                hvg_1d = to_long_1d(d.hvg_total_idx)  # <<< 변경
                mapped = map_index_from_data_to_model(hvg_1d, self.data2model_pos,
                                                      drop_invalid=True, unique=False, sort=False)
                d.hvg_total_idx = mapped if mapped.numel() == self.model_G \
                                  else torch.arange(self.model_G, dtype=torch.long)

            d.num_nodes = self.model_G
            self.data_list.append(d)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]