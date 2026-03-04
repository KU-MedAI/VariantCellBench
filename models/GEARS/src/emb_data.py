from __future__ import annotations

import zarr
import numpy as np
import requests
import mygene
import json
import os
import torch
import sys
import esm
import pickle
import time
from config import DATA_DIR, RAW_DATA_PATH, OUTPUT_DIR
from utils import *
from ProtT5 import *
import logging
logger = logging.getLogger(__name__)
from typing import List, Tuple, Union, Literal, Dict, Any, Optional
import os, subprocess, shutil
from pathlib import Path


### zarr 파일 관리
def zhas(root, path: str) -> bool:
    """Zarr 내 임의의 경로 존재 여부 (중첩 경로 안전 확인)."""
    try:
        _ = root[path]
        return True
    except KeyError:
        return False

def zread_text(root, path: str, encoding: str = "utf-8") -> str:
    """텍스트 dataset을 읽어 str로 반환 (uint8 1D 배열로 저장된 것을 복원)."""
    arr = root[path][...]
    return bytes(arr.tolist()).decode(encoding)

def zwrite_text(root, path: str, text: str, encoding: str = "utf-8", **create_kwargs):
    """
    텍스트를 uint8 1D dataset으로 저장. 기존이 있으면 덮어씀(overwrite).
    create_kwargs: chunks, compressor 등 전달 가능.
    """
    data = np.frombuffer(text.encode(encoding), dtype=np.uint8)
    # 상위 그룹 자동 생성
    parent = "/".join(path.split("/")[:-1])
    if parent:
        root.require_group(parent)
    # 기존 존재 시 삭제 후 생성
    try:
        del root[path]
    except Exception:
        pass
    ds = root.create_dataset(path, shape=data.shape, dtype=data.dtype, **create_kwargs)
    ds[...] = data
    ds.attrs["encoding"] = encoding
    return ds

# 경로 존재 여부 확인 및 디렉토리 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

# sys.path.append('/home/tech/ban/apocalypse/seqLM/esm/esm')
# pip install 'zarr>=2.18.7, <3.0.0'
# 임베딩 생성 함수 예시
def embed_GPNMSA(sequence):
    return np.random.rand(1024)





# prottrans_embeddings.py 같은 곳에 두면 좋음
import re
from functools import lru_cache
from typing import Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModel




# ============================
#   ANKH MODEL REGISTRY
# ============================
ANKH_MODELS: Dict[str, Dict[str, str]] = {
    # ------------------------------------------------------------
    # Ankh v1 계열 (Base / Large)
    # ------------------------------------------------------------
    # 특징:
    # - pip install sentencepiece
    # - Optimized Protein Language Model (OPLM)
    # - ProtTrans 대비 parameter 수 작은데 성능 동일/우수 주장
    # - embedding dimension: 상대적으로 작음 → downstream 사용 효율적
    # - Tokenization 방식: "list(sequence)" → is_split_into_words=True 필요
    # ------------------------------------------------------------
    "Ankh-Base": {
        "hf_name": "ElnaggarLab/ankh-base",
        # 예상 특징:
        # - Medium size
        # - 빠른 inference
        # - 일반 단백질 분석에 적합
    },
    "Ankh-Large": {
        "hf_name": "ElnaggarLab/ankh-large",
        # 예상 특징:
        # - 더 깊은 contextual representation
        # - Base 대비 약간 느리지만 여전히 가벼움
    },

    # ------------------------------------------------------------
    # Ankh v2 / v3 계열
    # ------------------------------------------------------------
    # 특징:
    # - architecture 개선 버전
    # - 더 크거나 더 최적화된 PLM
    # - embedding 성능 및 efficiency 개선
    # ------------------------------------------------------------
    "Ankh2-Large": {
        "hf_name": "ElnaggarLab/ankh2-large",
    },
    "Ankh3-Large": {
        "hf_name": "ElnaggarLab/ankh3-large",
    },
    "Ankh3-XL": {
        "hf_name": "ElnaggarLab/ankh3-xl",
        # 가장 큰 Ankh 모델
        # 성능 최상위 / 추론 cost는 가장 무거움
    },
}
@lru_cache(maxsize=8)
@lru_cache(maxsize=8)
def _load_ankh_encoder(model_id: str, device: str):
    """
    Ankh 계열은 encoder-only 표현을 쓰기 때문에
    T5EncoderModel을 사용해야 decoder_input_ids 에러를 피할 수 있음.
    """
    tokenizer = T5Tokenizer.from_pretrained(
        model_id,
        use_fast=False,   # 안정성을 위해 fast 끔 (필요 없으면 True로 바꿔도 됨)
    )
    encoder = T5EncoderModel.from_pretrained(model_id)
    encoder = encoder.to(device).eval()
    return tokenizer, encoder
def embed_Ankh(
    sequence: str,
    model_id: str,
    device: str | None = None,
    prefix: str = "[NLU]",   # 논문/모델 카드에서 권장하는 NLU prefix
) -> torch.Tensor:
    """
    Ankh / Ankh3 embedding 추출:
    - T5EncoderModel만 사용 (decoder는 사용하지 않음)
    - 입력 앞에 '[NLU]' prefix 붙이는 것이 권장됨.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer, encoder = _load_ankh_encoder(model_id, device)

    # 모델 카드 예시: "[NLU]" + sequence, is_split_into_words=False
    nlu_sequence = prefix + sequence

    encoded = tokenizer(
        nlu_sequence,
        add_special_tokens=True,
        return_tensors="pt",
        is_split_into_words=False,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = encoder(**encoded)

    # last_hidden_state: (1, L, D)
    hidden = outputs.last_hidden_state[:,1:-1,:]
    return hidden.cpu()






PROTTRANS_MODELS: Dict[str, Dict[str, str]] = {
    # ------------------------------------------------------------------------------------
    # ProtT5 계열 (Encoder-only, T5 architecture)
    # ------------------------------------------------------------------------------------
    # 특징
    # - 모든 ProtTrans 모델 중 **가장 강력하고 범용적인 residue-level 성능**
    # - T5 encoder-only 구조 → 양방향(Bi-directional) 문맥학습
    # - UniRef50(50% redundancy) 또는 BFD(>2.1B sequences) 데이터로 학습
    # - Half precision(16-bit) 가중치 제공 → VRAM 효율적 (8GB에서도 작동 가능)
    # - Amino acid per-residue embedding에서 downstream task 성능 가장 우수
    # ------------------------------------------------------------------------------------
    "ProtT5": {  # alias
        "hf_name": "Rostlab/prot_t5_xl_half_uniref50-enc",
    },
    "ProtT5-XL-U50": {
        "hf_name": "Rostlab/prot_t5_xl_half_uniref50-enc",  # UniRef50 기반
        # 장점: 가장 널리 사용되는 ProtTrans 모델, 고성능/안정적
        # 단점: ProtT5-XXL보다는 약간 작음 (XL)
        # VRAM: ~8–10GB (half precision)
    },
    "ProtT5-XL-BFD": {
        "hf_name": "Rostlab/prot_t5_xl_bfd",  # 2.1B BFD 기반
        # 장점: BFD 학습 → 더 다양한 서열 정보 학습, 일반화 소폭 향상
        # 단점: UniRef50 대비 특수 토큰 더 많이 등장 (필요시 주의)
    },
    # XXL 모델은 매우 큼
    "ProtT5-XXL-U50": {
        "hf_name": "Rostlab/prot_t5_xxl_uniref50"
    },
    "ProtT5-XXL-BFD": {
        "hf_name": "Rostlab/prot_t5_xxl_bfd"
    },

    # ------------------------------------------------------------------------------------
    # BERT 계열 (Transformer encoder, Masked LM)
    # ------------------------------------------------------------------------------------
    # 특징
    # - ProtTrans 초기 세대 모델
    # - BERT-base (~110M) 기반 구조 → ProtT5보다 가벼움
    # - UniRef100 또는 BFD 기반 pretraining
    # - residue-level에서는 ProtT5보다 약간 성능 낮음
    # - 그러나 inference 속도 빠르고 메모리 효율적
    # ------------------------------------------------------------------------------------
    "ProtBert-BFD": {
        "hf_name": "Rostlab/prot_bert_bfd",  # BFD 기반 학습
        # 장점: 경량, 빠름, VRAM 적게 필요 (~1.5–3GB)
        # 단점: ProtT5 대비 contextual 표현력 약함
        # 길이 제한: ~1024 토큰 (그 이상은 잘라서 입력 필요)
    },
    "ProtBert": {
        "hf_name": "Rostlab/prot_bert",  # UniRef100 기반
        # 장점: UniRef100로 학습 → redundancy 적음 → 정제된 표현
        # 단점: 다양한 remote homolog 학습은 BFD 버전이 더 유리
    },

    # ------------------------------------------------------------------------------------
    # Albert (ALBERT architecture)
    # ------------------------------------------------------------------------------------
    # 특징
    # - BERT보다 훨씬 parameter-efficient (layer sharing)
    # - 훨씬 가벼운 모델 → 속도 / 메모리 매우 좋음
    # - 그러나 bio-specific context 학습량은 ProtBert/BFD보다 약함
    # - 기능 예측 등 단순 task에는 충분하지만 구조/진화적 측면 성능은 낮은 편
    # ------------------------------------------------------------------------------------
    "ProtAlbert": {
        "hf_name": "Rostlab/prot_albert", # UniRef100 기반
        # 장점: 매우 가벼움 (수백 MB 수준), inference 빠름
        # 단점: deep contextual understanding 부족 → residue-level task 성능 하위권
        # pip install tiktoken
    },

    # ------------------------------------------------------------------------------------
    # XLNet (Permutation LM)
    # ------------------------------------------------------------------------------------
    # 특징 ⚠ ProtXLNet은 사용 불가 (HuggingFace 버전이 깨져있음)
    # - Permutation-based language model (BERT보다 강한 bidirectional context)
    # - 일반 NLP에서는 BERT보다 강력했음
    # - 하지만 protein domain에서는 ProtT5/BERT 대비 성능 열세
    # - 학습 sequence order 랜덤화가 단백질 서열에 최적화되지 않기 때문
    # ------------------------------------------------------------------------------------
    # "ProtXLNet": {
    #     "hf_name": "Rostlab/prot_xlnet",
    #     # 장점: 다양한 문맥 조합 학습
    #     # 단점: protein-specific downstream task에서 성능 낮음
    #     # VRAM: BERT보다 조금 더 요구됨
    # },

    # ------------------------------------------------------------------------------------
    # Electra (GAN-like replaced token detection)
    # ------------------------------------------------------------------------------------
    # 특징
    # - Masked LM이 아니라 "replaced token detection" 사용 → 매우 sample-efficient
    # - 작은 모델도 좋은 성능을 낼 수 있음
    # - Generator/Discriminator 쌍으로 구성
    # - 하지만 ProtT5/BERT 계열보다 context 표현력은 떨어짐
    # ------------------------------------------------------------------------------------
    "ProtElectra-Generator-BFD": {
        "hf_name": "Rostlab/prot_electra_generator_bfd",
        # 장점: 소형 generator, 빠름, BFD 기반
        # 단점: downstream per-residue 표현력 낮은 편
    },
    "ProtElectra-Discriminator-BFD": {
        "hf_name": "Rostlab/prot_electra_discriminator_bfd",
        # 장점: token replacement detection → efficient
        # 단점: 진화정보/long-range interaction 학습은 T5/BERT만큼 강하지 않음
    },
    

}



def _prep_protein_for_prottrans(sequence: str) -> str:
    """
    ProtTrans 권장 전처리:
    - U/Z/O/B → X 치환
    - 아미노산 사이에 공백 삽입 ("A E T C ...")
    """
    seq = sequence.strip().upper()
    seq = re.sub(r"[UZOB]", "X", seq)
    return " ".join(list(seq))


@lru_cache(maxsize=16)
def _load_prottrans_model(
    embedding_model_name: str,
    device: str | None = None,
):
    """
    ProtTrans 모델 로더:
      - ProtT5 계열 → T5EncoderModel (encoder-only)
      - 그 외 (ProtBert/ProtAlbert/ProtElectra 등) → AutoModel
    """
    if embedding_model_name not in PROTTRANS_MODELS:
        raise ValueError(f"지원하지 않는 ProtTrans 모델명: {embedding_model_name}")

    hf_name = PROTTRANS_MODELS[embedding_model_name]["hf_name"]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- ProtT5 계열인지 판별 ---
    is_prot_t5 = embedding_model_name.startswith("ProtT5")

    if is_prot_t5:
        # ✅ T5 encoder-only 사용 (decoder_input_ids 문제 해결)
        tokenizer = T5Tokenizer.from_pretrained(
            hf_name,
            use_fast=False,       # SentencePiece 기반, slow 토크나이저 사용
        )
        model = T5EncoderModel.from_pretrained(hf_name)
        model = model.to(device).eval()

    else:
        # ✅ 나머지 ProtBert/Albert/Electra는 기존 Auto 계열 사용
        tokenizer = AutoTokenizer.from_pretrained(
            hf_name,
            do_lower_case=False,
            use_fast=False,      # ProtAlbert bug 피하기 위해 fast 끔
        )
        model = AutoModel.from_pretrained(hf_name)
        model = model.to(device).eval()

    return tokenizer, model, device

def _strip_special_tokens(
    hidden: torch.Tensor,       # (1, T, D)
    input_ids: torch.Tensor,    # (1, T)
    tokenizer,
) -> torch.Tensor:
    """
    ProtTrans 계열 공통: CLS/SEP/PAD/EOS 등 special token을 제거하고
    순수 residue 토큰만 남긴다.
    반환: (1, L_residue, D)
    """
    # 배치는 지금 1개만 쓴다고 가정
    assert hidden.size(0) == 1 and input_ids.size(0) == 1
    ids = input_ids[0]       # (T,)
    reps = hidden[0]         # (T, D)

    # 토크나이저가 알고 있는 special token id 수집
    special_ids = set()
    for attr in ["pad_token_id", "cls_token_id", "sep_token_id",
                 "bos_token_id", "eos_token_id"]:
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            special_ids.add(tid)

    if not special_ids:
        # 혹시 모를 경우: 아무 special token 없다고 보고 전체 사용
        return hidden

    mask = torch.ones_like(ids, dtype=torch.bool)
    for tid in special_ids:
        mask &= (ids != tid)

    # True인 위치만 남김 = 순수 amino acid 토큰
    idx = mask.nonzero(as_tuple=True)[0]      # (L_residue,)
    residue_reps = reps[idx]                  # (L_residue, D)

    return residue_reps.unsqueeze(0)          # (1, L_residue, D)

def embed_ProtTrans(
    sequence: str,
    *,
    embedding_model_name: str = "ProtT5",
    per_residue: bool = True,
    device: str | None = None,
) -> torch.Tensor:
    tokenizer, model, device = _load_prottrans_model(
        embedding_model_name,
        device=device,
    )

    prepped = _prep_protein_for_prottrans(sequence)

    encoded = tokenizer(
        prepped,
        add_special_tokens=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    hidden = outputs.last_hidden_state        # (1, T, D)

    # ✅ 여기서 모델에 따라 CLS/SEP/PAD/EOS 제거
    residue_hidden = _strip_special_tokens(hidden, input_ids, tokenizer)  # (1, L, D)

    if not per_residue:
        return residue_hidden.mean(dim=1, keepdim=True)   # (1, D)

    return residue_hidden                                  # (1, L, D)






# ===== xTrimoPGLM / ProteinGLM embedding =====
from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForMaskedLM

# 모델/토크나이저를 한 번만 로드해서 재사용 (매 호출마다 로드 방지)
@lru_cache(maxsize=4)
def _load_xtrimo_mlm(model_id: str, device: str = None, dtype: torch.dtype = torch.bfloat16):
    """
    model_id 예시:
      - "biomap-research/xtrimopglm-1b-mlm"
      - "biomap-research/proteinglm-1b-mlm"
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=True,
    )
    model = AutoModelForMaskedLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    return tokenizer, model, device


def embed_xTrimoPGLM(
    sequence: str,
    *,
    model_id: str = "biomap-research/xtrimopglm-1b-mlm",
    device: str | None = None,
) -> torch.Tensor:
    """
    xTrimoPGLM / ProteinGLM 모델에서 residue-level embedding을 뽑는 함수.
    반환 shape: [1, L, D] (아미노산 길이 L, 마지막 layer 기준)

    sequence: 아미노산 서열 (예: "MEEPQSDPSV...")
    model_id: HuggingFace 모델 ID 또는 로컬 경로
    """
    tokenizer, model, default_device = _load_xtrimo_mlm(
        model_id,
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=torch.bfloat16,
    )
    device = default_device

    # HF README 예시와 동일하게 special token 포함해서 토크나이징 :contentReference[oaicite:0]{index=0}
    encoded = tokenizer(
        sequence,
        add_special_tokens=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attn_mask = encoded["attention_mask"].to(device)

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            return_last_hidden_state=True
        )

    hidden_states = outputs.hidden_states[:-1, 0].unsqueeze(0)  # tuple
    # hidden_states = hidden_states.permute(1, 0, 2)  
    # print(f"[DEBUG] num_layers = {len(hidden_states)}")
    # for i, h in enumerate(hidden_states):
    #     print(f"[DEBUG] layer {i}: shape = {tuple(h.shape)}")

    return hidden_states

# ############################################################
# ESM-MSA-1b: A3M에서 "쿼리 행" 임베딩 추출
# ############################################################
def _a3m_to_name_seq_list(a3m_text: str) -> List[Tuple[str, str]]:
    names, seqs = [], []
    cur_name, buf = None, []
    for raw in a3m_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if cur_name is not None:
                s = "".join(buf).replace(".", "-")
                s = "".join(c for c in s if (c == "-" or c.isupper()))
                seqs.append(s)
            cur_name = line[1:].strip()
            names.append(cur_name)
            buf = []
        else:
            buf.append(line)
    if cur_name is not None:
        s = "".join(buf).replace(".", "-")
        s = "".join(c for c in s if (c == "-" or c.isupper()))
        seqs.append(s)
    return list(zip(names, seqs))

def embed_msa_from_a3m_text(
    a3m_text: str,
    layer: int = 12,
    top_n: int | None = None,
    truncate_to: int | None = None,
    device: str | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, batch_converter = _load_esm_msa_t12(device)

    # --- 1) A3M 파싱 ---
    msa = _a3m_to_name_seq_list(a3m_text)
    if not msa:
        raise ValueError("빈 MSA입니다.")
    names, seqs = zip(*msa)
    names, seqs = list(names), list(seqs)

    # --- 2) 선택적 trim ---
    if top_n is not None:
        names, seqs = names[:top_n], seqs[:top_n]
    if truncate_to is not None:
        seqs = [s[:truncate_to] for s in seqs]

    # --- 3) sanity check (정렬 길이 동일해야) ---
    Lset = {len(s) for s in seqs}
    if len(Lset) != 1:
        raise ValueError(f"MSA 길이 불일치: {Lset}")

    # --- 4) batch builder ---
    batch = list(zip(names, seqs))
    _, _, tokens = batch_converter(batch)
    tokens = tokens.to(device)

    # --- 5) 모델 forward ---
    with torch.no_grad():
        out = model(tokens, repr_layers=[layer], return_contacts=False)

    # 반환: (1, N, L, C)
    rep = out["representations"][layer][..., 1:, :]  # BOS 제거
    return rep.cpu()



# ====== 공통 헬퍼 ======
def _prep_names_seqs(a3m_text: str, top_n: Optional[int], truncate_to: Optional[int]):
    msa = _a3m_to_name_seq_list(a3m_text)
    if not msa:
        raise ValueError("빈 MSA입니다.")
    names, seqs = map(list, zip(*msa))
    if top_n is not None and top_n > 0:
        names, seqs = names[:top_n], seqs[:top_n]
    if truncate_to is not None:
        seqs = [s[:truncate_to] for s in seqs]
    Lset = {len(s) for s in seqs}
    if len(Lset) != 1:
        raise ValueError(f"정렬 길이 불일치: {sorted(Lset)}")
    return names, seqs  # lists

def _safe_tokens_msa(names, seqs, alphabet, batch_converter):
    """여러 포맷을 시도하고 실패하면 수동 토크나이즈로 백업."""
    L, N = len(seqs[0]), len(seqs)
    # 1) 다양한 포맷 시도
    for data in [
        [("msa", seqs)],                         # A: 문자열 리스트
        [("msa", list(zip(names, seqs)))],       # B: (이름, 문자열) 리스트
        [list(zip(names, seqs))],                # C: 리스트만
    ]:
        try:
            _, _, toks = batch_converter(data)   # 기대: (1, N, L+1)
            if toks.dim() == 3 and toks.size(-1) == L + 1:
                return toks
        except Exception:
            pass
    # 2) 수동 토크나이즈(백업)
    tok_to_idx = alphabet.tok_to_idx
    pad_idx = alphabet.padding_idx
    cls_idx = tok_to_idx.get("<cls>", 0)
    unk_idx = tok_to_idx.get("<unk>", pad_idx)
    def _map(c): return tok_to_idx.get(c, unk_idx)
    toks = torch.full((1, N, L + 1), pad_idx, dtype=torch.long)
    toks[:, :, 0] = cls_idx
    for i, s in enumerate(seqs):
        toks[0, i, 1:] = torch.tensor([_map(c) for c in s], dtype=torch.long)
    return toks

def _ungapped_to_col(query_aln: str):
    m, u = {}, 0
    for j, ch in enumerate(query_aln):
        if ch != "-":
            u += 1; m[u] = j
    return m

# ====== 임베딩 추출 ======
@lru_cache(maxsize=1)
def _load_esm2_t33(device: str):
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    return model, batch_converter


def embed_esm2_t33_650M_UR50D(sequence: str, layer=33, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, batch_converter = _load_esm2_t33(device)
    data = [("protein1", sequence)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        out = model(batch_tokens, repr_layers=[layer], return_contacts=False)
    
    return out["representations"][layer].cpu()[:, 1:-1, :]




### MSA-transformer ###
# ====== 변이 점수(LLR) ======
# def msa_variant_llr(
#     a3m_text: str,
#     variants: List[Tuple[str, int, str]],  # [(refAA, pos(1-based, ungapped), mutAA)]
#     top_n: Optional[int] = None,
#     truncate_to: Optional[int] = None,
#     device: Optional[str] = None,
# ) -> float:
#     """
#     masked-marginal LLR 합계 반환 (쿼리/마스터 행만 마스크)
#     """
#     if device is None:
#         device = "cuda" if torch.cuda.is_available() else "cpu"

#     names, seqs = _prep_names_seqs(a3m_text, top_n, truncate_to)
#     q = seqs[0]
#     pos2col = _ungapped_to_col(q)

#     model, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
#     model.eval().to(device)
#     batch_converter = alphabet.get_batch_converter()

#     tokens = _safe_tokens_msa(names, seqs, alphabet, batch_converter).to(device)
#     BOS = 0
#     idx = alphabet.tok_to_idx
#     mask_ix = alphabet.mask_idx

#     llr = 0.0
#     for ref, pos, mut in variants:
#         col = pos2col.get(pos)
#         if col is None or q[col] == "-":
#             try:
#                 print(q[col])
#             except:
#                 print(col)
#             raise ValueError(f"위치 {pos}가 정렬에 없거나 gap입니다.")
#         if q[col] != ref:
#             raise ValueError(f"참조 불일치: 정렬상 {pos}='{q[col]}', 입력 ref='{ref}'")

#         toks = tokens.clone()
#         toks[0, 0, BOS + 1 + col] = mask_ix  # 쿼리 행만 마스크

#         with torch.no_grad():
#             out = model(toks, return_contacts=False)
#             log_probs = out["logits"].log_softmax(-1)  # (1, N, L+1, V)
#         lp = log_probs[0, 0, BOS + 1 + col]
#         wt_i, mut_i = idx.get(ref), idx.get(mut)
#         if wt_i is None or mut_i is None:
#             raise ValueError(f"AA 코드 확인(ref={ref}, mut={mut})")
#         llr += float(lp[mut_i] - lp[wt_i])

#     return llr
def msa_variant_llr(
    a3m_text: str,
    variants: List[Tuple[str, int, str]],  # [(refAA, pos(1-based ungapped), mutAA)]
    top_n: Optional[int] = None,
    truncate_to: Optional[int] = None,     # 주어지면 우선 사용 (단, max_len 한도 내)
    device: Optional[str] = 'cuda',
    max_len: int = 1024,                   # ESM-MSA 허용 열 수 한도
    pad_margin: int = 16,                  # 창 양끝 여유 (모델 문맥 안정)
    model: Object = None,
    alphabet: Object = None
) -> float:
    """
    masked-marginal LLR 합계 반환 (쿼리/마스터 행만 마스크).
    - 변이 열을 반드시 포함하도록 MSA를 동적 슬라이싱
    - 한 창에 모두 못 넣으면 변이별로 나눠 계산 후 합산
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    names, seqs = _prep_names_seqs(a3m_text, top_n=top_n, truncate_to=None)  # full 길이 유지
    q_full = seqs[0]
    pos2col_full = _ungapped_to_col(q_full)
    L_full = len(q_full)

    # 유틸: 열 범위 슬라이스
    def _slice_by_cols(names_, seqs_, start_col: int, end_col: int):
        # [start_col, end_col] inclusive
        names_new = list(names_)
        seqs_new = [s[start_col:end_col+1] for s in seqs_]
        return names_new, seqs_new

    # 유틸: 창 결정 (여러 변이 커버)
    def _decide_window(cols: List[int]) -> Tuple[int,int]:
        lo = max(0, min(cols) - pad_margin)
        hi = min(L_full - 1, max(cols) + pad_margin)
        if hi - lo + 1 <= eff_max_len:
            return lo, hi
        # 너무 길면 최소 창으로 축소
        span = min(eff_max_len, L_full)
        center = (min(cols) + max(cols)) // 2
        lo = max(0, center - span//2)
        hi = min(L_full - 1, lo + span - 1)
        return lo, hi

    # 허용 최대 길이 산정
    eff_max_len = max_len if truncate_to is None else min(max_len, int(truncate_to))

    # 전체 변이를 full 열좌표로 변환/검증
    cols_full = []
    for ref, pos, mut in variants:
        col = pos2col_full.get(pos)
        if col is None or q_full[col] == "-":
            raise ValueError(f"위치 {pos}가 정렬에 없거나 gap입니다. (truncate로 잘렸던 케이스면 auto-window가 실패)")
        if q_full[col] != ref:
            raise ValueError(f"참조 불일치: 정렬상 {pos}='{q_full[col]}', 입력 ref='{ref}'")
        cols_full.append(col)

    # 가능한 한 한 창에 모두 넣되, 불가하면 변이별로 분할
    llr_sum = 0.0

    def _score_on_window(cols_local: List[int], win_lo: int, win_hi: int):
        names_w, seqs_w = _slice_by_cols(names, seqs, win_lo, win_hi)
        q_w = seqs_w[0]
        # 로컬 col 재계산
        cols_local2 = [c - win_lo for c in cols_local]

        # ESM-MSA 실행
        model, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
        model.eval().to(device)
        batch_converter = alphabet.get_batch_converter()
        tokens = _safe_tokens_msa(names_w, seqs_w, alphabet, batch_converter).to(device)

        BOS = 0
        idx = alphabet.tok_to_idx
        mask_ix = alphabet.mask_idx

        llr_local = 0.0
        for (ref, pos, mut), col_local in zip(variants, cols_local2):
            # 여기서 variants의 순서와 cols_local이 1:1이라는 보장은 없음 → 매핑 필요
            # 안전하게 현재 창에 들어오는 변이만 골라서 처리
            if not (0 <= col_local < len(q_w)):
                continue
            if q_w[col_local] == "-":
                raise ValueError(f"[window] pos {pos}가 로컬 정렬에서 gap입니다.")
            if q_w[col_local] != ref:
                raise ValueError(f"[window] 참조 불일치: pos {pos} 정렬='{q_w[col_local]}', ref='{ref}'")

            toks = tokens.clone()
            toks[0, 0, BOS + 1 + col_local] = mask_ix  # 쿼리 행만 마스크

            with torch.no_grad():
                out = model(toks, return_contacts=False)
                log_probs = out["logits"].log_softmax(-1)  # (1, N, L+1, V)
            lp = log_probs[0, 0, BOS + 1 + col_local]
            wt_i, mut_i = idx.get(ref), idx.get(mut)
            if wt_i is None or mut_i is None:
                raise ValueError(f"AA 코드 확인(ref={ref}, mut={mut})")
            llr_local += float(lp[mut_i] - lp[wt_i])

        return llr_local

    # 먼저 모든 변이를 한 창에 넣을 수 있으면 그렇게
    win_lo, win_hi = _decide_window(cols_full)
    if win_hi - win_lo + 1 <= eff_max_len:
        # 창 안에 들어오는 변이만 골라 계산
        cols_in = [c for c in cols_full if win_lo <= c <= win_hi]
        var_in = [v for v, c in zip(variants, cols_full) if win_lo <= c <= win_hi]
        # 로컬화된 리스트 전달
        llr_sum += _score_on_window(cols_in, win_lo, win_hi)
        # 창 밖 변이가 남아있으면 개별 창으로 처리
        remain = [(v, c) for v, c in zip(variants, cols_full) if not (win_lo <= c <= win_hi)]
        for v, c in remain:
            lo = max(0, c - eff_max_len//2)
            hi = min(L_full - 1, lo + eff_max_len - 1)
            llr_sum += _score_on_window([c], lo, hi)
    else:
        # 처음부터 너무 넓음 → 변이별로 개별 창 처리
        for c in cols_full:
            lo = max(0, c - eff_max_len//2)
            hi = min(L_full - 1, lo + eff_max_len - 1)
            llr_sum += _score_on_window([c], lo, hi)

    return llr_sum



def _read_a3m(filepath: str) -> List[Tuple[str, str]]:
    """
    A3M에서 (name, aligned_seq) 리스트를 파싱.
    - 소문자(삽입) 제거, 대문자/갭('-')만 유지
    - '.' 갭은 '-'로 통일
    """
    names, seqs = [], []
    name, buf = None, []
    with open(filepath) as f:
        for line in f:
            if line.startswith('>'):
                if name is not None:
                    s = ''.join(buf)
                    s = s.replace('.', '-')               # 갭 통일
                    s = ''.join(c for c in s if (c == '-' or c.isupper()))
                    seqs.append(s)
                name = line.strip()[1:]
                names.append(name)
                buf = []
            else:
                buf.append(line.strip())
        if name is not None:
            s = ''.join(buf)
            s = s.replace('.', '-')
            s = ''.join(c for c in s if (c == '-' or c.isupper()))
            seqs.append(s)
    return list(zip(names, seqs))


def _ensure_aligned(seqs: List[str]) -> int:
    """모든 서열 길이가 동일한지 확인하고 길이를 반환."""
    lengths = {len(s) for s in seqs}
    if len(lengths) != 1:
        raise ValueError(
            f"MSA가 정렬되지 않았습니다(길이 다양): {sorted(lengths)}. "
            "A3M의 삽입(소문자)을 제거했는지 확인하세요."
        )
    return lengths.pop()


def _truncate_columns(seqs: List[str], trunc_len: Optional[int]) -> List[str]:
    """열 기준으로 앞쪽부터 trunc_len만 남김(BOS 이전의 정렬 열 기준)."""
    if trunc_len is None:
        return seqs
    return [s[:trunc_len] for s in seqs]

@lru_cache(maxsize=1)
def _load_esm_msa_t12(device: str):
    model, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    return model, batch_converter

def embed_msa_esm_msa1b(
    msa_input: Union[str, List[Tuple[str, str]], List[str]],
    layer: int = 12,
    device: Optional[Union[str, torch.device]] = None,
    top_n: Optional[int] = None,
    truncate_to: Optional[int] = None,
    aggregate: Optional[Literal["mean_seq", "mean_col", "mean_both"]] = None,
    return_contacts: bool = False,
) -> Dict[str, Any]:
    """
    MSA-Transformer(ESM-MSA-1b)로 MSA 임베딩(및 contact)을 추출.

    Args:
        msa_input: A3M 경로(str) 또는 (name, seq) 리스트 또는 정렬된 seq 리스트.
                   모든 seq는 같은 길이여야 하며 갭은 '-' 사용.
        layer: 반환할 representation layer (ESM-MSA-1b는 12층 모델이므로 기본=12)
        device: 'cuda' | 'cpu' (기본 자동 선택)
        top_n: 상위 N개 서열만 사용 (메모리 절약)
        truncate_to: 정렬 열 길이를 앞쪽부터 자를 최대 길이(L) (열 기반 트렁케이션)
        aggregate: 
            - None: (1, N, L, C) 그대로 반환
            - "mean_seq": 시퀀스 차원 평균 → (1, L, C)
            - "mean_col": 열(L) 차원 평균 → (1, N, C)
            - "mean_both": 두 차원 평균 → (1, C)
        return_contacts: True면 contact map도 함께 반환 ((1, L, L))

    Returns:
        {
            "representations": torch.Tensor  # shape 위 설명 참조 (CPU 텐서)
            "contacts": Optional[torch.Tensor],  # (1, L, L) or None
            "tokens": torch.Tensor,  # (1, N, L+1)
            "labels": List[str],     # 입력 라벨(배치명 1개)
            "n_seq": int,
            "L": int
        }
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) 입력 정리
    if isinstance(msa_input, str):
        # A3M 경로
        msa = _read_a3m(msa_input)
    else:
        # 리스트 형태
        if len(msa_input) == 0:
            raise ValueError("msa_input이 비어 있습니다.")
        if isinstance(msa_input[0], tuple):
            msa = [(n, s) for (n, s) in msa_input]
        else:
            # 이름이 없는 경우 인덱스로 이름 생성
            msa = [(f"seq{i+1}", s) for i, s in enumerate(msa_input)]

        # A3M이 아닐 수도 있으니, 여기선 삽입 소문자는 없다고 가정(이미 정렬됨).
        # 그래도 안전하게 '.' 갭을 '-'로 통일
        msa = [(n, s.replace('.', '-')) for (n, s) in msa]

    # 2) 길이 검증 및 옵션 처리(top_n, truncate_to)
    names, seqs = zip(*msa)
    if top_n is not None and top_n > 0:
        names, seqs = names[:top_n], seqs[:top_n]
    L0 = _ensure_aligned(list(seqs))
    seqs = _truncate_columns(list(seqs), truncate_to)
    L = len(seqs[0])
    if L == 0:
        raise ValueError("truncate_to가 너무 작아 L=0이 되었습니다.")

    # 3) 모델/알파벳/배치 컨버터
    model, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    # 4) 배치 변환 (B=1)
    batch = [("msa", list(seqs))]
    labels, strs, tokens = batch_converter(batch)
    tokens = tokens.to(device)

    # 5) 추론
    with torch.no_grad():
        out = model(tokens, repr_layers=[layer], return_contacts=return_contacts)

    rep = out["representations"][layer]   # (1, N, L+1, C)인 경우도 있음 -> BOS 제거 필요
    # ESM MSA Transformer는 BOS(<cls>)를 앞에 붙입니다. 열(L) 축에서 첫 토큰 제거.
    # rep shape 표준화: (1, N, L, C)
    rep = rep[..., 1:, :]

    # 6) 집계 옵션
    # 현재 rep: (1, N, L, C)
    if aggregate is None:
        rep_out = rep
    elif aggregate == "mean_seq":
        rep_out = rep.mean(dim=1, keepdim=True)  # (1, 1, L, C) → 일관성 위해 (1, L, C)로 변환할 수도 있음
        rep_out = rep_out.squeeze(1)             # (1, L, C)
    elif aggregate == "mean_col":
        rep_out = rep.mean(dim=2, keepdim=True)  # (1, N, 1, C) -> (1, N, C)
        rep_out = rep_out.squeeze(2)
    elif aggregate == "mean_both":
        rep_out = rep.mean(dim=(1, 2), keepdim=True)  # (1, 1, 1, C) -> (1, C)
        rep_out = rep_out.squeeze(1).squeeze(1)
    else:
        raise ValueError(f"알 수 없는 aggregate 옵션: {aggregate}")

    contacts = out["contacts"].cpu() if return_contacts else None
    if isinstance(contacts, torch.Tensor):
        # contacts는 (1, L, L) 형태. 이미 BOS 제외 상태로 반환됩니다.
        pass

    return {
        "representations": rep_out.cpu(),
        "contacts": contacts,
        "tokens": tokens.cpu(),
        "labels": labels,
        "n_seq": rep.shape[1] if rep.dim() == 4 else None,
        "L": rep.shape[2] if rep.dim() == 4 else (rep.shape[1] if aggregate == "mean_seq" else None),
    }




import json
import os
import time
import shutil
import logging
import mygene



def get_uniprot_id(
    gene_names,
    cache_file=os.path.join(DATA_DIR, "gene_cache.json"),
    manual_mapping_file=os.path.join(DATA_DIR, "manual_mapping.json"),
    use_manual_mapping=False,
    overwrite=False,  # 🔹 새 옵션: True일 때만 캐시 파일 쓰기
):
    """
    🔍 UniProt ID 조회 함수 (읽기 전용 + 선택적 캐시 업데이트)

    주된 기능:
        - 유전자명(gene name)을 입력받아 해당하는 UniProt ID를 반환합니다.
        - 캐시 파일과 수동 매핑 파일을 우선적으로 참조하고,
          없는 경우에만 MyGeneInfo API를 호출합니다.
        - 기본적으로는 **읽기 전용**으로 동작하며,
          `overwrite=True`일 때만 캐시 파일을 갱신(쓰기)합니다.

    Parameters
    ----------
    gene_names : str | list[str]
        조회할 유전자명 또는 유전자명 리스트.
        단일 문자열을 넣으면 단일 값으로 반환합니다.

    cache_file : str, optional
        UniProt ID 캐시 파일 경로 (기본: DATA_DIR/gene_cache.json).

    manual_mapping_file : str, optional
        사용자가 직접 정의한 수동 매핑 파일 경로 (기본: DATA_DIR/manual_mapping.json).

    use_manual_mapping : bool, optional
        True일 경우 수동 매핑 파일을 사용합니다.
        (manual_mapping_file이 존재해야 함)

    overwrite : bool, optional
        True일 때만 캐시 파일을 덮어씁니다.
        False이면 메모리 상 캐시만 갱신하고 디스크에는 쓰지 않습니다.

    Returns
    -------
    dict | str
        - 입력이 단일 문자열일 경우: 해당 gene의 UniProt ID (또는 None)
        - 입력이 리스트일 경우: {gene_name: UniProt ID} 딕셔너리

    Notes
    -----
    - 캐시 파일이 손상된 경우 경고 로그를 출력하고 빈 캐시로 진행합니다.
    - API 조회 실패 시 None을 반환합니다.
    - MyGeneInfo API 결과에서 Swiss-Prot > TrEMBL 순으로 ID를 선택합니다.
    """

    logger = logging.getLogger(__name__)
    if isinstance(gene_names, str):
        gene_names = [gene_names]

    # --- 1️⃣ 캐시 및 매핑 파일 읽기 ---
    gene_cache = {}
    manual_mapping = {}

    # 기존 캐시 파일 로드
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            try:
                gene_cache = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"[경고] 캐시 파일 손상 가능: {cache_file}, 빈 캐시로 진행")
                gene_cache = {}

    # 수동 매핑 파일 로드 (옵션)
    if use_manual_mapping and os.path.exists(manual_mapping_file):
        with open(manual_mapping_file, "r") as f:
            try:
                manual_mapping = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"[경고] 수동 매핑 파일 손상 가능: {manual_mapping_file}, 빈 매핑으로 진행")
                manual_mapping = {}

    mg = mygene.MyGeneInfo()
    results = {}

    # --- 2️⃣ 유전자별 UniProt ID 조회 ---
    for gene_name in gene_names:
        # (1) 캐시 존재 시 즉시 반환
        if gene_name in gene_cache:
            results[gene_name] = gene_cache[gene_name]
            continue

        # (2) 수동 매핑 파일에 존재할 경우 사용
        if use_manual_mapping and gene_name in manual_mapping:
            logger.info(f"[수동 매핑] {gene_name} -> {manual_mapping[gene_name]}")
            gene_cache[gene_name] = manual_mapping[gene_name]
            results[gene_name] = manual_mapping[gene_name]
            continue

        # (3) MyGeneInfo API 조회 수행
        try:
            result = mg.query(gene_name, fields="uniprot", species="human")
            if "hits" in result and result["hits"]:
                uniprot_id = None
                for hit in result["hits"]:
                    uniprot = hit.get("uniprot", {})
                    # 다양한 포맷 대응 (dict, str, list 등)
                    if isinstance(uniprot, dict):
                        uniprot_id = uniprot.get("Swiss-Prot") or uniprot.get("TrEMBL")
                    elif isinstance(uniprot, str):
                        uniprot_id = uniprot
                    if isinstance(uniprot_id, list):
                        uniprot_id = uniprot_id[0]

                    if uniprot_id:
                        gene_cache[gene_name] = uniprot_id
                        results[gene_name] = uniprot_id
                        logger.info(f"[API 조회] {gene_name} -> {uniprot_id}")
                        break

                if gene_name not in results:
                    logger.warning(f"[오류] {gene_name}에 대한 UniProt ID를 찾을 수 없습니다.")
                    results[gene_name] = None
            else:
                logger.warning(f"[오류] {gene_name}에 대한 UniProt ID를 찾을 수 없습니다.")
                results[gene_name] = None
        except Exception:
            logger.exception(f"[오류] {gene_name} 조회 중 오류 발생")
            results[gene_name] = None

    # --- 3️⃣ 캐시 파일 쓰기 (overwrite=True일 때만) ---
    if overwrite:
        try:
            with open(cache_file, "w") as f:
                json.dump(gene_cache, f)
            logger.info(f"[캐시 저장 완료] {cache_file}")
        except Exception:
            logger.exception(f"[오류] 캐시 파일 저장 실패: {cache_file}")

    # --- 4️⃣ 반환 형식 유지 (입력 형태에 따라) ---
    return results[gene_names[0]] if len(gene_names) == 1 else results




def add_manual_mapping(gene_name, uniprot_id,
                       manual_mapping_file=os.path.join(DATA_DIR, "manual_mapping.json")):
    """
    manual_mapping.json 파일에 유전자-유니프로트 ID 수동 매핑 추가

    Args:
        gene_name (str): 유전자 이름 (예: "MIR21")
        uniprot_id (str): 대응되는 UniProt ID (예: "Q9NQ66")
        manual_mapping_file (str): 매핑 파일 경로

    Returns:
        None
    """
    try:
        # 기존 매핑 로드
        if os.path.exists(manual_mapping_file):
            with open(manual_mapping_file, "r") as f:
                manual_mapping = json.load(f)
        else:
            manual_mapping = {}

        # 중복 확인
        if gene_name in manual_mapping:
            logger.warning(f"[중복 매핑] '{gene_name}'는 이미 '{manual_mapping[gene_name]}'으로 등록됨. 덮어씁니다.")

        # 매핑 추가 및 저장
        manual_mapping[gene_name] = uniprot_id
        with open(manual_mapping_file, "w") as f:
            json.dump(manual_mapping, f, indent=4)

        logger.info(f"[추가 완료] {gene_name} -> {uniprot_id} 수동 매핑 저장됨.")
    
    except Exception:
        logger.exception(f"[오류] 수동 매핑 저장 실패: {gene_name} -> {uniprot_id}")



# UniProt 서열 가져오기 함수
def get_protein_sequence(uniprot_id):
    """
    UniProt API에서 단백질 FASTA 서열을 가져오는 함수

    Args:
        uniprot_id (str): UniProt ID (예: "P04637")

    Returns:
        str 또는 None: 단백질 서열 문자열 (성공 시), 실패 시 None
    """
    """ 
    주어진 UniProt ID를 이용하여 REST API에서 FASTA 형식의 단백질 서열을 요청하고,
    응답에서 실제 서열만 파싱하여 반환하는 함수입니다.
    """

    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"  # UniProt FASTA API URL 구성
    try:
        response = requests.get(url)  # GET 요청으로 FASTA 데이터 요청
        if response.status_code == 200:  # 응답 성공 시
            lines = response.text.split('\n')  # 텍스트를 줄 단위로 분리
            sequence = ''.join([line.strip() for line in lines if not line.startswith('>')])  # 헤더('>') 제외하고 서열만 추출
            return sequence  # 서열 반환
        else:
            logger.error(f"[UniProt 응답 오류] {uniprot_id} → status {response.status_code}")  # 오류 로그 출력
    except Exception as e:
        logger.exception(f"[예외 발생] UniProt 서열 요청 실패: {uniprot_id}")  # 예외 발생 시 상세 로그 출력
    return None  # 실패 시 None 반환


from Bio.SeqUtils import seq1
import itertools
from emb_data import *


def generate_all_possible_mutations(seq):
    aa_list = list("ACDEFGHIKLMNPQRSTVWY")
    mutations = []
    for i, wt_aa in enumerate(seq):
        for mut_aa in aa_list:
            if mut_aa != wt_aa:
                mutations.append(f"{wt_aa}{i+1}{mut_aa}")
    return mutations


# 염기 치환 적용 함수
def apply_variant(sequence, position, variant):
    """
    염기 또는 아미노산 변이를 시퀀스에 적용
    """
    """ 
    단백질 시퀀스 상 특정 위치의 아미노산 또는 염기를 다른 값으로 치환하거나,
    '*' 변이인 경우 해당 위치부터 잘라내는 함수를 정의합니다.
    """

    try:
        position = int(position) - 1  # 1-based 인덱스를 0-based로 변환
        if variant == '*':  # 종결(stop) 변이인 경우
            return sequence[:position]  # 해당 위치 전까지 잘라냄

        if 0 <= position < len(sequence):  # 위치가 시퀀스 길이 범위 내인 경우
            new_seq = list(sequence)  # 문자열을 리스트로 변환하여 변경 가능하게 함
            new_seq[position] = variant  # 지정 위치의 아미노산을 변이로 교체
            return ''.join(new_seq)  # 리스트를 다시 문자열로 결합하여 반환
        return sequence  # 범위를 벗어난 경우 원본 시퀀스 반환
    except Exception as e:
        print(f"변이 적용 오류: {e}")  # 예외 발생 시 콘솔에 에러 출력
        return sequence  # 에러 발생 시에도 원본 시퀀스 반환


AA3_TO_1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C",
    "GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I",
    "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
    "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
    "SEC":"U","PYL":"O","TER":"*","STOP":"*" # stop
}
STOP_SYNONYMS_1 = {"*", "X"}                 # 1-letter stop
STOP_SYNONYMS_3 = {"TER", "STOP"}            # 3-letter stop (case-insensitive)

def aa3_to_1(s: str) -> str | None:
    return AA3_TO_1.get(s.upper())

def normalize_stop_token(tok: str) -> str:
    """Return '*' if token is any stop synonym, else original (uppercased for letters)."""
    t = tok.strip()
    if t.upper() in STOP_SYNONYMS_3 or t in STOP_SYNONYMS_1:
        return "*"
    # 1-letter amino
    if len(t) == 1 and t.isalpha():
        return t.upper()
    return t  # e.g., already '*'

def parse_variant(variant: str):
    """
    치환 변이 파싱:
    - 아미노(혼합 표기 허용):
        'Y220C', 'p.Y220C', 'p.Tyr220Cys',
        'Q331Ter', 'Y220*', 'Y220X', 'p.Gln331*', 'p.Tyr220Stop'
    - DNA:
        'A123G', 'c.123A>G', 'g.123A>G'
    return: (level, ref, pos(int), alt)
        level in {'amino','dna'}
    """
    v = variant.strip()

    # 1) 아미노: 1-letter ref + pos + (1-letter | stop-synonym)
    m = re.match(r'^(?:p\.)?([A-Za-z\*])(\d+)([A-Za-z\*]|Ter|Stop|X)$', v, flags=re.IGNORECASE)
    if m:
        ref_raw, pos, alt_raw = m.group(1), int(m.group(2)), m.group(3)
        ref = normalize_stop_token(ref_raw)
        alt = normalize_stop_token(alt_raw)
        if ref == "*" or len(ref) != 1 or (alt != "*" and len(alt) != 1):
            raise ValueError(f"지원하지 않는 아미노산 표기: {variant}")
        return "amino", ref, pos, alt

    # 2) 아미노: 3-letter ref + pos + (3-letter | stop-synonym)
    m = re.match(r'^(?:p\.)?([A-Za-z]{3})(\d+)([A-Za-z]{3}|Ter|Stop)$', v, flags=re.IGNORECASE)
    if m:
        ref3, pos, alt3 = m.group(1), int(m.group(2)), m.group(3)
        ref1 = aa3_to_1(ref3)
        alt1 = "*" if alt3.upper() in STOP_SYNONYMS_3 else aa3_to_1(alt3)
        if not ref1 or not alt1:
            raise ValueError(f"지원하지 않는 3-letter AA: {ref3}->{alt3}")
        if ref1 == "*":
            raise ValueError(f"Ref에 stop 사용 불가: {variant}")
        return "amino", ref1, pos, alt1

    # 3) DNA: c.123A>G / g.123A>G
    m = re.match(r'^(?:[cg]\.)?(\d+)([ACGT])>([ACGT])$', v, flags=re.IGNORECASE)
    if m:
        pos, ref, alt = int(m.group(1)), m.group(2).upper(), m.group(3).upper()
        return "dna", ref, pos, alt

    # 4) DNA: A123G
    m = re.match(r'^([ACGT])(\d+)([ACGT])$', v, flags=re.IGNORECASE)
    if m:
        ref, pos, alt = m.group(1).upper(), int(m.group(2)), m.group(3).upper()
        return "dna", ref, pos, alt

    raise ValueError(f"지원하지 않는 변이 표기: {variant}")

def apply_variant_amino(seq: str, pos: int, ref: str, alt: str) -> str:
    """
    아미노산 서열에 변이를 적용.
    - pos: 1-based 인덱스
    - ref == '' 또는 '-'  -> insertion (pos 위치 '앞'에 alt를 삽입; pos=len(seq)+1이면 맨 뒤 삽입)
    - alt == '' 또는 '-'  -> deletion (pos에서 시작하는 ref 문자열을 삭제)
    - len(ref)>0 and len(alt)>0 -> 치환(블록 치환 포함)
    - alt == '*' (STOP)      -> nonsense: pos 위치에서 번역 종료로 간주하여 seq[:pos-1] 반환(별표는 포함하지 않음)

    반환: 변이가 적용된 새로운 서열(대문자 권장)
    """
    if not isinstance(seq, str):
        raise TypeError("seq must be str")
    if pos < 1:
        raise IndexError(f"pos={pos} out of range (must be >=1)")
    ref = (ref or "").upper()
    alt = (alt or "").upper()
    seq_u = seq.upper()

    # --- Insertion ---
    if ref in ("", "-"):
        # 삽입은 pos가 len(seq)+1까지 허용(맨 뒤 삽입)
        if pos > len(seq) + 1:
            raise IndexError(f"insertion pos={pos} out of range (len={len(seq)})")
        return seq[:pos-1] + alt + seq[pos-1:]

    # 이후부터는 pos가 반드시 서열 범위 내여야 함
    if pos > len(seq):
        raise IndexError(f"pos={pos} out of range (len={len(seq)})")

    # 대상 구간 (블록 길이 = len(ref))
    end = pos - 1 + len(ref)
    if end > len(seq):
        raise IndexError(f"ref extends past sequence end: pos={pos}, len(ref)={len(ref)}, len(seq)={len(seq)}")

    # 참조 확인
    if seq_u[pos-1:end] != ref:
        raise ValueError(f"Ref mismatch at {pos}: expected {ref}, found {seq_u[pos-1:end]}")

    # --- Deletion ---
    if alt in ("", "-"):
        return seq[:pos-1] + seq[end:]

    # --- Substitution / Delins ---
    # STOP: 번역 종료로 처리(별표는 넣지 않고 그 지점에서 절단)
    if alt == "*" or alt == "Ter":
        # nonsense: pos 코돈에 STOP이 생겼다고 보고 pos-1까지 남김
        return seq[:pos-1]

    # 일반 치환(블록 치환 포함)
    return seq[:pos-1] + alt + seq[end:]


def apply_variant_dna(seq: str, pos: int, ref: str, alt: str) -> str:
    """
    핵산 서열에 변이를 적용 (VCF/HGVS 유사 규칙 단순화 버전).
    - pos: 1-based
    - ref == '' 또는 '-'  -> insertion (pos 위치 '앞'에 alt 삽입; pos=len(seq)+1이면 맨 뒤 삽입)
    - alt == '' 또는 '-'  -> deletion (pos에서 시작하는 ref 문자열 삭제)
    - len(ref)>0 and len(alt)>0 -> 치환(블록 치환 포함; delins)
    반환: 변이가 적용된 새로운 핵산 서열
    """
    if not isinstance(seq, str):
        raise TypeError("seq must be str")
    if pos < 1:
        raise IndexError(f"pos={pos} out of range (must be >=1)")
    ref = (ref or "").upper()
    alt = (alt or "").upper()
    seq_u = seq.upper()

    # 삽입
    if ref in ("", "-"):
        if pos > len(seq) + 1:
            raise IndexError(f"insertion pos={pos} out of range (len={len(seq)})")
        return seq[:pos-1] + alt + seq[pos-1:]

    # 그 외(치환/삭제): pos는 범위 내
    if pos > len(seq):
        raise IndexError(f"pos={pos} out of range (len={len(seq)})")

    end = pos - 1 + len(ref)
    if end > len(seq):
        raise IndexError(f"ref extends past sequence end: pos={pos}, len(ref)={len(ref)}, len(seq)={len(seq)}")

    if seq_u[pos-1:end] != ref:
        raise ValueError(f"Ref mismatch at {pos}: expected {ref}, found {seq_u[pos-1:end]}")

    # 삭제
    if alt in ("", "-"):
        return seq[:pos-1] + seq[end:]

    # 치환/블록 치환
    return seq[:pos-1] + alt + seq[end:]

def _parse_a3m_to_list(a3m_text: str):
    names, seqs = [], []
    cur_name, buf = None, []
    for line in a3m_text.splitlines():
        if not line: 
            continue
        if line.startswith('>'):
            if cur_name is not None:
                seqs.append(''.join(buf))
            cur_name = line[1:].strip()
            names.append(cur_name)
            buf = []
        else:
            buf.append(line.strip())
    if cur_name is not None:
        seqs.append(''.join(buf))
    return list(zip(names, seqs))

def _format_a3m(names_seqs, wrap: int = 80) -> str:
    out = []
    for n, s in names_seqs:
        out.append(f">{n}")
        for i in range(0, len(s), wrap):
            out.append(s[i:i+wrap])
    return "\n".join(out) + "\n"

def _find_column_for_master_pos(aligned_master: str, pos_1based: int) -> int:
    """
    마스터 정렬에서 비갭('-'이 아닌) 문자의 pos_1based번째가 나오는 열 index(0-based)를 반환.
    """
    cnt = 0
    for col, ch in enumerate(aligned_master):
        if ch != '-':
            cnt += 1
            if cnt == pos_1based:
                return col
    raise IndexError(f"Master pos {pos_1based} not found (too large?)")


def apply_variant_on_a3m(a3m_text: str, pos: int, ref: str, alt: str, *, stop_policy: str = "gap") -> str:
    """
    A3M 텍스트에 대해 마스터(첫 서열)만 치환.
    - alt == '*' 인 경우(조기 종결): 기본 stop_policy='gap'이면 stop 지점부터 끝까지 '-'로 마스킹.
      (A3M/HH-suite/MSA-Transformer 호환성 가장 좋음)
    - stop_policy='star' 를 쓰면 stop 위치만 '*'로 표기하고 그 뒤는 '-'로 마스킹(일부 도구에서 '*' 미지원일 수 있음).
    """
    names_seqs = _parse_a3m_to_list(a3m_text)
    if not names_seqs:
        raise ValueError("빈 A3M")
    master_name, master = names_seqs[0]

    col = _find_column_for_master_pos(master, pos)
    master_ref = master[col]
    if master_ref == '-':
        raise ValueError(f"Master column {col} is gap; pos={pos} not aligned to residue")
    if master_ref.upper() != ref.upper():
        raise ValueError(f"Ref mismatch at pos={pos} (col {col}): expected {ref}, found {master_ref}")

    master_list = list(master)

    if alt == '*':
        # === 조기 종결 처리 ===
        if stop_policy == "star":
            # 일부 파이프라인에서 '*' 미지원 가능. 필요할 때만 사용.
            master_list[col] = '*'
        else:
            # 기본: stop 위치도 포함해 뒤를 모두 gap으로 마스킹
            master_list[col] = '-'
        # stop 이후(오른쪽) 모두 gap으로 마스킹
        for j in range(col + 1, len(master_list)):
            if master_list[j] != '-':
                master_list[j] = '-'
    else:
        # === 일반 치환 ===
        master_list[col] = alt.upper()

    names_seqs[0] = (master_name, ''.join(master_list))
    return _format_a3m(names_seqs)


# 데이터 수집 함수
def collect_data(gene_name, variant, embedding_model_name, mutation_type,
                database_path=os.path.join(DATA_DIR,"sequence_embedding.zarr"),
                force: bool = False):
    """
    주어진 유전자 이름과 변이 정보를 이용해:
    1) UniProt ID 조회 → 2) 서열 추출 및 변이 적용 → 3) 선택된 임베딩 모델로 벡터화 후
    Zarr 데이터베이스에 변이별 임베딩 결과를 저장하는 함수입니다.

    Parameters
    ----------
    gene_name : str
        유전자 이름
    variant : str
        변이 정보 (예: "Y220C")
    embedding_model_name : str
        사용할 임베딩 모델 이름
    mutation_type : str
        'dna' 또는 'amino'
    database_path : str
        Zarr 데이터베이스 파일 경로
    force : bool
        이미 존재하는 경우 덮어쓸지 여부 (기본값: False)
    """

    try:
        # 유전자 명칭을 UniProt ID로 매칭
        uniprot_id = get_uniprot_id(gene_name)  # gene_name을 UniProt ID로 변환
        if not uniprot_id:  # ID가 없을 경우 수동 매핑 시도
            print(f"UniProt ID를 찾을 수 없습니다: {gene_name}")  # 에러 출력
            print(f'add_manual_mapping("{gene_name}" > "???")')  # 수동 매핑 유도 메시지
            return  # 중복 종료 방지용

        # 데이터베이스 경로 열기
        root = zarr.open(database_path, mode='a')  # Zarr 파일 열기 (append 모드)

        # 1) 기존 데이터 중복 확인 (임베딩 경로 존재 여부는 계층 경로 안전 체크로!)
        embeddings_path = f"{uniprot_id}/embeddings/{mutation_type}/{variant}/{embedding_model_name}"
        if embeddings_path in root and not force:
            print(f"이미 존재하는 데이터: {embeddings_path}")
            return

        # 2) 메타데이터
        gene_group = root.require_group(uniprot_id)  # 유전자 그룹 생성 또는 로딩
        metadata = gene_group.require_group("metadata")  # 메타데이터 그룹 생성
        metadata.attrs["gene_name"] = gene_name  # 메타데이터: 유전자 이름 저장
        metadata.attrs["uniprot_id"] = uniprot_id  # 메타데이터: UniProt ID 저장

        # 3) 입력 서열/정렬 로드
        if mutation_type in ("dna", "amino"):
            seq_ds_path = f"{uniprot_id}/sequences/{mutation_type}"
            if zhas(root, seq_ds_path):
                input_text = zread_text(root, seq_ds_path)
            else:
                time.sleep(1.5)  # API rate limit 여유
                input_text = get_protein_sequence(uniprot_id) if mutation_type == "amino" else get_dna_sequence(uniprot_id)
                if not input_text:
                    print(f"서열을 가져올 수 없습니다: {uniprot_id} ({mutation_type})")
                    return
                zwrite_text(root, seq_ds_path, input_text, chunks=(4096,))
            input_seq = input_text

        elif mutation_type == "aminoMSA":
            a3m_ds_path = f"{uniprot_id}/sequences/aminoMSA/a3m"
            if zhas(root, a3m_ds_path):
                a3m_text = zread_text(root, a3m_ds_path)
            else:
                # 없으면 MSA 생성 콜백으로 생성
                if build_msa_for_uniprot is None:
                    raise RuntimeError("MSA가 없고 msa_builder 콜백이 없습니다. aminoMSA 분기를 위해 msa_builder를 전달하세요.")
                a3m_file = build_msa_for_uniprot(
                                uniprot_id,
                                db_prefix="/NFS_DATA/samsung/apocalypse/database/msa/uniclust30_2018_08/uniclust30_2018_08/uniclust30_2018_08",  # 또는 환경변수
                                msa_dir=os.path.join(DATA_DIR, "msa"),
                            )
                a3m_text = Path(a3m_file).read_text()
                zwrite_text(root, a3m_ds_path, a3m_text, chunks=(65536,))
            input_seq = a3m_text

        else:
            raise ValueError(f"지원하지 않는 mutation_type: {mutation_type}")


        # 4. 염기 치환 처리 (아미노산 변이와 DNA 변이 모두 처리)
        if variant != 'REF':
            try:
                level, ref, pos, alt = parse_variant(variant)  # ('amino' or 'dna', ref, pos, alt)

                if mutation_type == 'aminoMSA':
                    # input_seq 가 A3M 텍스트여야 합니다. (앞서 zarr에서 aminoMSA/a3m 로드해둔 값)
                    if level != "amino":
                        raise ValueError(f"MSA 변이는 아미노 표기만 지원합니다: {variant}")
                    input_seq = apply_variant_on_a3m(a3m_text, pos, ref, alt)   # A3M 텍스트 반환
                    print(f"[MSA] 변이 적용: {variant} (마스터 행만 치환)")

                elif mutation_type == 'amino':
                    if level != "amino":
                        raise ValueError(f"아미노 변이 표기를 기대했으나 {level}: {variant}")
                    input_seq = apply_variant_amino(input_text, pos, ref, alt)    # 단일 AA 서열
                    print(f"[AA ] 변이 적용: {variant}")

                elif mutation_type == 'dna':
                    if level != "dna":
                        raise ValueError(f"DNA 변이 표기를 기대했으나 {level}: {variant}")
                    input_seq = apply_variant_dna(input_text, pos, ref, alt)      # 단일 DNA 서열
                    print(f"[DNA] 변이 적용: {variant}")

                else:
                    raise ValueError(f"지원하지 않는 mutation_type: {mutation_type}")

            except Exception as e:
                print(f"변이 처리 오류: {e}")
                print(f"서열/정렬 미리보기: {str(input_seq)[:120]}...")
                # 필요 시 반환/중단 정책 선택
                # return input_seq

        # 5. 임베딩 모델 선택
        if embedding_model_name == 'GPN-MSA':  # GPN-MSA 모델 사용 시
            embedding = embed_GPNMSA(input_seq)  # 임베딩 수행


        # === [NEW] xTrimoPGLM / ProteinGLM ===
        elif embedding_model_name.startswith("xTrimoPGLM"):
            if mutation_type != "amino":
                raise ValueError("xTrimoPGLM은 단일 아미노산 서열(mutation_type='amino')만 지원합니다.")
            
            xtrimo_hf_map = {
                "xTrimoPGLM-1B-MLM":  "biomap-research/xtrimopglm-1b-mlm",
                "xTrimoPGLM-3B-MLM":  "biomap-research/proteinglm-3b-mlm",
                "xTrimoPGLM-10B-MLM": "biomap-research/proteinglm-10b-mlm",
                # 필요하면 100B INT4도 추가 가능:
                # "xTrimoPGLM-100B-INT4-MLM": "biomap-research/xtrimopglm-100b-int4",
            }

            try:
                model_id = xtrimo_hf_map[embedding_model_name]
            except KeyError:
                raise ValueError(f"xTrimoPGLM 매핑이 없는 embedding_model_name: {embedding_model_name}")

            # torch.Tensor (1, L, D)
            emb_tensor = embed_xTrimoPGLM(input_seq,model_id=model_id,)
            embedding = emb_tensor  # 아래에서 numpy로 통일 변환


        elif embedding_model_name == 'esm2_t33_650M_UR50D':  # ESM2 모델 사용 시
            embedding = embed_esm2_t33_650M_UR50D(input_seq)  # 임베딩 수행



        elif embedding_model_name == 'esm_msa1_t12_100M_UR50S':  # MSA-Transformer
            if mutation_type != "aminoMSA":
                raise ValueError("esm_msa1_t12_100M_UR50S는 aminoMSA 입력만 지원합니다.")
            # input_seq는 A3M 텍스트이어야 함
            embedding = embed_msa_from_a3m_text(input_seq, layer=12, top_n=64, truncate_to=512)[:,0] # Query embedding만 추출. Human sequence embedding
            # 결과: (1, N, L, C)



        elif embedding_model_name in ANKH_MODELS:
            if mutation_type != "amino":
                raise ValueError("Ankh 모델은 amino 서열만 지원합니다.")

            model_id = ANKH_MODELS[embedding_model_name]["hf_name"]
            emb_tensor = embed_Ankh(
                input_seq,
                model_id=model_id,
                device=device
            )
            embedding = emb_tensor



        elif embedding_model_name in PROTTRANS_MODELS:
            if mutation_type != "amino":
                raise ValueError(
                    f"{embedding_model_name}는 단일 아미노산 서열만 지원합니다. (mutation_type='amino')"
                )
            emb_tensor = embed_ProtTrans(
                input_seq,
                embedding_model_name=embedding_model_name,
                per_residue=True,
            )  # (1, L, D) torch.Tensor
            embedding = emb_tensor


        else:  # 지원하지 않는 모델일 경우
            print(f"지원하지 않는 임베딩 모델: {embedding_model_name}")  # 에러 출력
            return  # 함수 종료

        # # 5-1. Tensor를 NumPy로 변환 (필수)
        if isinstance(embedding, torch.Tensor):  # Tensor일 경우
            if embedding.dtype == torch.bfloat16:
                embedding = embedding.to(torch.float32)
            embedding = embedding.detach().cpu().numpy()  # NumPy 배열로 변환 # [WARNING] 일단 현재 생성한 embedding data는 numpy로 저장되었지만, 추후 대규모로 생성할 경우 tensor 저장 고려
        # # 5-1. Tensor로 변환 (필수)
        # embedding = torch.as_tensor(embedding, dtype=torch.float32).detach().cpu()



        # 6. 임베딩 데이터 저장
        embeddings_group = gene_group.require_group("embeddings")  # 임베딩 그룹 생성
        variant_group = embeddings_group.require_group(mutation_type).require_group(variant)  # 변이별 그룹 생성

        if embedding_model_name in variant_group:
            if force:
                del variant_group[embedding_model_name]
                print(f"기존 데이터 삭제: {uniprot_id}, {variant}, {embedding_model_name}")
                # # 삭제 후 검증
                # if embedding_model_name in variant_group:
                #     raise RuntimeError(f"[오류] {embedding_model_name} 삭제 실패")
                # else:
                #     print(f"[확인] 기존 데이터 삭제 완료: {uniprot_id}, {variant}, {embedding_model_name}")
            else:
                print(f"이미 존재하는 데이터: {uniprot_id}, {variant}, {embedding_model_name}")
                return

        variant_group.create_dataset(
            embedding_model_name,
            shape=embedding.shape,
            dtype=np.float32, # ⚠
            data=embedding
        )
        print(f"데이터 저장 완료: {uniprot_id}, {variant}, {embedding_model_name}, {embedding.shape}")

    except Exception as e:  # 전체 예외 처리
        logger.exception(f"[collect_data 실패] {gene_name}, {variant}, {embedding_model_name}, {mutation_type}")  # 예외 로그 출력



def load_data(gene_name, variant, embedding_model_name, mutation_type, database_path=os.path.join(DATA_DIR,"sequence_embedding.zarr"), visible=True):
    """
    주어진 유전자명과 변이 정보를 기준으로 Zarr 데이터베이스에서
    사전 계산된 임베딩 데이터를 로드하는 함수입니다.
    존재하지 않는 경우 None을 반환하며, 로딩 성공 시 벡터와 로그를 출력합니다.
    """
    try:
        # 1. 데이터베이스 경로 열기
        root = zarr.open(database_path, mode='r')  # 읽기 모드로 Zarr DB 열기

        # UniProt ID 조회 (수동 매핑 허용)
        uniprot_id = get_uniprot_id(gene_name, use_manual_mapping=True)  # 유전자명을 UniProt ID로 변환
        if not uniprot_id:  # 변환 실패 시
            logger.error(f"UniProt ID를 찾을 수 없습니다: {gene_name}")  # 에러 로그 출력
            logger.info(f'[수동 매핑 필요] add_manual_mapping("{gene_name}", "???")')  # 수동 매핑 안내 로그
            return  # 종료

        # 2. 유전자 ID 매핑 (UniProt ID로 저장됨)
        if uniprot_id not in root:  # DB에 해당 ID가 없을 경우
            print(f"유전자 {uniprot_id} 데이터가 존재하지 않습니다.")  # 사용자 출력
            return None  # None 반환

        # 3. 데이터 경로 구성
        embeddings_path = f"embeddings/{mutation_type}/{variant}/{embedding_model_name}"  # 임베딩 경로 구성
        gene_group = root[uniprot_id]  # 해당 유전자 그룹 접근

        # 4. 데이터 존재 여부 확인
        if embeddings_path not in gene_group:  # 임베딩 경로가 존재하지 않을 경우
            logger.warning(f"[임베딩 없음] {uniprot_id}, {variant}, {embedding_model_name}, {mutation_type}")  # 경고 로그 출력
            return None  # None 반환

        # 5. 데이터 로딩
        embedding = gene_group[embeddings_path][:]  # 실제 임베딩 벡터 로드
        if visible:  # visible 옵션이 True일 경우 로그 출력
            logger.info(f"[로드 완료] {uniprot_id}, {variant}, {embedding_model_name}, {mutation_type}")  # 완료 로그
            logger.info(f"[임베딩 크기] {embedding.shape}")  # 임베딩 shape 로그

        return embedding  # 로드된 임베딩 벡터 반환

    except Exception as e:  # 예외 발생 시
        logger.exception(f"[데이터 로드 중 오류 발생]: {gene_name}, {variant}, {embedding_model_name}, {mutation_type}")  # 예외 정보 출력
        logger.exception(f"[오류 메세지]: {e}")  # 예외 메시지 로그
        return None  # None 반환



def preload_embeddings(variant_list, embedding_model_name, mutation_type,
                       database_path=os.path.join(DATA_DIR, "sequence_embedding.zarr")):
    """
    variant_list: ["KRAS~G12D+ctrl", "KRAS~G13D+ctrl", ..., "ctrl"]
    반환: {(gene, variant): embedding array}
    """
    """
    주어진 variant_list에 포함된 유전자-변이 조합들에 대해,
    미리 Zarr 데이터베이스에서 해당 임베딩 벡터를 로드하여 캐시에 저장합니다.
    'ctrl' 항목은 고정된 제로벡터로 처리하며, 변이가 없는 'REF'도 별도로 캐시합니다.
    """

    embedding_cache = {}  # 결과 저장용 임베딩 캐시 딕셔너리 초기화
    try:
        root = zarr.open(database_path, mode='r')  # 읽기 모드로 Zarr DB 열기
    except Exception:  # 열기 실패 시 예외 처리
        logger.exception(f"[Zarr 오류] 데이터베이스 열기 실패: {database_path}")  # 에러 로그 출력
        return {}  # 빈 딕셔너리 반환

    for gene_var_ in variant_list:  # variant 리스트 순회
        if gene_var_ == 'ctrl':  # 'ctrl'인 경우
            # [WARNING] ctrl을 사용하는 경우는 없겠지만, 모델마다 차원 수가 다르다는 것을 인지할 것.
            embedding_cache[('ctrl', 'REF')] = np.zeros((1280, 1))  # 제로벡터로 채운 기본 REF embedding 생성
            continue  # 다음 condition으로...

        if '+' in gene_var_:
            gene_var_ = gene_var_.split("+")[0]
        else:
            pass
        # 유전자 및 변이 파싱
        if '~' in gene_var_:  # Gene~Variant 형태인 경우
            gene, variant = gene_var_.split("~")  # gene과 variant 분리
        else:  # 변이 정보가 없는 경우
            gene, variant = gene_var_, "REF"  # 변이명을 REF로 설정

        # UniProt ID 매핑
        uniprot_id = get_uniprot_id(gene, use_manual_mapping=True)  # gene명을 UniProt ID로 변환
        if not uniprot_id:  # 매핑 실패 시
            logger.error(f"[UniProt ID 실패] {gene}")  # 에러 로그
            logger.info(f'[수동 매핑 필요] add_manual_mapping("{gene}", "???")')  # 수동 매핑 유도 로그
            continue  # 해당 항목 건너뜀

        if uniprot_id not in root:  # Zarr DB에 해당 ID가 없을 경우
            logger.warning(f"[Zarr 누락] {uniprot_id} not in root")  # 경고 로그 출력
            continue  # 다음 항목으로 이동

        group = root[uniprot_id]  # 해당 유전자 그룹 접근
        path = f"embeddings/{mutation_type}/{variant}/{embedding_model_name}"  # 임베딩 벡터 경로 구성
        if path in group:  # 경로가 존재하는 경우
            embedding = group[path][:]  # 벡터 로드
            embedding_cache[(gene, variant)] = embedding  # 캐시에 저장
            logger.debug(f"[로드 성공] {gene}~{variant}")  # 디버그 로그
        else:  # 임베딩 데이터가 존재하지 않을 경우
            logger.warning(f"[임베딩 누락] Missing: {gene}~{variant}")  # 경고 로그 출력

        # REF embedding 로드 (없으면 캐시)
        ref_path = f"embeddings/{mutation_type}/REF/{embedding_model_name}"  # REF 경로 구성
        if (gene, "REF") not in embedding_cache and ref_path in group:  # 아직 캐시에 없고 경로가 존재할 경우
            embedding_cache[(gene, "REF")] = group[ref_path][:]  # REF 벡터 로드 및 캐시에 저장
            logger.debug(f"[REF 로드] {gene}~REF")  # 디버그 로그 출력

    logger.info(f"✅ preload_embeddings 완료 - 총 {len(embedding_cache)}개 로드됨")  # 최종 로드 결과 출력
    return embedding_cache  # 로드된 임베딩 캐시 딕셔너리 반환


def get_cached_embedding(
    gene_name,
    variant,
    cache,
    representation: str = "DIFF",
    as_tensor: bool = False,
    device=None,
):
    """
    미리 로딩된 임베딩 캐시에서 (gene_name, variant)에 해당하는 임베딩을 가져오는 함수.

    Parameters
    ----------
    gene_name : str
        유전자 이름. 'TP53', 'KRAS' 또는 'ctrl' 등.
        'ctrl'인 경우 특수 처리한다.
    variant : str or None
        변이 표기. 예) 'Y220C', 'G12D'.
        ctrl에는 보통 None 또는 'REF'가 들어올 수 있으나, 실제 키는 ('ctrl', 'REF')만 사용.
    cache : dict
        preload_embeddings에서 반환한 dict.
        일반적인 구조 예:
            cache[('TP53', 'Y220C')] = {
                'ALT':  np.ndarray(shape=(1280,)),
                'REF':  np.ndarray(shape=(1280,)),
                'DIFF': np.ndarray(shape=(1280,)),
            }
            cache[('ctrl', 'REF')] = np.ndarray(shape=(1280,))  # 혹은 dict

    representation : {"REF", "ALT", "DIFF", None}, default = "DIFF"
        어떤 임베딩을 가져올지 지정.
        - "REF"  : REF 벡터
        - "ALT"  : ALT 벡터
        - "DIFF" : DIFF(보통 REF-ALT) 벡터
        - None   : 캐시에 저장된 dict 전체를 그대로 반환

    as_tensor : bool, default = False
        True이면 torch.Tensor로 변환해서 반환한다.

    device : torch.device or str or None, default = None
        as_tensor=True일 때 텐서를 올릴 device. 예) "cuda", "cuda:0", "cpu".

    Returns
    -------
    np.ndarray or torch.Tensor or dict or None
        - 요청한 임베딩이 존재하면 해당 벡터(또는 dict)를 반환
        - 캐시에 없거나 요청한 representation가 없으면 None 반환 (경고 로그 출력)

    Notes
    -----
    * gene_name == "ctrl" 인 경우:
        - 항상 ('ctrl', 'REF') 키를 사용해 불러온 뒤 REF 벡터를 반환한다.
        - representation 인자는 무시하고 REF로 고정 (control baseline).
    """

    import numpy as np

    # 내부 helper: numpy array를 tensor로 바꾸기
    def _to_tensor(x):
        if not as_tensor:
            return x
        import torch

        t = torch.as_tensor(x)
        if device is not None:
            t = t.to(device)
        return t

    # ------------------------------
    # 1) ctrl 특수 처리
    # ------------------------------
    if gene_name == "ctrl":
        ctrl_key = ("ctrl", "REF")
        if ctrl_key not in cache:
            logger.warning(f"[Cache miss - ctrl] {ctrl_key}")
            print(f"Cache miss for {ctrl_key}")
            return None

        ctrl_entry = cache[ctrl_key]

        # ctrl이 dict 형태이든, 바로 벡터이든 모두 처리
        if isinstance(ctrl_entry, dict):
            if "REF" not in ctrl_entry:
                logger.warning(f"[Cache miss - ctrl REF] {ctrl_key} has no 'REF' field")
                print(f"Cache miss for {ctrl_key}['REF']")
                return None
            vec = ctrl_entry["REF"]
        else:
            vec = ctrl_entry  # 이미 벡터라고 가정

        logger.debug(f"[Cache hit - ctrl] {ctrl_key}")
        return _to_tensor(vec)

    # ------------------------------
    # 2) 일반 (gene, variant) 처리
    # ------------------------------
    key = (gene_name, variant)

    if key not in cache:
        logger.warning(f"[Cache miss] {key}")
        print(f"Cache miss for {key}")
        return None

    entry = cache[key]
    logger.debug(f"[Cache hit] {key}")

    # representation=None 이면 dict 그대로 반환 (tensor 변환 안 함)
    if representation is None:
        return entry

    # entry가 dict인 경우: 'REF', 'ALT', 'DIFF' 중 하나 선택
    if isinstance(entry, dict):
        if representation not in entry:
            logger.warning(f"[Cache missing representation] {key} has no '{representation}' field")
            print(f"Cache missing representation '{representation}' for {key}")
            return None
        vec = entry[representation]
    else:
        # dict가 아니라면 단일 벡터라고 보고 그대로 반환
        vec = entry

    return _to_tensor(vec)



def save_embedding_cache(cache, filepath):
    """
    임베딩 캐시를 pickle 파일로 저장합니다.
    cache: dict {(gene, variant): embedding}
    filepath: 저장할 .pkl 경로
    """
    """
    메모리에 존재하는 임베딩 캐시 딕셔너리를 주어진 경로에 pickle 포맷으로 저장합니다.
    저장 성공 시 로그를 출력하고, 실패 시 예외 로그를 기록합니다.
    """
    try:
        with open(filepath, 'wb') as f:  # 바이너리 쓰기 모드로 파일 열기
            pickle.dump(cache, f)  # pickle을 이용하여 cache 저장
        logger.info(f"✅ 임베딩 캐시 저장 완료: {filepath}")  # 저장 성공 로그 출력
    except Exception:  # 예외 발생 시
        logger.exception(f"[오류] 캐시 저장 실패: {filepath}")  # 예외 로그 출력


def load_embedding_cache(filepath):
    """
    저장된 pickle 파일에서 임베딩 캐시를 불러옵니다.
    반환값: dict {(gene, variant): embedding}
    """
    """
    지정된 경로의 pickle 파일에서 임베딩 캐시를 로드하여 반환합니다.
    파일이 없거나 오류가 발생하면 빈 딕셔너리를 반환합니다.
    """
    try:
        with open(filepath, 'rb') as f:  # 바이너리 읽기 모드로 파일 열기
            cache = pickle.load(f)  # pickle을 이용하여 캐시 로딩
        logger.info(f"✅ 임베딩 캐시 불러오기 완료: {filepath}")  # 로딩 성공 로그 출력
        return cache  # 불러온 딕셔너리 반환
    except Exception:  # 예외 발생 시
        logger.exception(f"[오류] 캐시 불러오기 실패: {filepath}")  # 예외 로그 출력
        return {}  # 빈 딕셔너리 반환


from torch.nn.functional import pad
def pad_to_length(embedding, target_len):
    """
    주어진 임베딩 텐서를 target_len 길이에 맞게 pad 또는 truncate하는 함수입니다.
    """
    pad_len = target_len - embedding.shape[1]  # 필요한 padding 길이 계산
    if pad_len > 0:  # 현재 길이가 부족할 경우
        return pad(embedding, (0, 0, 0, pad_len))  # 시퀀스 길이 차원에 패딩 추가
    else:  # 너무 길면 자름
        return embedding[:, :target_len, :]  # target_len까지 잘라서 반환




### ================================================================ ###
# NEW: imports
import subprocess, random, re
from pathlib import Path

def _which_or_die(name: str):
    p = shutil.which(name)
    if not p:
        raise FileNotFoundError(f"필수 실행파일 '{name}'이 PATH에 없습니다.")
    return p

VALID_AA = set("ACDEFGHIKLMNPQRSTVWYBXZJUO*")  # *은 stop → 제거할 예정
def _sanitize_seq(seq: str) -> str:
    """공백/갭/소문자 제거, stop(*) 제거, 비표준 문자는 X로 치환."""
    s = "".join(seq.split()).replace("-", "").replace(".", "")
    s = s.upper().replace("*", "")
    return "".join(ch if ch in VALID_AA else "X" for ch in s)

def _run(cmd: list[str], cwd: str | None = None):
    print("$", " ".join(cmd))
    cp = subprocess.run(cmd, cwd=cwd, text=True,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if cp.returncode != 0:
        print(cp.stdout)
        print(cp.stderr)
        raise RuntimeError(f"command failed: {' '.join(cmd)}")
    return cp




# NEW: A3M 로더 (소문자 삽입 제거, '-' 유지)
def read_a3m(a3m_path: str):
    names, seqs, cur = [], [], []
    with open(a3m_path) as f:
        for line in f:
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                    cur = []
                names.append(line[1:].strip())
            else:
                cur.append(line.strip())
        if cur: seqs.append("".join(cur))
    # 삽입 제거
    seqs = ["".join([c for c in s if (c == "-" or c.isupper())]) for s in seqs]
    Lset = {len(s) for s in seqs}
    if len(Lset) != 1:
        raise ValueError("MSA 열 길이가 일치하지 않습니다. reformat.pl -M first -r 로 정리 필요")
    return names, seqs

# NEW: 쿼리 ungapped 1-based 위치 → 정렬 열(0-based)
def ungapped_pos_to_col(aligned_query: str, pos_1based: int) -> int:
    cnt = 0
    for col, aa in enumerate(aligned_query):
        if aa != "-":
            cnt += 1
            if cnt == pos_1based:
                return col
    raise ValueError(f"쿼리 길이를 넘는 위치: {pos_1based}")

# NEW: MSA-Transformer 점수화 (masked-marginal ΔlogP; 첫 행만 마스크/스코어)
@torch.no_grad()
def score_variants_with_msa(
    a3m_path: str,
    variants: list[list[str]],         # 예: [["Y220C"], ["G245S","R248Q"]]
    n_context: int = 384,
    n_repeats: int = 3,
    device: str | None = None
) -> dict[str, float]:
    """
    반환: {"Y220C": score, "G245S+R248Q": score, ...}   (높을수록 허용/중립 경향)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 모델 로드
    model, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    mask_idx = alphabet.mask_idx

    # MSA 읽기
    names, seqs = read_a3m(a3m_path)
    query = seqs[0]
    L = len(query)

    # 변이 파싱 → 정렬열/문자
    varsets = []
    for varset in variants:
        cols, wts, muts, key_parts = [], [], [], []
        skip_this = False
        for v in varset:
            _, ref, pos, mt = parse_variant(v)  # 기존 함수 재사용 (A100T / Q100Ter)
            if mt == "*" or (ref is None or pos is None or mt is None):
                logger.warning(f"[MSA score] 종결/비정형 변이 스킵: {v}")
                skip_this = True; break
            pos = int(pos)
            col = ungapped_pos_to_col(query, pos)
            if query[col] != "-" and ref and query[col] != ref:
                logger.warning(f"[MSA score] WT 불일치: {v} (정렬WT={query[col]})")
            cols.append(col); wts.append(ref); muts.append(mt)
            key_parts.append(v)
        if not skip_this:
            varsets.append(("+".join(key_parts), cols, wts, muts))

    results = {k: 0.0 for (k, _, _, _) in varsets}
    if not varsets:
        return results

    # 앙상블
    for _ in range(n_repeats):
        n_take = min(len(seqs), n_context)
        idxs = [0] + (random.sample(range(1, len(seqs)), k=n_take-1) if len(seqs) > 1 else [])
        msa_sub = [seqs[i] for i in idxs]

        # 토큰화 (B=1, Nseq, L+1(BOS))
        _, _, tokens = batch_converter([("msa", msa_sub)])
        tokens = tokens.to(device)

        # 마스킹: 평가할 모든 열을 첫 행에서만 마스크
        tokens_masked = tokens.clone()
        cols_all = sorted({c for (_, cols, _, _) in varsets for c in cols})
        for c in cols_all:
            tokens_masked[0, 0, c + 1] = mask_idx

        out = model(tokens_masked)
        lprobs = out["logits"].log_softmax(-1)  # (1, N, L+1, |V|)

        for (key, cols, wts, muts) in varsets:
            delta = 0.0
            for c, wt, mt in zip(cols, wts, muts):
                try:
                    wt_idx = alphabet.get_idx(wt); mt_idx = alphabet.get_idx(mt)
                except KeyError:
                    logger.warning(f"[MSA score] 알파벳 미지원 문자 스킵: {wt}->{mt}")
                    continue
                lp = lprobs[0, 0, c + 1]
                delta += (lp[mt_idx] - lp[wt_idx]).item()
            results[key] += delta

    # 평균
    for k in results:
        results[k] /= max(1, n_repeats)
    return results



# --- main ---
def build_msa_for_uniprot(
    uniprot_id: str,
    db_prefix: str | None = None,
    msa_dir: str = os.path.join(DATA_DIR, "msa"),
    n_iter: int = 3,
    evalue: float = 1e-3,
    cpu: int = 8,
    max_id: int = 90,
    min_cov: int = 75,
    diff: int = 1000,
    force: bool = False,
    run_alistat: bool = False,  # QC가 필요하면 True (esl-alistat 필요)
) -> str:
    """
    반환: 최종 ESM 친화적 A3M 경로: <msa_dir>/<uniprot_id>/<uniprot_id>.esm.a3m
    """
    # 0) 바이너리 확인
    _which_or_die("hhblits"); _which_or_die("hhfilter"); _which_or_die("reformat.pl")
    if db_prefix is None:
        db_prefix = os.environ.get("HHBLITS_DB_PREFIX")
    if not db_prefix:
        raise ValueError("HHblits DB prefix가 필요합니다. 인자 db_prefix 또는 환경변수 HHBLITS_DB_PREFIX 설정")

    out_base = Path(msa_dir).expanduser().resolve() / uniprot_id
    out_base.mkdir(parents=True, exist_ok=True)

    # 1) 캐시 확인
    esm_a3m = out_base / f"{uniprot_id}.esm.a3m"
    if esm_a3m.exists() and not force:
        return str(esm_a3m)

    # 2) 서열 → query.fa
    seq = get_protein_sequence(uniprot_id)
    if not seq:
        raise ValueError(f"UniProt 서열을 가져올 수 없음: {uniprot_id}")
    seq = _sanitize_seq(seq)
    if not seq:
        raise ValueError("정제 후 시퀀스 길이가 0")

    query_fa = out_base / "query.fa"
    with open(query_fa, "w") as f:
        f.write(f">{uniprot_id}\n")
        for i in range(0, len(seq), 60):
            f.write(seq[i:i+60] + "\n")

    # 3) HHblits → raw.a3m
    raw_a3m = out_base / f"{uniprot_id}.raw.a3m"
    _run([
        "hhblits",
        "-i", str(query_fa),
        "-d", db_prefix,            # 확장자 없는 프리픽스
        "-oa3m", str(raw_a3m),
        "-n", str(n_iter),
        "-e", str(evalue),
        "-cpu", str(cpu),
    ])

    # 4) HHfilter → filt.a3m
    filt_a3m = out_base / f"{uniprot_id}.filt.a3m"
    _run([
        "hhfilter",
        "-i", str(raw_a3m),
        "-o", str(filt_a3m),
        "-id", str(max_id),
        "-cov", str(min_cov),
        "-diff", str(diff),
    ])

    # 5) reformat.pl → esm.a3m (삽입 제거 & 열 정합)
    _run([
        "reformat.pl", "a3m", "a3m",
        str(filt_a3m), str(esm_a3m),
        "-M", "first",  # 첫 서열 기준
        "-r",           # 삽입(lowercase) 제거
    ])

    # 6) (선택) QC
    if run_alistat and shutil.which("esl-alistat"):
        qc = _run(["esl-alistat", str(esm_a3m)])
        with open(out_base / f"{uniprot_id}.alistat.txt", "w") as f:
            f.write(qc.stdout)

    return str(esm_a3m)


"""
MSA 생성 파이프라인 (HHblits → hhfilter → reformat) + 자동 QC + 클리닝 + 재-QC

요약:
- build_msa_for_uniprot(uniprot_id, ...) -> str
  - <msa_dir>/<uniprot_id>/<uniprot_id>.esm.a3m 생성
  - 1차 QC (strict)
  - 필요 시 hhfilter 완화 재시도 → raw 직행 구제
  - 성공 시 자동 클리닝(*.esm.clean.a3m) + 재-QC
  - 최종적으로 clean이 OK면 clean 경로 반환, 아니면 원본 esm.a3m 반환

필수 의존(외부에 정의돼 있어야 함):
- _run(cmd: List[str]) -> subprocess.CompletedProcess-like (stdout 속성 사용)
- _which_or_die(name: str) -> str
- _sanitize_seq(seq: str) -> str
- get_protein_sequence(uniprot_id: str) -> str | None
"""


import os
import re
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

# =============================
# 설정 상수
# =============================
AA_STD_UPPER_GAP = set("ACDEFGHIKLMNPQRSTVWY-")  # QC에서 허용(비표준은 fatal)
AA_KEEP_FOR_CLEAN = set("ACDEFGHIKLMNPQRSTVWY-") # 클리닝 시 유지(그 외는 '-'로 마스킹)
GAP = "-"

# HHblits 민감도 강화 기본값
HHBLITS_EXTRA = {
    "maxfilt": "1000000",
    "all": True,
    "B": "100000",
    "Z": "100000",
    "mact": "0.2",  # default 0.35보다 완화
}

# 초기 hhfilter(완화)
HHFILTER_LOOSE = {"id": "99", "cov": "60"}
# 재시도 hhfilter(더 완화)
HHFILTER_VERY_LOOSE = {"id": "100", "cov": "50"}

# 클리닝 파라미터 기본값
DEFAULT_MAX_GAP_FRAC_PER_COL = 0.90
DEFAULT_MIN_COV_TO_MASTER = 0.25
REPLACE_NONSTD_WITH_GAP = True

# =============================
# A3M 파서/유틸
# =============================
def _parse_a3m_text(a3m_text: str) -> Tuple[List[str], List[str]]:
    names, seqs, name, buf = [], [], None, []
    for line in a3m_text.splitlines():
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                seqs.append("".join(buf)); buf = []
            name = line[1:].strip(); names.append(name)
        else:
            buf.append(line.strip())
    if name is not None:
        seqs.append("".join(buf))
    return names, seqs

def _strip_insertions(a3m_seq: str) -> str:
    # A3M 소문자(삽입) 제거
    return re.sub(r"[a-z]", "", a3m_seq)

def _pid(seqA: str, seqB: str) -> float:
    same, comp = 0, 0
    for a, b in zip(seqA, seqB):
        if a != GAP and b != GAP:
            comp += 1
            if a == b:
                same += 1
    return 0.0 if comp == 0 else same / comp

def _meff62(seqs: List[str], sample_cap: int = 400, pid_thr: float = 0.62) -> float:
    """샘플링 기반 Meff(62%) 근사치"""
    N = len(seqs)
    if N == 0:
        return 0.0
    if N > sample_cap:
        idx = sorted(random.sample(range(N), sample_cap))
        sub = [seqs[i] for i in idx]
    else:
        sub = seqs
    M = len(sub)
    w = np.zeros(M, dtype=np.float64)
    for i in range(M):
        cnt = 0
        si = sub[i]
        for j in range(M):
            if _pid(si, sub[j]) >= pid_thr:
                cnt += 1
        w[i] = 1.0 / cnt
    return float(w.sum()) * (N / M)

# =============================
# QC
# =============================
def qc_a3m(a3m_text: str, strict: bool = True) -> Dict[str, Any]:
    """
    A3M 텍스트를 QC하여 dict 반환.
    strict=True이면 비표준 문자/마스터 소문자/열 길이 불일치 등은 fatal.
    """
    names, seqs_raw = _parse_a3m_text(a3m_text)
    res: Dict[str, Any] = {"ok": True, "warnings": [], "n_seq_raw": len(seqs_raw)}

    if len(seqs_raw) == 0:
        res["ok"] = False
        res["warnings"].append(("fatal", "빈 A3M (시퀀스 0)"))
        return res

    # 마스터 소문자 검사
    if re.search(r"[a-z]", seqs_raw[0]):
        res["warnings"].append(("fatal" if strict else "warn", "마스터 행(1행)에 소문자(삽입) 존재"))

    # 삽입 제거 후 길이 정합
    seqs = [_strip_insertions(s) for s in seqs_raw]
    Ls = list(map(len, seqs))
    res["L_after_strip"] = int(Ls[0])
    if len(set(Ls)) != 1:
        res["ok"] = False
        res["warnings"].append(("fatal", "삽입 제거 후에도 열 길이 불일치"))
        res["lengths_first10"] = Ls[:10]
        return res

    # 비표준 문자 검사 (X 포함)
    bad_idx = []
    for i, s in enumerate(seqs[:200]):
        bad = set(s) - AA_STD_UPPER_GAP
        if bad:
            bad_idx.append((i, ''.join(sorted(bad))))
    if bad_idx:
        res["warnings"].append(("fatal" if strict else "warn", f"비표준 문자 발견 (예: {bad_idx[:5]})"))

    # 통계
    n = len(seqs); L = len(seqs[0])
    arr = np.frombuffer("".join(seqs).encode("ascii"), dtype="S1").reshape(n, L)
    is_gap = (arr == b"-")
    gap_per_col = is_gap.mean(axis=0)
    gap_per_row = is_gap.mean(axis=1)

    res.update({
        "n_seq": n, "L": L,
        "gap_frac_col_mean": float(gap_per_col.mean()),
        "gap_frac_col_med": float(np.median(gap_per_col)),
        "gap_frac_col>0.5": float((gap_per_col > 0.5).mean()),
        "gap_frac_row_mean": float(gap_per_row.mean()),
        "gap_frac_row_med": float(np.median(gap_per_row)),
    })

    # 마스터 대비 커버리지
    master = "".join(arr[0].astype(str))
    mask_master = np.array([c != "-" for c in master], dtype=bool)
    covs = []
    denom = mask_master.sum()
    for r in arr:
        s = "".join(r.astype(str))
        arr_row = np.array([c != "-" for c in s], dtype=bool)
        num = (mask_master & arr_row).sum()
        covs.append(0.0 if denom == 0 else num / denom)
    covs = np.array(covs, dtype=np.float64)
    res.update({
        "coverage_to_master_mean": float(covs.mean()),
        "coverage_to_master_med": float(np.median(covs)),
        "coverage_to_master_p10": float(np.percentile(covs, 10)),
        "coverage_to_master_p90": float(np.percentile(covs, 90)),
    })

    # 휴리스틱 경고
    if n < 2:
        res["warnings"].append(("warn", "시퀀스 수 1개 → 실질적으로 MSA 아님"))
    if res["gap_frac_col_mean"] > 0.5:
        res["warnings"].append(("warn", f"열 평균 갭 비율 높음: {res['gap_frac_col_mean']:.3f}"))
    if res["coverage_to_master_mean"] < 0.6:
        res["warnings"].append(("warn", f"마스터 대비 평균 커버리지 낮음: {res['coverage_to_master_mean']:.3f}"))

    # Meff 근사
    try:
        seqs_str = ["".join(r.astype(str)) for r in arr]
        res["meff62_est"] = _meff62(seqs_str, sample_cap=400, pid_thr=0.62)
        if res["meff62_est"] < 10:
            res["warnings"].append(("info", f"Meff 낮음(≈{res['meff62_est']:.1f}) → 정보량 부족 가능"))
    except Exception as e:
        res["warnings"].append(("info", f"Meff 계산 실패: {e!r}"))

    if any(lvl == "fatal" for (lvl, _msg) in res["warnings"]):
        res["ok"] = False
    return res

def pretty_print_qc(res: Dict[str, Any], tag: str = "") -> None:
    head = f"[A3M QC] {'OK' if res.get('ok') else 'FAIL'}"
    if tag:
        head += f" :: {tag}"
    print(head)
    for k in [
        "n_seq_raw", "n_seq", "L", "L_after_strip",
        "gap_frac_col_mean", "gap_frac_col_med", "gap_frac_col>0.5",
        "gap_frac_row_mean", "gap_frac_row_med",
        "coverage_to_master_mean", "coverage_to_master_med",
        "coverage_to_master_p10", "coverage_to_master_p90",
        "meff62_est",
    ]:
        if k in res:
            v = res[k]
            if isinstance(v, float):
                print(f"  - {k}: {v:.4f}")
            else:
                print(f"  - {k}: {v}")
    if "lengths_first10" in res:
        print(f"  - lengths_first10: {res['lengths_first10']}")
    if res.get("warnings"):
        print("  - warnings:")
        for lvl, msg in res["warnings"]:
            print(f"    [{lvl.upper()}] {msg}")

# =============================
# 클리닝
# =============================
def clean_a3m_for_esm(
    a3m_path: str,
    out_path: Optional[str] = None,
    max_gap_frac_per_col: float = DEFAULT_MAX_GAP_FRAC_PER_COL,
    min_cov_to_master: float = DEFAULT_MIN_COV_TO_MASTER,
    replace_nonstd_with_gap: bool = REPLACE_NONSTD_WITH_GAP,
    verbose: bool = False,   # 단계별 길이/통계 로그 on/off
) -> str:
    """
    간단 클리닝:
      - 소문자(삽입) 제거
      - 비표준(X 등) → '-' 마스킹(옵션)
      - 마스터 대비 커버리지 < min_cov_to_master 시퀀스 제거 (row 필터)
      - 열 갭비율 > max_gap_frac_per_col 컬럼 제거 (col 필터)
    """

    def _log_len(tag: str, seqs_list: List[str]) -> None:
        if not verbose: 
            return
        if not seqs_list:
            print(f"[{tag}] sequences=0")
            return
        lens = list(map(len, seqs_list))
        Lmin, Lmax = min(lens), max(lens)
        uniq = len(set(lens))
        print(f"[{tag}] #seq={len(seqs_list)}, L(min/max)={Lmin}/{Lmax}, "
              f"len_unique={uniq}{' (unaligned lengths!)' if uniq>1 else ''}")

    def _log_arr(tag: str, arr: np.ndarray) -> None:
        if not verbose:
            return
        n, L = arr.shape
        gap = (arr == b"-")
        print(f"[{tag}] #seq={n}, L={L}, gap_col_mean={gap.mean(axis=0).mean():.3f}")

    p = Path(a3m_path)
    txt = p.read_text(errors="replace")
    names, seqs = _parse_a3m_text(txt)
    if not seqs:
        raise RuntimeError("clean_a3m_for_esm: 빈 A3M")

    _log_len("load_raw", seqs)

    # 1) 삽입 제거
    seqs = [_strip_insertions(s) for s in seqs]
    _log_len("after_strip_insertions", seqs)

    # 2) 비표준 마스킹 (X 등 → '-')
    if replace_nonstd_with_gap:
        seqs = ["".join(ch if ch in AA_KEEP_FOR_CLEAN else "-" for ch in s) for s in seqs]
        _log_len("after_nonstd_masking", seqs)

    # 3) 길이 정합: 최대 길이에 '-' 패딩 (열 보존)
    L = max(map(len, seqs))
    seqs = [s + ("-" * (L - len(s))) for s in seqs]
    _log_len("after_padding_to_maxlen", seqs)  # 여기서 len_unique는 반드시 1이어야 정상

    # 4) 배열화
    arr = np.frombuffer("".join(seqs).encode("ascii"), dtype="S1").reshape(len(seqs), L)
    _log_arr("array_init", arr)

    # 5) row 필터: 마스터 대비 커버리지
    gap = (arr == b"-")
    master_mask = ~gap[0]
    denom = int(master_mask.sum()) or 1
    cov = ((~gap) & master_mask).sum(axis=1) / denom
    keep_row = cov >= min_cov_to_master
    arr = arr[keep_row, :]
    names = [nm for nm, k in zip(names, keep_row) if k]
    if arr.shape[0] < 1:
        raise RuntimeError("clean_a3m_for_esm: 필터 후 시퀀스 0")
    _log_arr("after_row_filter", arr)

    # # 6) col 필터: 갭 과다 컬럼 제거
    # gap = (arr == b"-")
    # col_gap = gap.mean(axis=0)
    # keep_col = (col_gap <= max_gap_frac_per_col)
    # if keep_col.sum() < 1:
    #     raise RuntimeError("clean_a3m_for_esm: 모든 열 제거됨(갭 과다)")
    # arr = arr[:, keep_col]
    # _log_arr("after_col_filter", arr)

    # 7) 저장
    out_txt = []
    for nm, row in zip(names, arr):
        out_txt.append(f">{nm}")
        line = row.tobytes().decode("ascii")
        for i in range(0, len(line), 60):
            out_txt.append(line[i : i + 60])
    out_txt = "\n".join(out_txt) + "\n"

    if out_path is None:
        out_path = str(p.with_suffix(".clean.a3m"))
    Path(out_path).write_text(out_txt)

    if verbose:
        print(f"[write] {out_path} :: #seq={arr.shape[0]}, L={arr.shape[1]}")
    return out_path


# # =============================
# # 메인: MSA 생성 + 자동 QC/클리닝
# # =============================
# def build_msa_for_uniprot(
#     uniprot_id: str,
#     db_prefix: Optional[str] = None,
#     msa_dir: str = os.path.join(os.getcwd(), "msa"),
#     n_iter: int = 3,
#     evalue: float = 1e-3,
#     cpu: int = 8,
#     max_id: int = 90,   # (사용하지 않음: 보존)
#     min_cov: int = 75,  # (사용하지 않음: 보존)
#     diff: int = 1000,   # (사용하지 않음: 보존)
#     force: bool = False,
#     run_alistat: bool = False,
# ) -> str:
#     """
#     최종 반환: *.esm.clean.a3m (클린 성공 시) 또는 *.esm.a3m
#     """
#     # 필수 바이너리 확인
#     _which_or_die("hhblits"); _which_or_die("hhfilter"); _which_or_die("reformat.pl")

#     if db_prefix is None:
#         db_prefix = os.environ.get("HHBLITS_DB_PREFIX")
#     if not db_prefix:
#         raise ValueError("HHblits DB prefix 필요: 인자 db_prefix 또는 HHBLITS_DB_PREFIX")

#     out_base = Path(msa_dir).expanduser().resolve() / uniprot_id
#     out_base.mkdir(parents=True, exist_ok=True)

#     esm_a3m = out_base / f"{uniprot_id}.esm.a3m"
#     if esm_a3m.exists() and not force:
#         # 이미 있음 → 바로 클린+QC 시도 후 반환
#         return _finalize_with_cleaning(esm_a3m, out_base, uniprot_id, run_alistat)

#     # 1) 쿼리 작성
#     seq = _sanitize_seq(get_protein_sequence(uniprot_id) or "")
#     if not seq:
#         raise ValueError(f"UniProt 서열을 가져올 수 없음 또는 길이 0: {uniprot_id}")
#     query_fa = out_base / "query.fa"
#     with open(query_fa, "w") as f:
#         f.write(f">{uniprot_id}\n")
#         for i in range(0, len(seq), 60):
#             f.write(seq[i : i + 60] + "\n")

#     # 2) HHblits (민감도 강화)
#     raw_a3m = out_base / f"{uniprot_id}.raw.a3m"
#     hhblits_cmd = [
#         "hhblits",
#         "-i", str(query_fa),
#         "-d", str(db_prefix),
#         "-oa3m", str(raw_a3m),
#         "-n", str(n_iter),
#         "-e", str(evalue),
#         "-cpu", str(cpu),
#         "-maxfilt", HHBLITS_EXTRA["maxfilt"],
#         "-all" if HHBLITS_EXTRA["all"] else "",
#         "-B", HHBLITS_EXTRA["B"], "-Z", HHBLITS_EXTRA["Z"],
#         "-mact", HHBLITS_EXTRA["mact"],
#     ]
#     hhblits_cmd = [c for c in hhblits_cmd if c != ""]
#     _run(hhblits_cmd)

#     # 3) 1차 filter (완화)
#     filt_a3m = out_base / f"{uniprot_id}.filt.a3m"
#     _run([
#         "hhfilter",
#         "-i", str(raw_a3m),
#         "-o", str(filt_a3m),
#         "-id", HHFILTER_LOOSE["id"],
#         "-cov", HHFILTER_LOOSE["cov"],
#     ])

#     # 4) reformat (삽입 제거/대문자/첫 행 기준) → esm.a3m
#     _run([
#         "reformat.pl", "a3m", "a3m",
#         str(filt_a3m), str(esm_a3m),
#         "-M", "first",  # 첫 서열 기준
#         "-r",           # 소문자(삽입) 제거
#         "-uc",          # 대문자화
#     ])

#     # 5) QC 1차
#     qc1 = qc_a3m(esm_a3m.read_text(errors="replace"), strict=False)
#     pretty_print_qc(qc1, tag=f"{uniprot_id}.esm.a3m (pass1)")
#     (out_base / f"{uniprot_id}.qc.json").write_text(json.dumps(qc1, ensure_ascii=False, indent=2))

#     if not _good_enough(qc1):
#         # 5-1) 재시도: hhfilter 더 완화
#         print("[QC] 재시도: 필터 완화하여 재생성")
#         filt2_a3m = out_base / f"{uniprot_id}.filt2.a3m"
#         _run([
#             "hhfilter",
#             "-i", str(raw_a3m),
#             "-o", str(filt2_a3m),
#             "-id", HHFILTER_VERY_LOOSE["id"],
#             "-cov", HHFILTER_VERY_LOOSE["cov"],
#         ])
#         _run([
#             "reformat.pl", "a3m", "a3m",
#             str(filt2_a3m), str(esm_a3m),
#             "-M", "first", "-r", "-uc",
#         ])
#         qc2 = qc_a3m(esm_a3m.read_text(errors="replace"), strict=False)
#         pretty_print_qc(qc2, tag=f"{uniprot_id}.esm.a3m (retry1)")
#         (out_base / f"{uniprot_id}.qc.json").write_text(json.dumps(qc2, ensure_ascii=False, indent=2))
#         if _good_enough(qc2):
#             return _finalize_with_cleaning(esm_a3m, out_base, uniprot_id, run_alistat)

#         # 5-2) 최후 구제: filter 스킵(raw→reformat 직행)
#         print("[QC] 최후 구제: filter 스킵 후 raw→reformat 직행")
#         _run([
#             "reformat.pl", "a3m", "a3m",
#             str(raw_a3m), str(esm_a3m),
#             "-M", "first", "-r", "-uc",
#         ])
#         qc3 = qc_a3m(esm_a3m.read_text(errors="replace"), strict=False)
#         pretty_print_qc(qc3, tag=f"{uniprot_id}.esm.a3m (retry2)")
#         (out_base / f"{uniprot_id}.qc.json").write_text(json.dumps(qc3, ensure_ascii=False, indent=2))
#         if _good_enough(qc3):
#             return _finalize_with_cleaning(esm_a3m, out_base, uniprot_id, run_alistat)

        
#         try:
#             print("[QC] 최후 구제2: fatal이어도 클리닝 시도 (X 등 비표준 마스킹)")
#             cleaned_path = clean_a3m_for_esm(
#                 str(esm_a3m),
#                 out_path=str(esm_a3m).replace(".esm.a3m", ".esm.clean.a3m"),
#                 max_gap_frac_per_col=DEFAULT_MAX_GAP_FRAC_PER_COL,  # 0.90
#                 min_cov_to_master=DEFAULT_MIN_COV_TO_MASTER,        # 0.25
#                 replace_nonstd_with_gap=REPLACE_NONSTD_WITH_GAP,    # True (X→'-')
#             )
#             qc_clean = qc_a3m(Path(cleaned_path).read_text(errors="replace"), strict=True)
#             pretty_print_qc(qc_clean, tag=f"{uniprot_id}.esm.clean.a3m (final rescue)")
#             (out_base / f"{uniprot_id}.qc.clean.json").write_text(
#                 json.dumps(qc_clean, ensure_ascii=False, indent=2)
#             )
#             if qc_clean.get("ok") and qc_clean.get("n_seq", 0) >= 2:
#                 if run_alistat and shutil.which("esl-alistat"):
#                     qc = _run(["esl-alistat", str(cleaned_path)])
#                     (out_base / f"{uniprot_id}.alistat.clean.txt").write_text(qc.stdout)
#                 return str(cleaned_path)
#         except Exception as e:
#             print(f"[WARN] 최후 클리닝 구제에서도 예외 발생: {e!r}")

#         # 그래도 실패 → 경고 요약 포함 에러
#         # 최종 실패
#         msgs = [f"[{lvl}] {msg}" for (lvl, msg) in qc3.get("warnings", [])]
#         raise RuntimeError("MSA 생성/QC 실패: " + "; ".join(msgs))

#     # 6) QC 통과 → 클리닝 + 재-QC + 반환
#     return _finalize_with_cleaning(esm_a3m, out_base, uniprot_id, run_alistat)

# =============================
# 내부 헬퍼
# =============================
def _good_enough(res: Dict[str, Any]) -> bool:
    """
    ESM 토큰화 가능 여부 판단:
    - ok == True 이면 통과
    - ok == False라도 fatal이 없고 시퀀스 수>=2면 통과(경고 수준)
    """
    if res.get("ok"):
        return True
    fatals = [m for (lvl, m) in res.get("warnings", []) if lvl == "fatal"]
    return (res.get("n_seq", 0) >= 2) and (len(fatals) == 0)

def _finalize_with_cleaning(esm_a3m: Path, out_base: Path, uniprot_id: str, run_alistat: bool) -> str:
    """
    *.esm.a3m가 준비된 상태에서:
      - 클리닝 → 재-QC
      - strict OK & n_seq>=2 면 clean 반환
      - 아니면 원본 esm.a3m 반환
    """
    try:
        cleaned_path = clean_a3m_for_esm(
            str(esm_a3m),
            out_path=str(esm_a3m).replace(".esm.a3m", ".esm.clean.a3m"),
            max_gap_frac_per_col=DEFAULT_MAX_GAP_FRAC_PER_COL,
            min_cov_to_master=DEFAULT_MIN_COV_TO_MASTER,
            replace_nonstd_with_gap=REPLACE_NONSTD_WITH_GAP,
        )
        qc_clean = qc_a3m(Path(cleaned_path).read_text(errors="replace"), strict=True)
        pretty_print_qc(qc_clean, tag=f"{uniprot_id}.esm.clean.a3m")
        (out_base / f"{uniprot_id}.qc.clean.json").write_text(
            json.dumps(qc_clean, ensure_ascii=False, indent=2)
        )
        if qc_clean.get("ok") and qc_clean.get("n_seq", 0) >= 2:
            if run_alistat and shutil.which("esl-alistat"):
                qc = _run(["esl-alistat", str(cleaned_path)])
                (out_base / f"{uniprot_id}.alistat.clean.txt").write_text(qc.stdout)
            return str(cleaned_path)
        else:
            print("[WARN] 클리닝 후 strict OK는 아님 → 원본 esm.a3m 반환")
    except Exception as e:
        print(f"[WARN] 클리닝 단계에서 예외 발생: {e!r} → 원본 esm.a3m 반환")

    if run_alistat and shutil.which("esl-alistat"):
        qc = _run(["esl-alistat", str(esm_a3m)])
        (out_base / f"{uniprot_id}.alistat.txt").write_text(qc.stdout)
    return str(esm_a3m)