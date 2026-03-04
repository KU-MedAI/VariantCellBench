# GO data crawling
import io  # StringIO를 사용하기 위해 import
from tqdm import tqdm
import requests
import sys, os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from zipfile import ZipFile
import tarfile
from dcor import distance_correlation
from config import BASE_DIR, DATA_DIR, RAW_DATA_PATH, GEARS_DATA_PATH, OUTPUT_DIR
import random



import numpy as np
import anndata as ad

def clean_lowly_expressed_genes(adata: ad.AnnData, max_cells: int = 3, copy: bool = True) -> ad.AnnData:
    """
    전체 cell 중 발현된 cell 수가 `max_cells` 이하(<=)인 유전자를 제거.
    - layers, obsm/varm, varm 등은 AnnData 슬라이싱 규칙에 따라 함께 서브셋됩니다.
    - adata.raw가 있으면 raw도 함께 서브셋합니다.

    # 사용 예시
    # max_cells=3 → 3개 이하에서만 발현된 유전자 제거
    adata_clean = clean_lowly_expressed_genes(adata, max_cells=3, copy=True)
    """
    A = adata.X
    # gene별(열방향) nonzero count
    gene_nonzero = (A > 0).sum(axis=0)
    # sparse면 .A1로 1D numpy array로 변환
    if hasattr(gene_nonzero, "A1"):
        gene_nonzero = gene_nonzero.A1
    else:
        gene_nonzero = np.asarray(gene_nonzero).ravel()

    # keep mask: 발현된 cell 수가 max_cells보다 큰 유전자만 유지
    # (요구사항: <= max_cells 를 제거)
    keep_mask = gene_nonzero > max_cells

    n_drop = (~keep_mask).sum()
    n_keep = keep_mask.sum()
    print(f"[Clean] 제거 대상 유전자 수 (<= {max_cells} cells): {n_drop}")
    print(f"[Clean] 유지 유전자 수: {n_keep} / 총 {adata.n_vars}")

    adt = adata.copy() if copy else adata
    # raw도 함께 서브셋
    if adt.raw is not None:
        # raw의 var_names가 현재 var_names와 동일하다는 가정
        # (보통 동일하지만, 다르면 매칭해서 슬라이싱 필요)
        adt.raw = ad.AnnData(X=adt.raw.X, var=adt.raw.var.copy(), obs=adt.obs.copy())
        adt.raw = adt.raw[:, keep_mask]

    # 본체 서브셋
    adt = adt[:, keep_mask]

    # 정리 후 간단 검증
    assert adt.n_vars == n_keep
    return adt





def fetch_gene_data(gene_name):
    """
    Gene 이름을 이용하여 HGNC, Ensembl, UniProt 정보를 가져옵니다.
    """
    url = f"https://rest.genenames.org/fetch/symbol/{gene_name}"
    headers = {"Accept": "application/json"}

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if data['response']['numFound'] > 0:
            doc = data['response']['docs'][0]
            SYMBOL = doc.get('symbol', 'N/A')
            HGNC_ID = doc.get('hgnc_id', 'N/A')
            ENSEMBL_ID = doc.get('ensembl_gene_id', 'N/A')
            UNIPROT_ID = doc.get('uniprot_ids', 'N/A')
            ALIAS = doc.get('alias_name', 'N/A')

            return SYMBOL, HGNC_ID, ENSEMBL_ID, UNIPROT_ID, ALIAS
        else:
            print(f"No data found for Gene: {gene_name}")
            return None, None, None, None, None
    else:
        print(f"Error fetching data: {response.status_code}")
        return None, None, None, None, None


def fetch_go_data(gene_name):
    """
    Gene Ontology 데이터를 CSV 형식으로 가져와 Pandas DataFrame으로 변환합니다.
    """
    url = f"https://golr-aux.geneontology.io/solr/select?defType=edismax&qt=standard&indent=on&wt=csv&rows=100000&start=0&fl=bioentity_internal_id%2Cbioentity_label%2Cannotation_class%2Cevidence_type%2Caspect%2Csynonym%2Ctaxon%2Cbioentity_isoform&facet=true&facet.mincount=1&facet.sort=count&json.nl=arrarr&facet.limit=25&hl=true&hl.simple.pre=%3Cem%20class%3D%22hilite%22%3E&hl.snippets=1000&csv.encapsulator=&csv.separator=%09&csv.header=false&csv.mv.separator=%7C&fq=document_category:%22annotation%22&fq=taxon_subset_closure_label:%22Homo%20sapiens%22&facet.field=aspect&facet.field=taxon_subset_closure_label&facet.field=type&facet.field=evidence_subset_closure_label&facet.field=regulates_closure_label&facet.field=isa_partof_closure_label&facet.field=annotation_class_label&facet.field=qualifier&facet.field=annotation_extension_class_closure_label&facet.field=assigned_by&facet.field=panther_family_label&q={gene_name}*&qf=annotation_class%5E2&qf=annotation_class_label_searchable%5E1&qf=bioentity%5E2&qf=bioentity_label_searchable%5E1&qf=bioentity_name_searchable%5E1&qf=annotation_extension_class%5E2&qf=annotation_extension_class_label_searchable%5E1&qf=reference_searchable%5E1&qf=panther_family_searchable%5E1&qf=panther_family_label_searchable%5E1&qf=bioentity_isoform%5E1&qf=isa_partof_closure%5E1&qf=isa_partof_closure_label_searchable%5E1"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.content.decode('utf-8')

        # 데이터가 비어있는 경우 빈 DataFrame 반환
        if not data.strip():
            print(f"No GO data found for gene: {gene_name}")
            return pd.DataFrame(columns=["bioentity_internal_id", "bioentity_label", "annotation_class",
                                         "evidence_type", "aspect", "synonym", "bioentity_isoform", "taxon"])

        df = pd.read_csv(io.StringIO(data), sep="\t", header=None)
        df.columns = ["bioentity_internal_id", "bioentity_label", "annotation_class",
                      "evidence_type", "aspect", "synonym", "bioentity_isoform", "taxon"]
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for gene {gene_name}: {e}")
        return pd.DataFrame(columns=["bioentity_internal_id", "bioentity_label", "annotation_class",
                                     "evidence_type", "aspect", "synonym", "bioentity_isoform", "taxon"])
    except pd.errors.EmptyDataError:
        print(f"No GO data available for gene: {gene_name}")
        return pd.DataFrame(columns=["bioentity_internal_id", "bioentity_label", "annotation_class",
                                     "evidence_type", "aspect", "synonym", "bioentity_isoform", "taxon"])


def create_synonym_to_symbol_mapping(symbol, alias):
    """
    synonym 컬럼과 ALIAS 데이터를 기반으로 synonym -> symbol 매핑 사전을 생성합니다.
    :param df: GO 데이터가 포함된 DataFrame
    :param symbol: Gene Symbol
    :param alias: ALIAS 데이터 (문자열 또는 리스트)
    :return: synonym -> symbol 매핑 사전
    """
    synonym_to_symbol = {}

    # ALIAS 데이터를 '|'로 분리 (ALIAS가 문자열로 제공될 경우)
    if alias and isinstance(alias, list):
        aliases = alias
    else:
        aliases = []

    # synonym -> symbol 매핑 생성
    for synonym in set(aliases):
        synonym_to_symbol[synonym] = symbol

    return synonym_to_symbol


def create_gene2GO_dictionary(gene_name):
    """
    Gene 이름을 기반으로 gene2GO 딕셔너리를 생성합니다.
    """

    # HGNC Gene 데이터 가져오기
    SYMBOL, HGNC_ID, ENSEMBL_ID, UNIPROT_ID, ALIAS = fetch_gene_data(gene_name)
    if SYMBOL is None:
        print(f"Failed to fetch data for {gene_name}")
        SYMBOL=gene_name

    # GO 데이터 가져오기
    go_data = fetch_go_data(SYMBOL)
    if go_data is not None:
        go_data_filtered = go_data[go_data['bioentity_label'] == SYMBOL]

        # annotation_class 컬럼에서 고유한 GO annotation ID 목록 추출
        go_annotations = go_data_filtered[['annotation_class','aspect']]
        BP = set(go_annotations[go_annotations['aspect']=="P"]['annotation_class'].unique())
        MF = set(go_annotations[go_annotations['aspect']=="F"]['annotation_class'].unique())
        CC = set(go_annotations[go_annotations['aspect']=="C"]['annotation_class'].unique())
        # go_annotations = set(go_data_filtered[['annotation_class','aspect']].unique())

        # gene2GO dictionary 생성
        gene2GO = {
            gene_name: {
                "HGNC_ID": HGNC_ID,
                "ENSEMBL_ID": ENSEMBL_ID,
                "UNIPROT_ID": UNIPROT_ID,
                "annotation_class": {
                    "BP":BP,
                    "MF":MF,
                    "CC":CC
                }
            }
        }
        return gene2GO
    else:
        print(f"No GO data found for {gene_name}")
        return None


def process_gene_list_with_existing_dict(gene_list, gene2GO_dict, synonym_to_symbol_dict):
    """
    기존 dictionary를 활용하여 Gene 이름 리스트를 처리하고, 처리되지 않은 Gene에 대해서만 데이터를 추가합니다.
    
    :param gene_list: Gene 이름 리스트 (예: ['TP53', 'KRAS'])
    :param gene2GO_dict: 기존 gene2GO dictionary
    :param synonym_to_symbol_dict: 기존 synonym_to_symbol dictionary
    :return: 업데이트된 gene2GO_dict와 synonym_to_symbol_dict
    """
    i = 0
    for gene_name in tqdm(gene_list):
        if gene_name in gene2GO_dict:
            print(f"Skipping already processed gene: {gene_name}")
            continue  # 이미 처리된 Gene은 건너뜀

        print(f"Processing gene: {gene_name}...")

        # Gene 및 GO 데이터 가져오기
        SYMBOL, HGNC_ID, ENSEMBL_ID, UNIPROT_ID, ALIAS = fetch_gene_data(gene_name)
        if SYMBOL is not None:
            # Gene에 대한 Symbol이 있는 경우에 Symbol로 GO-term 검색
            go_data = fetch_go_data(SYMBOL)
        else:
            # Gene에 대한 Symbol이 없는 경우에 Gene name으로 GO-term 검색
            go_data = fetch_go_data(gene_name)

        if go_data is not None and SYMBOL is not None:
            """GO-term 있음, Symbol 있음"""
            # Synonym to Symbol Mapping 생성
            synonym_to_symbol = create_synonym_to_symbol_mapping(SYMBOL, ALIAS)
            
            # Synonym to Symbol 사전 병합
            synonym_to_symbol_dict.update(synonym_to_symbol)

            # Gene2GO Dictionary 생성
            gene2GO = create_gene2GO_dictionary(SYMBOL)
            if gene2GO:
                gene2GO_dict.update(gene2GO)
            print(f"Successed process gene SYMBOL: {SYMBOL}")
        elif go_data is None and SYMBOL is not None:
            """GO-term 없음, Symbol 있음"""
            # Synonym to Symbol Mapping 생성
            synonym_to_symbol = create_synonym_to_symbol_mapping(SYMBOL, ALIAS)
            
            # Synonym to Symbol 사전 병합
            synonym_to_symbol_dict.update(synonym_to_symbol)

            # Gene2GO Dictionary 생성
            gene2GO = create_gene2GO_dictionary(SYMBOL)
            if gene2GO:
                gene2GO_dict.update(gene2GO)
            print(f"Successed process gene SYMBOL: {SYMBOL}")

        elif go_data is not None and SYMBOL is None:
            """GO-term 있음, Symbol 없음"""
            synonym_to_symbol = create_synonym_to_symbol_mapping(gene_name, ALIAS)
            
            # Synonym to Symbol 사전 병합
            synonym_to_symbol_dict.update(synonym_to_symbol)

            # Gene2GO Dictionary 생성
            gene2GO = create_gene2GO_dictionary(gene_name)
            if gene2GO:
                gene2GO_dict.update(gene2GO)
            print(f"Successed process gene name: {gene_name}")

        else:
            print(f"Failed to process gene: {gene_name}")

        i = i + 1

        # 주기적으로 저장
        if i % 100 == 0:
            with open('/NFS_DATA/apocalypse/database/gene2GO_dict.pkl', 'wb') as f:
                pickle.dump(gene2GO_dict, f, pickle.HIGHEST_PROTOCOL)
            with open('/NFS_DATA/apocalypse/database/synonym_to_symbol_dict.pkl', 'wb') as f:
                pickle.dump(synonym_to_symbol_dict, f, pickle.HIGHEST_PROTOCOL)

    return gene2GO_dict, synonym_to_symbol_dict

# numpy-free [fixed issue {250802-1}]: numpy._core version mismatch, numpy-free data generation process
def get_gene_condition_list(condition):
    """
    주어진 perturbation 리스트에서 SNV 변이(gene~variant)와 일반 유전자 perturbation을 분리하여,
    SNV에 포함된 고유한 유전자 이름 리스트를 반환한다.
    
    Parameters
    ----------
    condition : list of str
        Perturbation condition list (e.g., ['TP53~R175H', 'KRAS', 'TP53~R248Q'])

    Returns
    -------
    list of str
        Unique gene names from SNV conditions
    """

    variants = [item for item in condition if '~' in item]
    snv_gene_list = [p.split('~')[0] for p in set(variants)]  # set으로 중복 제거
    return sorted(set(snv_gene_list))

# # original [fixed issue {250802-1}]: numpy._core version mismatch, numpy-free data generation process
# def get_gene_condition_list(condition):
#     """
#     주어진 perturbation 리스트에서 SNV 변이와 유전자 perturbation을 분리하여 목록화한다.
#     """
#     variants = [item for item in condition if '~' in item]  # '~'가 포함된 항목은 SNV 변이로 간주하여 추출
#     snv_gene_list = [p.split('~')[0] for p in np.unique(variants)]  # SNV에서 gene 이름만 추출하여 리스트로 저장
#     genes_list = [item for item in condition if '~' not in item]  # perturbation (유전자 조작) 항목만 분리
#     return  np.unique(snv_gene_list)
#     # 해당 함수는 현재 반환값이 없음 (향후 리스트 반환 또는 저장 용도로 사용될 가능성 있음)




# numpy-free [fixed issue {250802-1}]: numpy._core version mismatch, numpy-free data generation process
def get_condition_lists(condition):
    """
    Returns list of unique genes involved in a given perturbation list (numpy-free)

    Parameters
    ----------
    condition : str or list of str
        Perturbation conditions (e.g., ['TP53', 'KRAS+EGFR'])

    Returns
    -------
    list
        Unique list of genes excluding 'ctrl'
    """
    if isinstance(condition, str):
        condition = [condition]

    # split by '+', flatten, and filter out 'ctrl'
    condition_list = []
    for cond in set(condition):  # unique 처리
        condition_list.extend(g for g in cond.split('+') if g != 'ctrl')

    return sorted(set(condition_list))  # 유일한 유전자 이름, 정렬된 리스트 반환

# # original [fixed issue {250802-1}]: numpy._core version mismatch, numpy-free data generation process
# def get_condition_lists(condition):
#     """
#     Returns list of genes involved in a given perturbation list
#     """
#     """ 
#     주어진 condition 문자열 또는 리스트에서 ctrl을 제외한 모든 유전자 이름을 분리하고, 유일한 값 리스트로 반환한다.
#     """

#     if type(condition) is str:  # 입력이 문자열이면 리스트로 변환
#         condition = [condition]
#     condition_list = [p.split('+') for p in np.unique(condition)]  # '+' 기준으로 분할하여 리스트화
#     condition_list = [item for sublist in condition_list for item in sublist]  # 중첩 리스트를 평탄화
#     condition_list = [g for g in condition_list if g != 'ctrl']  # 'ctrl' 항목 제거
    
#     return  np.unique(condition_list)  # 유일한 유전자 이름 리스트 반환




def parse_condition(p):
    """
    주어진 condition 문자열이 단일(ctrl 포함)인지, 이중 조합인지 판별하여 유전자 이름을 리스트로 반환한다.
    """
    # single gene perturbation
    # 단일 condition
    if ('ctrl' in p) and (p != 'ctrl'):  # 'ctrl'이 포함되었으나 단일 condition일 경우
        a = p.split('+')[0]  # 첫 번째 perturbation 추출
        b = p.split('+')[1]  # 두 번째 perturbation 추출
        if a == 'ctrl':
            pert = b  # ctrl이 앞에 있으면 뒤가 pert
        else:
            pert = a  # ctrl이 뒤에 있으면 앞이 pert
        return [pert]  # pert 리스트 반환

    # dual gene perturbation
    # 이중 condition
    elif 'ctrl' not in p:  # ctrl이 포함되지 않은 경우, 두 유전자 조합
        return [p.split('+')[0], p.split('+')[1]]  # 각각 분리하여 리스트 반환


def get_condition_from_genes(genes, pert_list, type_='both'):
    """
    Returns all single/bungle/both perturbations that include a gene
    """
    """ 
    주어진 유전자 목록과 perturbation 리스트를 바탕으로, 해당 유전자를 포함하는 condition(단일/이중/모두)을 반환한다.
    """

    single_perts = [p for p in pert_list if ('ctrl' in p) and (p != 'ctrl')]  # 단일 perturbation (ctrl 포함)
    bungle_perts = [p for p in pert_list if 'ctrl' not in p]  # 이중 perturbation (ctrl 없음)

    perts = []  # 결과를 저장할 리스트 초기화

    if type_ == 'single':  # 단일 조건만 필터링
        pert_candidate_list = single_perts
    elif type_ == 'bungle':  # 이중 조건만 필터링
        pert_candidate_list = bungle_perts
    elif type_ == 'both':  # 모든 perturbation 포함
        pert_candidate_list = pert_list

    for p in pert_candidate_list:  # 후보 perturbation 순회
        for g in genes:  # 주어진 유전자 리스트 순회
            if g in parse_condition(p):  # 해당 유전자가 현재 perturbation에 포함되어 있으면
                perts.append(p)  # 결과 리스트에 추가
                break  # 중복 방지를 위해 내부 루프 종료

    return perts  # 유전자가 포함된 perturbation 리스트 반환



# numpy-free [fixed issue {250802-1}]: numpy._core version mismatch, numpy-free data generation process
def split_conditions_train_valid_test(condition_list, split_ratio=0.85, oo_ratio=0.85, seed=1, split_for='test'):
    random.seed(seed)  # numpy 대신 Python random 사용
    
    # 전체 condition 중 perturbed gene 목록
    all_genes = get_condition_lists(condition_list)
    all_genes = list(all_genes)

    # train / test 유전자를 분할
    train_genes = random.sample(all_genes, int(len(all_genes) * split_ratio))
    test_genes = list(set(all_genes) - set(train_genes))

    # 단일 perturbation 조건
    train_single = get_condition_from_genes(train_genes, condition_list, type_='single')
    test_single_x = get_condition_from_genes(test_genes, condition_list, type_='single')

    # 이중 perturbation 조건 (seen1, seen2, unseen0 분리)
    double_conditions_all = get_condition_from_genes(train_genes, condition_list, type_='bungle')

    # seen 1개 (train gene 중 1개 포함)
    test_double_ox = [
        cond for cond in double_conditions_all
        if sum(gene in train_genes for gene in cond.split('+')) == 1
    ]

    # seen2만 남기기
    double_seen2 = [cond for cond in double_conditions_all if cond not in test_double_ox]

    # seen2 중 일부를 train, 나머지는 test
    n_train_double_oo = int(len(double_seen2) * oo_ratio)
    train_double_oo = random.sample(double_seen2, n_train_double_oo)
    test_double_oo = [cond for cond in double_seen2 if cond not in train_double_oo]

    # test_genes 기반 unseen (seen0)
    test_double_all_x = get_condition_from_genes(test_genes, condition_list, type_='bungle')
    test_double_xx = [
        cond for cond in test_double_all_x
        if all(gene not in train_genes for gene in cond.split('+'))
    ]

    # 최종 반환 리스트 구성
    condition_train = list(train_single) + list(train_double_oo)
    condition_test = list(test_single_x) + list(test_double_ox) + list(test_double_oo) + list(test_double_xx)

    return condition_train, condition_test, {
        f'{split_for}_condition_single_x': list(test_single_x),
        f'{split_for}_condition_double_ox': list(test_double_ox),
        f'{split_for}_condition_double_oo': list(test_double_oo),
        f'{split_for}_condition_double_xx': list(test_double_xx),
    }

# # original [fixed issue {250802-1}]: numpy._core version mismatch, numpy-free data generation process
# def split_conditions_train_valid_test(condition_list, split_ratio=0.85, oo_ratio=0.85, seed=1, split_for='test'):
#     np.random.seed(seed=seed)
#     # 전체 condition 중 perturbed gene 목록
#     all_genes = get_condition_lists(condition_list)

#     # train / test 유전자를 분할
#     train_genes = np.random.choice(
#         all_genes, int(len(all_genes) * split_ratio), replace=False
#     )
#     test_genes = np.setdiff1d(all_genes, train_genes)

#     # 단일 perturbation 조건
#     train_single = get_condition_from_genes(train_genes, condition_list, type_='single')
#     test_single_x = get_condition_from_genes(test_genes, condition_list, type_='single')

#     # 이중 perturbation 조건 (seen1, seen2, unseen0 분리)
#     double_conditions_all = get_condition_from_genes(train_genes, condition_list, type_='bungle')

#     # seen 1개 (train gene 중 1개 포함)
#     test_double_ox = [
#         cond for cond in double_conditions_all
#         if sum(gene in train_genes for gene in cond.split('+')) == 1
#     ]

#     # seen 2개만 남기기
#     double_seen2 = np.setdiff1d(double_conditions_all, test_double_ox)

#     # seen2 중 일부를 train, 나머지는 test
#     train_double_oo = np.random.choice(
#         double_seen2, int(len(double_seen2) * oo_ratio), replace=False
#     )
#     test_double_oo = np.setdiff1d(double_seen2, train_double_oo)

#     # test_genes 기반 unseen (seen0)
#     test_double_all_x = get_condition_from_genes(test_genes, condition_list, type_='bungle')
#     test_double_xx = [
#         cond for cond in test_double_all_x
#         if all(gene not in train_genes for gene in cond.split('+'))
#     ]

#     # 최종 반환 리스트 구성
#     condition_train = list(train_single) + list(train_double_oo)
#     condition_test = list(test_single_x) + list(test_double_ox) + list(test_double_oo) + list(test_double_xx)

#     return condition_train, condition_test, {
#         f'{split_for}_condition_single_x': test_single_x,
#         f'{split_for}_condition_double_ox': test_double_ox,
#         f'{split_for}_condition_double_oo': test_double_oo.tolist(),
#         f'{split_for}_condition_double_xx': test_double_xx,
#     }

# # original
# def split_conditions_train_valid_test(condition_list, split_ratio=0.85, oo_ratio=0.85, seed=1123, split_for='test'):
#     np.random.seed(seed=seed)
#     # 전체 condition 중 perturbed gene 목록
#     all_genes = get_condition_lists(condition_list)

#     # train / test 유전자를 분할
#     train_genes = np.random.choice(
#         all_genes, int(len(all_genes) * split_ratio), replace=False
#     )
#     test_genes = np.setdiff1d(all_genes, train_genes)

#     # 단일 perturbation 조건
#     train_single = get_condition_from_genes(train_genes, condition_list, type_='single')
#     test_single_x = get_condition_from_genes(test_genes, condition_list, type_='single')

#     # 이중 perturbation 조건 (seen1, seen2, unseen0 분리)
#     double_conditions_all = get_condition_from_genes(train_genes, condition_list, type_='bungle')

#     # seen 1개 (train gene 중 1개 포함)
#     test_double_ox = [
#         cond for cond in double_conditions_all
#         if sum(gene in train_genes for gene in cond.split('+')) == 1
#     ]

#     # seen 2개만 남기기
#     double_seen2 = np.setdiff1d(double_conditions_all, test_double_ox)

#     # seen2 중 일부를 train, 나머지는 test
#     train_double_oo = np.random.choice(
#         double_seen2, int(len(double_seen2) * oo_ratio), replace=False
#     )
#     test_double_oo = np.setdiff1d(double_seen2, train_double_oo)

#     # test_genes 기반 unseen (seen0)
#     test_double_all_x = get_condition_from_genes(test_genes, condition_list, type_='bungle')
#     test_double_xx = [
#         cond for cond in test_double_all_x
#         if all(gene not in train_genes for gene in cond.split('+'))
#     ]

#     # 최종 반환 리스트 구성
#     condition_train = list(train_single) + list(train_double_oo)
#     condition_test = list(test_single_x) + list(test_double_ox) + list(test_double_oo) + list(test_double_xx)

#     return condition_train, condition_test, {
#         f'{split_for}_condition_single_x': test_single_x,
#         f'{split_for}_condition_double_ox': test_double_ox,
#         f'{split_for}_condition_double_oo': test_double_oo.tolist(),
#         f'{split_for}_condition_double_xx': test_double_xx,
#     }

# numpy-free [fixed issue {250802-1}]: numpy._core version mismatch, numpy-free data generation process
def get_pert_idx(pert_category, pert_names):
    """
    Get perturbation index for a given perturbation category (numpy-free)

    Parameters
    ----------
    pert_category : str
        Perturbation condition string (e.g., 'TP53', 'TP53+KRAS', 'TP53~R175H+EGFR~L858R')
    pert_names : list of str
        List of known gene or variant-level perturbation names

    Returns
    -------
    tuple
        (pert_idx: list of int or None, pert_: list of str or None)
    """
    if '~' not in pert_category:  # gene-level perturbation
        try:
            pert_ = [p for p in pert_category.split('+') if p != 'ctrl']
            pert_idx = [pert_names.index(p) for p in pert_]
        except ValueError as e:
            print(f"[get_pert_idx] Error locating gene in pert_names: {pert_category} - {e}")
            pert_ = None
            pert_idx = None
        return pert_idx, pert_

    else:  # variant-level perturbation (e.g., 'TP53~R175H+EGFR~L858R')
        try:
            pert_ = [p for p in pert_category.split('+') if p != 'ctrl']
            genelv_pert_category = [p.split('~')[0] for p in pert_]
            variantlv_pert_category = [p.split('~')[1] for p in pert_]
            genelv_pert_idx = [pert_names.index(gene) for gene in genelv_pert_category]
            # variantlv_pert_idx = [variant_names.index(var) for var in variantlv_pert_category]  # If needed
        except ValueError as e:
            print(f"[get_pert_idx] Error parsing gene~variant format: {pert_category} - {e}")
            pert_ = None
            genelv_pert_idx = None
        return genelv_pert_idx, pert_



# # original [fixed issue {250802-1}]: numpy._core version mismatch, numpy-free data generation process
# def get_pert_idx(pert_category, pert_names):
#     """
#     Get perturbation index for a given perturbation category

#     Parameters
#     ----------
#     pert_category: str
#         Perturbation category

#     Returns
#     -------
#     list
#         List of perturbation indices
#     """
#     """ 
#     perturbation 카테고리를 입력 받아 pert_names 내에서 해당 perturbation의 인덱스를 반환하는 함수입니다.
#     Gene 수준과 Gene~variant 수준을 모두 처리합니다.
#     """

#     if '~' not in pert_category:  # gene 수준 perturbation인지 확인
#         try:
#             pert_ = [p for p in pert_category.split('+') if p != 'ctrl']  # 'ctrl'이 아닌 gene 이름 추출
#             pert_idx = [np.where(p == pert_names)[0][0]                  # pert_names 내에서 해당 gene 인덱스 추출
#                         for p in pert_category.split('+') if p != 'ctrl']
#         except:  # 오류 발생 시
#             print(pert_category)  # 오류 원인 출력
#             pert_idx = None       # pert_idx를 None으로 설정
#         return pert_idx, pert_    # 인덱스 및 gene 이름 리스트 반환

#     else:  # variant 수준 perturbation (예: Gene~variant)
#         try:
#             pert_ = [p for p in pert_category.split('+') if p != 'ctrl']  # Gene~variant 리스트 추출
#             genelv_pert_category = [p.split('~')[0] for p in pert_]       # Gene 이름만 추출
#             variantlv_pert_category = [p.split('~')[1] for p in pert_]    # Variant 이름만 추출
#             genelv_pert_idx = [np.where(p == pert_names)[0][0]            # Gene 이름의 인덱스 추출
#                                for p in genelv_pert_category]
#             # variantlv_pert_idx = [...]  # 필요시 variant 인덱스도 추출 가능
#         except Exception as e:  # 예외 발생 시
#             print(f"Error parsing gene~variant format: {pert_category} - {e}")  # 오류 로그 출력
#             genelv_pert_idx = None  # None 처리
#             pert_ = None
#         return genelv_pert_idx, pert_  # 인덱스 및 perturbation 이름 반환

# numpy-free [fixed issue {250802-1}]: numpy._core version mismatch, numpy-free data generation process
def create_cell_graph(X, y, de_idx, de_non_dropout_idx, pert, pert_idx=None, variant=None):
    """
    Create a cell graph from a given cell

    Parameters
    ----------
    X: array-like or torch.Tensor
        Gene expression matrix (1D array or 2D matrix)
    y: array-like or torch.Tensor
        Label vector (perturbed cell)
    de_idx: list or tensor
        DE gene indices
    pert: str
        Perturbation category
    pert_idx: list or tensor
        Perturbation indices (optional)
    variant: list or str
        Variant information (optional)

    Returns
    -------
    torch_geometric.data.Data
        Cell graph to be used in dataloader
    """

    # X: (n_genes, ) or (1, n_genes)
    feature_mat = torch.Tensor(X).T

    # perturbation 인덱스가 주어지지 않으면 기본값 -1 사용
    if pert_idx is None:
        pert_idx = [-1]

    return Data(
        x=feature_mat,
        y=y,
        de_idx=de_idx,
        de_non_dropout_idx=de_non_dropout_idx,
        pert=pert,
        pert_idx=pert_idx,
        variant=variant
    )

# # original [fixed issue {250802-1}]: numpy._core version mismatch, numpy-free data generation process
# def create_cell_graph(X, y, de_idx, pert, pert_idx=None, variant=None):
#     """
#     Create a cell graph from a given cell

#     Parameters
#     ----------
#     X: np.ndarray
#         Gene expression matrix
#     y: np.ndarray
#         Label vector
#     de_idx: np.ndarray
#         DE gene indices
#     pert: str
#         Perturbation category
#     pert_idx: list
#         List of perturbation indices

#     Returns
#     -------
#     torch_geometric.data.Data
#         Cell graph to be used in dataloader
#     """
#     """ 
#     하나의 셀 표현(X, y)을 기반으로 GNN 학습에 사용할 cell graph 오브젝트를 생성합니다.
#     torch_geometric의 Data 객체로 리턴합니다.
#     """

#     feature_mat = torch.Tensor(X).T  # X를 전치하여 gene feature로 변환 (cell마다 하나의 feature vector)
#     if pert_idx is None:            # perturbation 인덱스가 없을 경우
#         pert_idx = [-1]             # default로 -1 설정
#     return Data(x=feature_mat, pert_idx=pert_idx,  # torch_geometric의 Data 객체 생성
#                 y=torch.Tensor(y), de_idx=de_idx, pert=pert, variant=variant)



# numpy-free [fixed issue {250802-1}]: numpy._core version mismatch, numpy-free data generation process
def create_cell_graph_dataset(split_adata, pert_category, num_samples, pert_names, ctrl_adata, gex_layer):
    """
    Combine cell graphs to create a dataset of cell graphs (numpy-free)
    """

    num_de_genes = 20
    num_de_non_dropout_genes = 20  # non_dropout
    adata_ = split_adata[split_adata.obs['condition'] == pert_category]

    if 'rank_genes_groups_cov_all' in adata_.uns:
        de_genes = adata_.uns['rank_genes_groups_cov_all']
        de_non_dropout_genes = adata_.uns['top_non_dropout_de_20']  # non_dropout
        de = True
    else:
        de = False
        num_de_genes = 1

    Xs = []
    ys = []

    if pert_category != 'ctrl':
        if '~' not in pert_category: # gene-level perturbation
            pert_idx, pert_ = get_pert_idx(pert_category, pert_names)
        else:                        # variant-level perturbation
            pert_idx, pert_ = get_pert_idx(pert_category, pert_names)

        pert_de_category = adata_.obs['condition_name'][0]

        if de:
            top_genes = list(de_genes[pert_de_category][:num_de_genes])
            de_idx = [i for i, g in enumerate(adata_.var_names) if g in top_genes]

            top_de_non_dropout_genes = de_non_dropout_genes[pert_de_category][:num_de_genes]  # non_dropout
            de_non_dropout_idx = [i for i, g in enumerate(adata_.var_names) if g in top_de_non_dropout_genes]
        else:
            de_idx = [-1] * num_de_genes
            de_non_dropout_idx = [-1] * num_de_non_dropout_genes # non_dropout

        for cell_z in adata_.X:
            # choose control cells without numpy
            sample_indices = random.choices(range(len(ctrl_adata)), k=num_samples)
            ctrl_samples = ctrl_adata[sample_indices, :]
            if gex_layer == 'norm':
                for c in ctrl_samples.X:
                    Xs.append(c)
                    ys.append(cell_z)
            elif gex_layer == 'counts':
                for c in ctrl_samples.layers[gex_layer]:
                    Xs.append(c)
                    ys.append(cell_z)

    else:
        pert_idx = None
        pert_ = ["ctrl"]

        de_idx = [-1] * num_de_genes
        de_non_dropout_idx = [-1] * num_de_non_dropout_genes
        
        for cell_z in adata_.X:
            Xs.append(cell_z)
            ys.append(cell_z)

    cell_graphs = []
    for X, y in zip(Xs, ys):
        # convert to dense if sparse
        X_tensor = torch.tensor(X.toarray() if hasattr(X, 'toarray') else X, dtype=torch.float32)
        y_tensor = torch.tensor(y.toarray() if hasattr(y, 'toarray') else y, dtype=torch.float32)
        cell_graphs.append(create_cell_graph(X_tensor, y_tensor, de_idx, de_non_dropout_idx, pert_category, pert_idx, pert_))

    return cell_graphs

# # original [fixed issue {250802-1}]: numpy._core version mismatch, numpy-free data generation process
# def create_cell_graph_dataset(split_adata, pert_category, num_samples, pert_names, ctrl_adata):
#     """
#     Combine cell graphs to create a dataset of cell graphs

#     Parameters
#     ----------
#     split_adata: anndata.AnnData
#         Annotated data matrix
#     pert_category: str
#         Perturbation category
#     num_samples: int
#         Number of samples to create per perturbed cell

#     Returns
#     -------
#     list
#         List of cell graphs
#     """
#     """ 
#     특정 perturbation condition에 대한 모든 cell들을 GNN 학습에 사용할 cell graph로 변환하여 리스트로 반환합니다.
#     각 cell은 control sample과 매칭되어 그래프가 생성되며, DE gene index도 포함됩니다.
#     """

#     num_de_genes = 20  # 사용할 DE gene 수 기본값
#     adata_ = split_adata[split_adata.obs['condition'] == pert_category]  # 선택한 condition에 해당하는 샘플 필터링

#     if 'rank_genes_groups_cov_all' in adata_.uns:  # DE gene 정보가 있을 경우
#         de_genes = adata_.uns['rank_genes_groups_cov_all']  # DE gene dictionary 로딩
#         de = True
#     else:  # 없을 경우 fallback 처리
#         de = False
#         num_de_genes = 1  # 기본값

#     Xs = []  # control expression 저장 리스트
#     ys = []  # perturbation expression 저장 리스트

#     if pert_category != 'ctrl':  # control이 아닌 perturbation인 경우
#         if '~' not in pert_category:  # 일반적인 gene perturbation
#             pert_idx, pert_ = get_pert_idx(pert_category, pert_names)
#         else:  # variant perturbation
#             pert_idx, pert_ = get_pert_idx(pert_category, pert_names)

#         pert_de_category = adata_.obs['condition_name'][0]  # condition 이름 (e.g., A549_G266E+ctrl_1+1)

#         if de:  # DE gene이 존재할 경우
#             test = np.array(de_genes[pert_de_category][:num_de_genes])  # top-N DE gene 이름
#             de_idx = np.where(adata_.var_names.isin(test))[0]  # 해당 유전자 이름의 인덱스 추출
#         else:
#             de_idx = [-1] * num_de_genes  # default index

#         for cell_z in adata_.X:  # perturbation 샘플마다
#             ctrl_samples = ctrl_adata[np.random.randint(0, len(ctrl_adata), num_samples), :]  # 랜덤 control 샘플 추출
#             for c in ctrl_samples.X:
#                 Xs.append(c)  # control expression 추가
#                 ys.append(cell_z)  # 해당 perturb cell 추가

#     else:  # control인 경우
#         pert_idx = None
#         de_idx = [-1] * num_de_genes
#         pert_ = ["ctrl"]
#         for cell_z in adata_.X:  # control sample 그대로 사용
#             Xs.append(cell_z)
#             ys.append(cell_z)

#     cell_graphs = []  # 최종 cell graph 리스트

#     for X, y in zip(Xs, ys):  # 각 샘플에 대해
#         if '~' not in pert_category:  # 일반 perturbation
#             cell_graphs.append(create_cell_graph(X.toarray(), y.toarray(), de_idx, pert_category, pert_idx, pert_))
#         else:  # variant perturbation
#             cell_graphs.append(create_cell_graph(X.toarray(), y.toarray(), de_idx, pert_category, pert_idx, pert_))

#     return cell_graphs  # torch_geometric Data 객체 리스트 반환

# # original-past
# def create_cell_graph_dataset(split_adata, pert_category, num_samples, pert_names, ctrl_adata, rank_genes_option=False):
#     """
#     Combine cell graphs to create a dataset of cell graphs

#     Parameters
#     ----------
#     split_adata: anndata.AnnData
#         Annotated data matrix
#     pert_category: str
#         Perturbation category
#     num_samples: int
#         Number of samples to create per perturbed cell (i.e. number of
#         control cells to map to each perturbed cell)
#     pert_names: str
#         Perturbation list
#     Returns
#     -------
#     list
#         List of cell graphs

#     """

#     num_de_genes = 20
#     adata_ = split_adata[split_adata.obs['condition'] == pert_category]
#     if ('rank_genes_groups_cov_all' in adata_.uns) and (rank_genes_option is True):
#         de_genes = adata_.uns['rank_genes_groups_cov_all']
#         de = True
#     else:
#         de = False
#         num_de_genes = 1
#     Xs = []
#     ys = []

#     # When considering a non-control perturbation
#     # ctrl이 아닌 경우
#     if pert_category != 'ctrl':
#         # Get the indices of applied perturbation
#         # perturbation index
#         pert_idx, variant_list = get_pert_idx(pert_category, pert_names)

#         # Store list of genes that are most differentially expressed for testing
#         # print(adata_.obs['condition_name'])
#         pert_de_category = adata_.obs['condition_name'][0] # A549_TP53~G266E+ctrl_1+1
#         if de: # when (rank_genes_option == True)
#             de_idx = np.where(adata_.var_names.isin(
#             np.array(de_genes[pert_de_category][:num_de_genes])))[0]
#         else: # when (rank_genes_option != True)
#             de_idx = [-1] * num_de_genes
#         for cell_z in adata_.X:
#             # Use samples from control as basal expression
#             ctrl_samples = ctrl_adata[np.random.randint(0,
#                                     len(ctrl_adata), num_samples), :]
#             for c in ctrl_samples.X:
#                 Xs.append(c)
#                 ys.append(cell_z)

#     # When considering a control perturbation
#     else:
#         pert_idx = None
#         de_idx = [-1] * num_de_genes
#         for cell_z in adata_.X:
#             Xs.append(cell_z)
#             ys.append(cell_z)

#     # Create cell graphs
#     cell_graphs = []
#     for X, y in zip(Xs, ys):
#         cell_graphs.append(create_cell_graph(X.toarray(),
#                             y.toarray(), de_idx, pert_category, pert_idx))

#     return cell_graphs

def create_dataset_file(adata, pert_names, ctrl_adata, gex_layer):
    """
    Create dataset file for each perturbation condition
    """
    """ 
    주어진 AnnData 객체에서 각 perturbation condition에 대해
    GNN 학습용 cell graph를 생성하고 이를 딕셔너리 형태로 저장하는 함수입니다.
    """

    print("Creating dataset file...")  # 시작 로그 출력
    dataset_processed = {}  # 결과 저장용 딕셔너리 초기화
    for p in tqdm(adata.obs['condition'].unique()):  # 모든 unique condition에 대해 반복
        dataset_processed[p] = create_cell_graph_dataset(adata, p, 1, pert_names, ctrl_adata, gex_layer)  # 각 condition에 대해 cell graph 생성
    print("Done!")  # 완료 로그 출력
    return dataset_processed  # 생성된 cell graph 딕셔너리 반환


# numpy-free [fixed issue {250802-1}]: numpy._core version mismatch, numpy-free data generation process
def np_pearson_cor(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute column-wise Pearson correlation between two matrices x and y using PyTorch.

    Parameters
    ----------
    x : torch.Tensor
        (n_samples, n_features)
    y : torch.Tensor
        (n_samples, n_features)

    Returns
    -------
    torch.Tensor
        (n_features, n_features) Pearson correlation matrix
    """
    # Centering
    xv = x - x.mean(dim=0)
    yv = y - y.mean(dim=0)

    # Variance components
    xvss = (xv * xv).sum(dim=0)
    yvss = (yv * yv).sum(dim=0)

    # Outer product of std deviations
    denom = torch.sqrt(torch.ger(xvss, yvss))  # outer product

    # Correlation matrix
    corr = xv.T @ yv / denom

    # Clip for numerical stability
    return torch.clamp(corr, -1.0, 1.0)

# # original [fixed issue {250802-1}]: numpy._core version mismatch, numpy-free data generation process
# def np_pearson_cor(x, y):
#     """ 
#     두 행렬 x, y 간 column-wise Pearson correlation을 계산하는 함수입니다.
#     numpy만 사용하며, precision 보정을 위해 -1 ~ 1로 값 클리핑합니다.
#     """
#     xv = x - x.mean(axis=0)  # x의 각 column에서 평균을 제거 (중심화)
#     yv = y - y.mean(axis=0)  # y도 마찬가지로 중심화
#     xvss = (xv * xv).sum(axis=0)  # x의 제곱합 (분산 요소)
#     yvss = (yv * yv).sum(axis=0)  # y의 제곱합
#     result = np.matmul(xv.transpose(), yv) / np.sqrt(np.outer(xvss, yvss))  # correlation 계산
#     return np.maximum(np.minimum(result, 1.0), -1.0)  # precision 문제 방지를 위해 -1 ~ 1로 클리핑



def tar_data_download_wrapper(url, save_path, data_path):
    """
    Wrapper for tar file download

    Args:
        url (str): the url of the dataset
        save_path (str): the path where the file is donwloaded
        data_path (str): the path to save the extracted dataset
    """
    """ 
    Dataverse에서 .tar.gz 파일을 다운로드 받고 압축을 해제하는 래퍼 함수입니다.
    이미 존재하면 재다운로드하지 않습니다.
    """

    if os.path.exists(save_path):  # 이미 존재하는 경우
        print('Found local copy...')  # 스킵
    else:
        dataverse_download(url, save_path + '.tar.gz')  # .tar.gz 다운로드
        print('Extracting tar file...')  # 압축 해제 시작 로그
        with tarfile.open(save_path + '.tar.gz') as tar:  # tar 열기
            tar.extractall(path=data_path)  # 압축 해제
        print("Done!")  # 완료 로그 출력


def dataverse_download(url, save_path):
    """
    Dataverse download helper with progress bar

    Args:
        url (str): the url of the dataset
        path (str): the path to save the dataset

    주어진 URL에서 데이터를 스트리밍 방식으로 다운로드하고,
    tqdm을 이용하여 진행률(progress bar)을 출력하는 함수입니다.
    """

    if os.path.exists(save_path):  # 이미 다운로드 되어 있으면
        print('Found local copy...')  # 스킵
    else:
        print("Downloading...")  # 다운로드 시작 로그
        response = requests.get(url, stream=True)  # 스트리밍 방식 요청
        total_size_in_bytes = int(response.headers.get('content-length', 0))  # 전체 파일 크기
        block_size = 1024  # 블록 단위 크기
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)  # tqdm 진행률 바
        with open(save_path, 'wb') as file:  # 파일 열기
            for data in response.iter_content(block_size):  # 블록 단위로 읽기
                progress_bar.update(len(data))  # 진행률 업데이트
                file.write(data)  # 파일에 기록
        progress_bar.close()  # 진행률 바 종료


def add_gene_to_condition(row):
    """
    Ex)
    adata.obs['condition'] = adata.obs.apply(add_gene_to_condition, axis=1)

    condition 컬럼 값에 gene 이름을 붙여 'Gene~Variant' 형태로 변환하는 함수입니다.
    """
    variants = row['condition'].split('+')  # '+'로 condition 분리
    new_variants = [  # 각 변이 이름에 대해
        f"{row['gene']}~{variant}" if variant not in ['unassigned', 'ctrl', 'WT'] else variant
        for variant in variants
    ]
    new_condition = '+'.join(new_variants)  # 다시 '+'로 합침
    return new_condition  # 변환된 condition 반환


def find_rare_conditions(adata):
    """
    anndata 객체에서 condition별 sample 개수가 2개 미만인 condition을 반환하는 함수
    Ex)
    rare_conditions = find_rare_conditions(adata)

    주어진 AnnData에서 condition별 샘플 수를 계산하고,
    샘플이 2개 미만인 rare condition을 리스트로 반환합니다.
    """
    condition_counts = adata.obs['condition'].value_counts()  # condition별 count 계산
    rare_conditions = condition_counts[condition_counts < 2].index.tolist()  # count < 2 인 조건만 추출
    return rare_conditions  # rare condition 리스트 반환


def move_ctrl_to_end(cond):
    """
    pandas DataFrame에서 condition 열의 각 문자열을 '+'로 분리한 후 'ctrl'을 가장 뒤로 보내고 다시 '+'로 결합하는 코드
    'F+ctrl+G' -> 'F+G+ctrl'
    adata.obs['condition'] = adata.obs['condition'].apply(move_ctrl_to_end)

    condition 내 ctrl을 마지막으로 이동시켜 통일된 표현 형식을 갖도록 정리하는 함수입니다.
    """
    parts = cond.split('+')  # '+' 기준 분리
    non_ctrl = [p for p in parts if p != 'ctrl']  # ctrl이 아닌 부분
    ctrl = [p for p in parts if p == 'ctrl']  # ctrl만 따로
    return '+'.join(non_ctrl + ctrl)  # ctrl을 마지막으로 정렬


def count_non_ctrl_variants(cond):
    """ 
    주어진 condition에서 ctrl이 아닌 perturbation 수를 카운트하는 함수입니다.
    예: 'A+B+ctrl' → 2
    """
    variants = cond.split('+')  # '+' 기준 분리
    return len([v for v in variants if v != 'ctrl'])  # ctrl 제외한 항목 개수 반환


def add_metadata(adata):
    """ 
    condition 수에 따라 dose_val, control 여부, condition_name을 새롭게 정의해
    obs 메타데이터에 추가하는 함수입니다.
    """
    # fixed 20250516: 3개 이상의 condition인 경우 존재
    adata.obs.loc[:, 'dose_val'] = adata.obs.condition.apply(lambda x: '+'.join(['1'] * len(x.split('+'))))  # condition 수 만큼 '1' 추가
    adata.obs.loc[:, 'control'] = adata.obs.condition.apply(lambda x: 0 if len(x.split('+')) >= 2 else 1)  # 2개 이상이면 control=0
    adata.obs.loc[:, 'condition_name'] =  adata.obs.apply(lambda x: '_'.join([x.cell_type, x.condition, x.dose_val]), axis=1)  # 고유한 이름 생성
    return adata  # 수정된 adata 반환


def filter_pert_in_go(condition, pert_names):
    """
    Filter perturbations in GO graph

    Args:
        condition (str): whether condition is 'ctrl' or not
        pert_names (list): list of perturbations

    주어진 condition에 포함된 perturbation들이 모두 GO graph 상에 있는지 확인하는 함수입니다.
    조건에 포함된 모든 항목이 'ctrl' 또는 pert_names에 있어야 True를 반환합니다.
    """
    if condition == 'ctrl':  # control이면 바로 True
        return True
    else:
        conds = condition.split('+')  # '+' 기준 분할
        num_ctrl = sum(c == 'ctrl' for c in conds)  # ctrl 개수 카운트
        num_in_perts = sum(c in pert_names for c in conds)  # pert_names 내 포함된 항목 수
        return (num_ctrl + num_in_perts) == len(conds)  # 모두 포함되었는지 확인
