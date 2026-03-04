# TODO: raw count version PyG 생성하기
# TODO: Terminate variant 제거하기
# TODO: Fold 데이터 계획

# 표준 라이브러리
import os
import json
import pickle
from collections import defaultdict
from datetime import datetime
from zipfile import ZipFile

# 외부 패키지
import numpy as np
import pandas as pd
import requests
import networkx as nx
import scanpy as sc
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

# Bio/omics 관련
import zarr
import mygene

# 시각화
import matplotlib.pyplot as plt

# tqdm (반복 시 progress bar)
from tqdm import tqdm

# GEARS 및 관련 유틸
from gears import PertData, GEARS
from gears_data_utils import get_DE_genes, get_dropout_non_zero_genes, DataSplitter
from gears_utils import print_sys, zip_data_download_wrapper, dataverse_download,\
                  filter_pert_in_go, get_genes_from_perts, tar_data_download_wrapper
# 사용자 정의 모듈 (로컬)
from data import *
from network import *
from emb_data import *
from config import DATA_DIR, RAW_DATA_PATH, OUTPUT_DIR




class CustomConditionData:
    """
    CustomConditionData 클래스는 주어진 단일 cell perturbation dataset을 로드하고,
    gene-level 및 condition-level graph를 구성하며, PyTorch Geometric에 맞게 데이터를
    가공하여 train/valid/test split된 데이터로 반환하는 전체 전처리 파이프라인을 구성합니다.
    """
    def __init__(self, data_path, gene_set_path=None, default_pert_graph=True):
        self.data_path = data_path  # 데이터 루트 경로 설정
        self.default_pert_graph = default_pert_graph
        self.dataset_path = None  # 실제 데이터셋이 위치한 경로
        self.gene_set_path = gene_set_path
        self.adata = None  # AnnData 객체
        self.go = None  # GO-term 매핑 dict
        self.dataloader = None  # PyG 데이터로더
        self.subgroup = None  # 서브그룹 정보 (test, val)
        self.gene_list = None  # 전체 유전자 리스트
        self.conditions = None  # control 제외한 condition 리스트
        self.conditions_ = None  # control 포함 condition 리스트
        self.param_G_expression_thr = 0.4  # gene expression edge threshold
        self.param_G_expression_num = 20  # gene expression edge 개수 제한
        self.param_G_condition_thr = 0.1  # condition graph edge threshold
        self.param_G_condition_num = 20  # condition edge 개수 제한
        self.embedding_cache = None  # 임베딩 캐시
        self.data_seed = None  # seed
        self.train_test_ratio = 0.75  # train/test 비율
        self.train_valid_ratio = 0.9  # train/valid 비율
        self.hvg_rank_dict = None  # HVG 순위 정보
        self.hvg_n_idx = None  # top-N HVG index
        self.E_condition = None
        self.E_gene = None


        self.dataset_name = None
        self.get_dataloader = None
        self.ctrl_adata = None
        self.gene_names = []
        self.node_map = {}

        # Split attributes
        self.split = None
        self.seed = None
        self.train_gene_set_size = None

        self.gex_layer = None
        self.x_suffix = ''
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        server_path = 'https://dataverse.harvard.edu/api/access/datafile/6153417'
        dataverse_download(server_path,
                           os.path.join(self.data_path, 'gene2go_all.pkl'))
        with open(os.path.join(self.data_path, 'gene2go_all.pkl'), 'rb') as f:
            self.gene2go = pickle.load(f)
    def _suffixer(self):

        if self.gex_layer == 'counts':
            self.x_suffix = '_[counts]'
        else: 
            self.x_suffix = ''
    def set_pert_genes(self):
        """
        Set the list of genes that can be perturbed and are to be included in 
        perturbation graph
        """
        
        if self.gene_set_path is not None:
            # If gene set specified for perturbation graph, use that
            path_ = self.gene_set_path
            self.default_pert_graph = False
            with open(path_, 'rb') as f:
                essential_genes = pickle.load(f)
            
        elif self.default_pert_graph is False:
            # Use a smaller perturbation graph 
            all_pert_genes = get_genes_from_perts(self.adata.obs['condition'])
            essential_genes = list(self.adata.var['gene_name'].values)
            essential_genes += all_pert_genes
            
        else:
            # Otherwise, use a large set of genes to create perturbation graph
            server_path = 'https://dataverse.harvard.edu/api/access/datafile/6934320'
            path_ = os.path.join(self.data_path,
                                     'essential_all_data_pert_genes.pkl')
            dataverse_download(server_path, path_)
            with open(path_, 'rb') as f:
                essential_genes = pickle.load(f)
    
        gene2go = {i: self.gene2go[i] for i in essential_genes if i in self.gene2go}

        # self.pert_names = np.unique(list(gene2go.keys())) # original [fixed issue {250802-1}]: numpy._core version mismatch, numpy-free data generation process
        self.pert_names = sorted(set(gene2go.keys()))
        self.node_map_pert = {x: it for it, x in enumerate(self.pert_names)}
    def new_data_process(self, dataset_name,
                         adata = None,
                         skip_calc_de = False, 
                         config = None):
        """
        새로운 데이터셋을 처리하는 함수
    
        Parameters
        ----------
        dataset_name: str
            데이터셋 이름
        adata: AnnData object
            유전자 발현 정보를 포함하는 AnnData 객체
        skip_calc_de: bool
            True일 경우, 차등 발현(DE) 유전자 계산을 생략함
    
        Returns
        -------
        None
        """
        
        # 필수 메타데이터 컬럼 존재 여부 확인
        if 'condition' not in adata.obs.columns.values:
            raise ValueError("Please specify condition")  # 'condition' 컬럼 없음 → 에러
        if 'gene_name' not in adata.var.columns.values:
            raise ValueError("Please specify gene name")  # 'gene_name' 없음 → 에러
        if 'cell_type' not in adata.obs.columns.values:
            raise ValueError("Please specify cell type")  # 'cell_type' 없음 → 에러
        # 데이터셋 이름 소문자 처리 및 저장 경로 지정
        dataset_name = dataset_name.lower()
        self.dataset_name = dataset_name
        save_data_folder = os.path.join(self.data_path, dataset_name)
        if not os.path.exists(os.path.join(save_data_folder, 'perturb_processed.h5ad')):
            
            # 저장 경로가 없으면 생성
            if not os.path.exists(save_data_folder):
                os.mkdir(save_data_folder)
            self.dataset_path = save_data_folder
        
            # 차등 발현 유전자 계산 (필요시 dropout 및 0값 필터링도 수행)
            self.adata = get_DE_genes(adata, skip_calc_de)
            if not skip_calc_de:
                self.adata = get_dropout_non_zero_genes(self.adata)
        
            # 전처리된 AnnData 저장
            self.adata.write_h5ad(os.path.join(save_data_folder, 'perturb_processed.h5ad'))
        else:
            self.adata = sc.read_h5ad(os.path.join(save_data_folder, 'perturb_processed.h5ad'))
        # Perturbation gene 설정 및 control 데이터 분리
        self.set_pert_genes()
        self.gex_layer = config['gex_layer']
        self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']
        self.gene_names = self.adata.var.gene_name
        
        self._suffixer()
        # GNN용 PyG 오브젝트 저장 경로 지정
        pyg_path = os.path.join(save_data_folder, 'data_pyg')
        if not os.path.exists(pyg_path):
            os.mkdir(pyg_path)
        dataset_fname = os.path.join(pyg_path, f'cell_graphs{self.x_suffix}.pkl')


        if not os.path.exists(dataset_fname):
            # 셀 단위 그래프 데이터 생성 및 저장
            print_sys("Creating pyg object for each cell in the data...")
            self.dataset_processed = create_dataset_file(self.adata, self.pert_names, self.ctrl_adata, self.gex_layer)
            print_sys("Saving new dataset pyg object at " + dataset_fname)
            pickle.dump(self.dataset_processed, open(dataset_fname, "wb"))    
            print_sys("Done!")

    def _build_pyg_and_loaders_for_current_split(self):
        """
        # k-fold 루프를 돌면서 fold별로 저장하는 wrapper

        self.adata.obs['split']와 self.dataset_path를 기준으로
        - PyG cell graph 생성/저장/로드
        - Gene Expression Net 생성/저장/로드
        - Condition Graph 생성/저장/로드
        - DataLoader 구성
        을 수행하는 기존의 긴 코드 블록을 여기로 그대로 옮긴다.
        """
        self._suffixer()
        # split별 condition 목록 구성 (groupby + lambda 활용)
        splits = dict(self.adata.obs.groupby('split').agg({'condition': lambda x: x}).condition)
        
        # 중복 제거 및 numpy array -> list 변환
        splits = {i: np.unique(j).tolist() for i, j in splits.items()}
        #---------------------------------------------------------#
        # self.adata 기반으로 PyTorch Geometric 데이터셋 생성 또는 로드
        #---------------------------------------------------------#
        # 처리된 Graph 데이터가 존재하지 않는 경우 새로 생성
        print('p1')
        if not os.path.exists(self.dataset_path + f'/data_pyg/cell_graphs{self.x_suffix}.pkl'):
            print("Creating PyG data from anndata")
            
            # Control 샘플 추출 (조건: 'ctrl')
            ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']
            
            # perturbation에 해당하는 GO-term 정의된 condition 목록 추출
            go_related_condition_list = list(set(self.go.keys()))
            
            print('p1')
            dataset_processed = {}                     # PyG 형식으로 가공한 cell-level graph 저장할 dict initialize
            for condition in tqdm(self.conditions_):   # 모든 condition(perturbation) 별로 cell-level graph 생성
                # 각 condition에 대해 create_cell_graph_dataset 실행
                dataset_processed[condition] = create_cell_graph_dataset(
                    self.adata,                        # 전체 데이터
                    condition,                         # 현재 condition
                    1,                                 # 샘플 수 (ctrl 샘플 하나 매칭)
                    go_related_condition_list,         # perturbation 리스트
                    ctrl_adata,                         # control 샘플
                    self.gex_layer
                )
            print('p1')
            # 생성한 그래프를 pickle로 저장
            if not os.path.exists(self.dataset_path + '/data_pyg'):
                os.mkdir(self.dataset_path + '/data_pyg')
            if not os.path.exists(self.dataset_path + '/dataloader'):
                os.mkdir(self.dataset_path + '/dataloader')
            print('p1')
            with open(file=self.dataset_path + f'/data_pyg/cell_graphs{self.x_suffix}.pkl', mode='wb') as f:
                pickle.dump(dataset_processed, f)
            print('p1')
            print("PyG data created and saved to {}".format(self.dataset_path + f'/data_pyg/cell_graphs{self.x_suffix}.pkl'))
        
        else:
            # 이미 생성된 파일이 있으면 로드
            with open(file=self.dataset_path + f'/data_pyg/cell_graphs{self.x_suffix}.pkl', mode='rb') as f:
                print(f"Load data from {self.dataset_path}/data_pyg/cell_graphs{self.x_suffix}.pkl")
                dataset_processed = pickle.load(f)
        
        # HVG 정보를 각 graph에 추가 (선택 사항)
        if self.top_n_hvg:
            hvg_tensor = torch.tensor(self.hvg_total_idx, dtype=torch.long)  # 전체 HVG index tensor 생성
            for condition, data_list in dataset_processed.items():
                for data in data_list:
                    data.hvg_total_idx = hvg_tensor  # 각 데이터에 HVG 인덱스 부착
        #------------------------------------------------------------#




        
        #-------------------------------------------------------------------------------------#
        # dataset_processed를 기반으로 train/valid/test 데이터셋 분할 및 PyTorch DataLoader 구성
        # TODO: PyG 있으면 다시 split 할 필요가 없어요. Force 인 경우에만 다시 split하게 수정하는게 좋아요
        self._suffixer()
        if not os.path.exists(self.dataset_path + f'/dataloader/dataloader{self.x_suffix}.pkl'):
            print("Split PyG data into train, valid, test set")
            
            split_into = ['train','valid','test']  # split 세 가지 기준
            cell_graphs = {}  # split별 그래프 저장 dict
            
            for s in split_into:
                cell_graphs[s] = []
                for condition in splits[s]:  # 해당 split 조건에 속한 모든 condition에 대해
                    cell_graphs[s].extend(dataset_processed[condition])  # cell graph list 확장
            
            # PyTorch DataLoader 구성
            train_loader = DataLoader(cell_graphs['train'],
                                        batch_size=self.batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(cell_graphs['valid'],
                                    batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(cell_graphs['test'],
                                        batch_size=self.batch_size, shuffle=False)
            
            # DataLoader 저장
            self.dataloader = {
                'train_loader': train_loader,
                'val_loader': val_loader,
                'test_loader': test_loader
            }


            split_dir = f'{self.dataset_path}/dataloader'
            if not os.path.exists(split_dir):
                os.mkdir(split_dir)

            with open(f'{split_dir}/dataloader{self.x_suffix}.pkl', 'wb') as f:
                pickle.dump(self.dataloader, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("PyG data done")
        else:
            # 이미 생성된 파일이 있으면 로드
            with open(file=self.dataset_path + f'/dataloader/dataloader{self.x_suffix}.pkl', mode='rb') as f:
                print(f"Load data from {self.dataset_path}/dataloader/dataloader{self.x_suffix}.pkl")
                self.dataloader = pickle.load(f)
        #-------------------------------------------------------------------------------------#




        #--------------------------------------#
        # Gene Expression 기반 유전자 네트워크 생성
        #--------------------------------------#
        if os.path.isfile(self.dataset_path + '/G_express_importance.csv') == False:
            print("Calculating Gene Expression Net")
        
            # 유전자 인덱스 → 유전자 이름 매핑 dict 생성
            idx2gene = dict(zip(range(len(self.gene_list)), self.gene_list))
        
            X = self.adata.X  # 전체 gene expression matrix
            train_perts = splits['train']  # train condition 리스트
        
            # train condition 중 'ctrl'이 포함된 샘플만 추출
            X_train = X[np.isin(self.adata.obs.condition, [i for i in train_perts if 'ctrl' in i])]
            X_train = X_train.toarray()  # sparse matrix → dense
        
            # Pearson 상관계수 행렬 계산
            out = pearson_correlation(X_train, X_train)
        
            # NaN을 0으로, 음수 상관도 포함하여 절대값 취함
            out[np.isnan(out)] = 0
            out = np.abs(out)
        
            # 유전자별로 상관도 높은 N개 추출
            out_sort_idx = np.argsort(out)[:, -(self.param_G_expression_num + 1):]  # 인덱스 정렬
            out_sort_val = np.sort(out)[:, -(self.param_G_expression_num + 1):]     # 값 정렬
        
            # 엣지 리스트 구성 (source, target, 상관계수)
            df_g = []
            for i in range(out_sort_idx.shape[0]):
                target = idx2gene[i]  # 타깃 유전자
                for j in range(out_sort_idx.shape[1]):
                    df_g.append((idx2gene[out_sort_idx[i, j]], target, out_sort_val[i, j]))
        
            # 상관계수 threshold 이하인 엣지 제거
            df_g = [i for i in df_g if i[2] > self.param_G_expression_thr]
        
            # DataFrame으로 변환 및 저장
            G_expression_importance = pd.DataFrame(df_g).rename(columns={0: 'source', 1: 'target', 2: 'importance'})
            G_expression_importance.to_csv(self.dataset_path + '/G_express_importance.csv')
        else:
            # 기존 생성된 유전자 네트워크 파일 로드
            print(f"Loading Gene Expression Net from: {self.dataset_path}/G_express_importance.csv")
            G_expression_importance = pd.read_csv(self.dataset_path + '/G_express_importance.csv')
        #--------------------------------------#



        #------------------------#
        # NetworkX 그래프 객체 생성
        #------------------------#
        G_expression_edge_list = G_expression_importance.groupby('target').apply(
            lambda x: x.nlargest(self.param_G_expression_num + 1, ['importance'])  # 타깃 유전자당 상위 N개만 유지
        ).reset_index(drop=True)
        
        # DiGraph (방향 그래프)로 변환
        self.G_expression = nx.from_pandas_edgelist(
            G_expression_edge_list,
            source='source',
            target='target',
            edge_attr=['importance'],
            create_using=nx.DiGraph()
        )
        
        # 그래프에 없는 유전자 노드 보완 추가
        for g in self.gene_list:
            if g not in self.G_expression.nodes():
                self.G_expression.add_node(g)
        
        # gene 이름 → index로 변환
        gene2idx = {x: it for it, x in enumerate(self.gene_list)}
        
        # 엣지 index 리스트 구성: (target, source) 방향
        edge_index_ = [(gene2idx[e[1]], gene2idx[e[0]]) for e in self.G_expression.edges]
        self.G_expression_edge_index = torch.tensor(edge_index_, dtype=torch.long).T  # shape: [2, num_edges]
        
        # 엣지 중요도(weight) 추출 및 Tensor 변환
        edge_attr = nx.get_edge_attributes(self.G_expression, 'importance')
        importance = np.array([edge_attr[e] for e in self.G_expression.edges])
        self.G_expression_edge_weight = torch.Tensor(importance)
        #------------------------#




        #--------------------------------------------------#
        # Condition 간 그래프 (condition-level GNN을 위한 구조)
        # perturbation 간 유사도를 바탕으로 네트워크 생성
        # variant-level 조건은 현재 미사용 (예: KRAS~G12D 등), gene-level만 처리
        #--------------------------------------------------#
        if self.default_pert_graph:
            # GO 기반 기본 condition 그래프 파일이 없으면 다운로드
            if os.path.isfile(os.path.join(self.data_path, 'go_essential_all/go_essential_all.csv')) == False:
                server_path = 'https://dataverse.harvard.edu/api/access/datafile/6934319'
                # .tar 파일 다운로드 및 압축 해제
                tar_data_download_wrapper(
                    server_path,
                    os.path.join(self.data_path, 'go_essential_all'),
                    self.data_path
                )
        
            # 이미 존재하는 경우는 파일 로드
            G_condition_importance = pd.read_csv(os.path.join(self.data_path, 'go_essential_all/go_essential_all.csv'))
        
        # 사용자 정의 GO-term 조건 그래프 (예: GO:BP 등)
        elif os.path.isfile(self.dataset_path + f'/G_condition_importance_{self.E_condition}.csv') == False:
            print(f"Calculating Gene Expression Net {self.dataset_path}/G_condition_importance_{self.E_condition}.csv")
        
            # GO Biological Process (GO:BP) 기반 그래프
            if self.E_condition == "GO:BP":
                gene_names = self.condition  # perturbation gene list
                condition_list = list(set(self.go.keys()))  # GO-term이 정의된 condition만 추출
        
                # Jaccard similarity를 활용한 condition 간 유사도 계산
                edge_list = []
                for idx, g1 in enumerate(tqdm(condition_list, desc="Calculating Jaccard Similarity")):
                    go_set1 = set(self.go[g1])  # gene1의 GO set
                    for g2 in condition_list[idx:]:  # 자기 자신 포함 이후 gene2에 대해 반복
                        if g1 == g2:
                            similarity = 1.0  # 동일 gene 간 유사도 1
                        else:
                            go_set2 = set(self.go[g2])
                            intersection = len(go_set1.intersection(go_set2))
                            union = len(go_set1.union(go_set2))
                            similarity = intersection / union if union != 0 else 0
        
                        # (g1, g2, similarity) 튜플로 엣지 저장
                        edge_list.append((g1, g2, similarity))
        
                # 유사도가 threshold 이상인 엣지만 필터링
                edge_list_filter = [i for i in edge_list if i[2] > self.param_G_condition_thr]
        
                # DataFrame으로 변환 및 저장
                G_condition_importance = pd.DataFrame(edge_list_filter).rename(
                    columns={0: 'source', 1: 'target', 2: 'importance'}
                )
                G_condition_importance.to_csv(self.dataset_path + f'/G_condition_importance_{self.E_condition}.csv')
            else:
                pass  # 향후 다른 조건 유형(GO:MF, CC 등)을 위해 확장 가능
        
        # 이미 계산된 condition 그래프가 존재하면 로드
        else:
            print(f"Loading Gene Condition Net from: {self.dataset_path}/G_condition_importance_{self.E_condition}.csv")
            G_condition_importance = pd.read_csv(self.dataset_path + f'/G_condition_importance_{self.E_condition}.csv')
        #--------------------------------------------------#



        #---------------------------------------------#
        # condition-level 그래프 구성 및 torch 텐서로 변환
        #---------------------------------------------#
        # 각 target condition에 대해 중요도 상위 N개 엣지만 유지
        G_condition_edge_list = G_condition_importance.groupby('target').apply(
            lambda x: x.nlargest(self.param_G_condition_num + 1, ['importance'])  # 중요도 기준 상위 N+1개 엣지 선택
        ).reset_index(drop=True)
        
        # NetworkX 방향성 그래프 (DiGraph)로 생성
        self.G_condition = nx.from_pandas_edgelist(
            G_condition_edge_list,
            source='source',
            target='target',
            edge_attr=['importance'],
            create_using=nx.DiGraph()
        )
        
        # 조건 목록에 포함된 모든 node가 그래프에 있는지 확인, 없으면 추가
        for n in self.condition:
            if n not in self.G_condition.nodes():
                self.G_condition.add_node(n)
        
        # 각 condition에 대해 고유 index 부여 (예: {"TP53": 0, "KRAS": 1, ...})
        condition2idx = {cond: idx for idx, cond in enumerate(self.condition)}
        
        # 그래프 edge를 index 쌍으로 변환: (target, source) → edge_index[:, i] 형식
        edge_index_ = [(condition2idx[e[1]], condition2idx[e[0]]) for e in self.G_condition.edges]
        self.G_condition_edge_index = torch.tensor(edge_index_, dtype=torch.long).T  # shape: [2, num_edges]
        
        # edge importance (가중치) 추출 → edge 순서대로 importance 배열 생성
        edge_attr = nx.get_edge_attributes(self.G_condition, 'importance')
        importance = np.array([edge_attr[e] for e in self.G_condition.edges])
        self.G_condition_edge_weight = torch.Tensor(importance)  # Tensor로 변환
        #---------------------------------------------#
    def load(self, config):
        """
        - 데이터를 불러오고 (AnnData)
        - gene2go, perturbation condition, embedding 등을 로드/생성하며
        - HVG 설정, PyG cell graph 생성, train/valid/test 분할
        - gene-level, condition-level graph 생성 후 self에 저장
        """
        self.config = config
        # 데이터셋 경로 설정
        self.dataset_path = os.path.join(self.data_path, self.config['data_name'])  # 데이터셋 디렉토리 경로 설정
        self.adata_path = os.path.join(self.dataset_path, f'{self.config['adata_name']}.h5ad')  # anndata 파일 경로 설정
        self.adata = sc.read_h5ad(self.adata_path)                   # anndata 객체 로드
        self.data_seed = self.config.get('data_seed', 42)            # 데이터 전처리 seed.
        self.embedding_model = self.config['embedding_model']        # 사용할 단백질 임베딩 모델. 'esm2_t33_650M_UR50D' / 'esm_msa1_t12_100M_UR50S'
        self.mutation_type = self.config.get('mutation_type', 'amino')            # amino, dna
        self.E_condition = self.config.get('E_condition', 'GO:BP')   # GO:BP, ...
        self.batch_size = self.config.get('batch_size', 32)
        self.top_n_hvg = self.config.get("top_n_hvg", None)
        self.split = self.config['split']
        self.gex_layer = self.config.get('gex_layer','norm')
        if self.gex_layer == 'counts':
            self.x_suffix = '_[counts]'
        else: 
            self.x_suffix = ''
        
        default_pert_graph = True
        
        # gene2go (GO term) 데이터 로드
        with open(self.data_path + "/gene2go_all.pkl", 'rb') as f:
            gene2go = pickle.load(f)
        self.go = gene2go  # 전체 Gene-GO dictionary 저장



        #-------------------------------#
        # condition 및 유전자 리스트 초기화
        #-------------------------------#
        self.condition_col = 'condition'  # condition 컬럼명 지정
        self.conditions = [p for p in self.adata.obs[self.condition_col].unique() if p != 'ctrl']  # 'ctrl'을 제외한 perturbation 목록
        self.conditions_ = [p for p in self.adata.obs[self.condition_col].unique()]  # 전체 condition 목록 (ctrl 포함)
        self.condition = get_condition_lists(self.adata.obs[self.condition_col])#.tolist()  # 1-gene perturbation만 추출 [fixed issue {250802-1}]: numpy._core version mismatch, numpy-free data generation process
        # variant-seq인 경우 유전 변이가 발생한 유전자의 목록을 가져와야합니다.
        # [Example] self.condition는 'TP53~R175H'로 되어있어서 유전자를 못가져옴. 저걸 바꾸면 일이 커짐. 그냥 새로 하나 만드는 게 나음
        if 'gene' in self.adata.obs.columns: # [Warning]! Perturb-seq인데 adata.obs에 'gene' 컬럼이 존재하면 안됩니다. # ToDo: 시간 남을때 여기 리팩토링 하쇼 # 스노우볼 커지기 전에만 하자
            print('adata.obs.columns에 gene column이 존재합니다. variant-seq data로 인식')
            self.pert_condition =  get_condition_lists(self.adata.obs['gene'])#.tolist() [fixed issue {250802-1}]: numpy._core version mismatch, numpy-free data generation process
        self.gene_list = list(self.adata.var.gene_name.values)  # 유전자 이름 리스트
        if len(self.gene_list) == 0:
            self.gene_list = list(self.adata.var.index.values)  # gene_name이 없을 경우 index 사용
        print(f'# of conditions: {len(self.conditions)}')
        print(f'# of genetic perturbations: {len(self.condition)}')
        print(f'# of genes: {len(self.gene_list)}')
        #------------------------------#





        # self.embedding_model = "esm2_t33_650M_UR50D"  # 사용할 sequence language model
        #-------------------------#
        # sequence embedding cache
        #-------------------------#
        # self.cache_path = self.dataset_path + f"/embedding_cache/embedding_cache_[{self.embedding_model}].pkl"  # 임베딩 캐시 경로 지정
        self.cache_path = f'/NFS_DATA/samsung/database/gears/embedding/embedding_cache_variant_position_[{self.embedding_model}].pkl'
        self.database_path = os.path.join(DATA_DIR,"sequence_embedding.zarr")
        variant_list = list(self.condition)  # perturbation 조건 리스트 → 예: ['KRAS~G12D', 'TP53~R175H']
        print("variant_list:")
        print(variant_list)

        # Embedding이 전부 존재하는지 확인
        for gene_var_ in variant_list:
            unknown_embeddding_stack = []
            if gene_var_ == 'ctrl':
                continue
            # 유전자 및 변이 파싱
            if '~' in gene_var_:  # 변이 정보가 있는 경우, [Gene~Variant] 형태
                gene, variant = gene_var_.split("~")  # gene과 variant 분리
            else:                 # 변이 정보가 없는 경우, [Gene] 형태
                gene, variant = gene_var_, "REF"  # 변이명을 REF로 설정

            # UniProt ID 매핑
            uniprot_id = get_uniprot_id(gene, use_manual_mapping=True)  # gene명을 UniProt ID로 변환
            if not uniprot_id:  # 매핑 실패 시
                logger.error(f"[UniProt ID 실패] {gene}")  # 에러 로그
                logger.info(f'[수동 매핑 필요] add_manual_mapping("{gene}", "???")')  # 수동 매핑 유도 로그
                continue  # 해당 항목 건너뜀

            root = zarr.open(self.database_path, mode='a')  # Zarr 파일 열기 (append 모드)

            if uniprot_id not in root:  # Zarr DB에 해당 ID가 없을 경우
                logger.warning(f"[Zarr 누락] {uniprot_id} not in root")  # 경고 로그 출력
                continue  # 다음 항목으로 이동

            group = root[uniprot_id]  # 해당 유전자 그룹 접근
            path = f"embeddings/{self.mutation_type}/{variant}/{self.embedding_model}"  # 임베딩 벡터 경로 구성
            if path in group:  # 경로가 존재하는 경우
                continue
            else:  # 임베딩 데이터가 존재하지 않을 경우
                unknown_embeddding_stack.append([gene,variant])
                logger.warning(f"[임베딩 누락] Missing: {gene}~{variant}")  # 경고 로그 출력
                collect_data(gene, variant, self.embedding_model, self.mutation_type)

        # load cache / create cache
        if os.path.exists(self.cache_path):  # 기존 캐시 파일 존재 여부 확인
            print(f'cache file exist: {self.cache_path}')
            self.embedding_cache = load_embedding_cache(self.cache_path)  # 캐시 로드
        else:
            print(f'cache file does not exist: {self.cache_path}')
            self.embedding_cache = preload_embeddings(variant_list, self.embedding_model, self.mutation_type)  # 임베딩 생성
            if not os.path.exists(self.dataset_path + "/embedding_cache"):  # 디렉토리 없으면 생성
                os.makedirs(self.dataset_path + "/embedding_cache")
            save_embedding_cache(self.embedding_cache, self.cache_path)  # 생성한 캐시 저장
        #----------------------------#



        #----------------------------#
        # HVG(top variable genes) 계산
        #----------------------------#
        if self.top_n_hvg:
            if 'rank_overall' not in self.adata.var.columns:  # 'rank_overall' 순위 열이 없으면
                if 'dispersions_norm' not in self.adata.var.columns:  # dispersion 값이 없으면
                    sc.pp.highly_variable_genes(self.adata, n_top_genes=len(self.adata.var))  # HVG 계산
                self.adata.var["rank_overall"] = self.adata.var["dispersions_norm"].rank(
                    method="min",  # 동일값은 동일 순위, 건너뛰기
                    ascending=False,  # 높은 값이 높은 순위
                    na_option="bottom"  # NaN은 최하위
                )
            # rank_overall 기반 HVG index 정보 생성
            self.hvg_rank_dict = dict(enumerate(self.adata.var['rank_overall'].values.astype(int)))  # {index: 순위}
            self.hvg_total_idx = sorted(self.hvg_rank_dict.keys(), key=lambda k: self.hvg_rank_dict[k])  # 순위로 정렬된 인덱스 리스트
            self.hvg_n_idx = set([k for k, v in self.hvg_rank_dict.items() if v <= self.top_n_hvg])  # top-N 안에 드는 index들만 추림
        #----------------------------#

        #------------------------------#
        # GO-term 기반 gene subset 필터링
        #------------------------------#
        if default_pert_graph == True:
            # 필수 유전자 리스트 로드
            with open(os.path.join(self.data_path,'essential_all_data_pert_genes.pkl'), 'rb') as f:
                essential_genes = pickle.load(f)
            # 필수 유전자에 해당하는 GO-term만 필터링
            self.go = {i: self.go[i] for i in essential_genes if i in self.go}
            # [Caution]! dataset내의 condition만 self.condition에 포함 => GO-Term에 존재하는 모든 유전자를 self.condition에 포함.
            # self.condition 목록 업데이트
            self.condition = np.unique(list(self.go.keys()))
        else:
            # 사용자 정의 condition만 필터링 (GO-term 정의가 존재하는 경우만)
            self.go = {i: self.go[i] for i in self.condition if i in self.go}
        #------------------------------#



        #-----------#
        # split data
        #-----------#
        if self.split == 'exist':
            self._build_pyg_and_loaders_for_current_split()
        elif self.split == 'default':  # 기본 split 전략 (condition 기반)
            print("split variant option: default")
            # train/test split (perturbation 기준)
            train, test, test_ = split_conditions_train_valid_test(
                self.conditions, 
                split_ratio=self.train_test_ratio, 
                oo_ratio=self.train_test_ratio, 
                seed=self.data_seed
            )
            # train/valid split
            train, valid, valid_ = split_conditions_train_valid_test(
                train, 
                split_ratio=self.train_valid_ratio, 
                oo_ratio=self.train_valid_ratio, 
                seed=self.data_seed, 
                split_for='valid'
            )
            # ctrl은 항상 train에 포함
            train.append('ctrl')
            # subgroup 정보 저장
            self.subgroup = {'test_subgroup': test_, 'val_subgroup': valid_}
            print("split variant done")
        
            # split 정보를 adata.obs['split']에 매핑
            map_dict = {x: 'train' for x in train}
            map_dict.update({x: 'valid' for x in valid})
            map_dict.update({x: 'test' for x in test})
            map_dict.update({'ctrl': 'train'})  # control은 항상 train
            self.adata.obs['split'] = self.adata.obs['condition'].map(map_dict)

            self._build_pyg_and_loaders_for_current_split()
        
        elif self.split == 'random':
            # 샘플 단위 랜덤 split 전략
            train_ratio = 0.7
            valid_ratio = 0.15
            test_ratio = 0.15
            np.random.seed(self.data_seed)  # 시드 고정
            n_samples = self.adata.n_obs  # 총 샘플 수
            indices = np.random.permutation(n_samples)  # 무작위 순열
        
            # 분할 인덱스 계산
            train_end = int(train_ratio * n_samples)
            valid_end = train_end + int(valid_ratio * n_samples)
        
            # 각 샘플에 split 레이블 부여
            split_labels = np.array([""] * n_samples, dtype=object)
            split_labels[indices[:train_end]] = "train"
            split_labels[indices[train_end:valid_end]] = "valid"
            split_labels[indices[valid_end:]] = "test"
        
            # split 정보를 adata.obs에 저장
            self.adata.obs["split"] = split_labels

            self._build_pyg_and_loaders_for_current_split()

        elif isinstance(self.split, str) and self.split.endswith("-fold"):
            # 예: self.split = '3-fold'
            print(f"split variant option: {self.split}")

            # ex) self.split = '3-fold'
            k_str = self.split.split("-")[0]
            k = int(k_str)

            # 1) condition 테이블 생성
            cond_df = build_condition_table(self.adata, cond_col="condition")

            # 2) ctrl 제외 + variant_count == 1만 사용
            cond_df = cond_df[cond_df["condition"] != "ctrl"].copy()
            df_split = cond_df[cond_df["variant_count"] == 1].copy()

            # 3) best split 탐색
            # config['group_by_pos']에 설정 안하면, 그냥 seen/unseen position 적용
            self.group_by_pos = self.config.get('group_by_pos', True)
            best_imbalance, best_info = find_best_split(
                df_split,
                n_folds=k,
                n_trials=5000,
                random_state=42,
                group_by_pos=self.group_by_pos,   # default
            )

            # print("최적 candidate의 fold별 총 샘플 수:", best_info["fold_sums"])
            # print("fold 간 샘플 수 차이(최대-최소):", best_imbalance)

            # 4) df_split에 fold 라벨 저장
            df_split["fold"] = -1
            for f, idxs in enumerate(best_info["folds"]): # condition list
                df_split.loc[idxs, "fold"] = f

            # # 5) A/B/C variant 목록이 필요하면
            list_A = df_split.loc[best_info["groups"]["A"], "condition"].tolist()
            list_B = df_split.loc[best_info["groups"]["B"], "condition"].tolist()
            list_C = df_split.loc[best_info["groups"]["C"], "condition"].tolist()


            # print("A (중복 pos에서 대표 1개):", list_A[:10], "...")
            # print("B (비중복 pos):", list_B[:10], "...")
            # print("C (중복 pos의 나머지):", list_C[:10], "...")

            ordered_conditions = df_split.loc[best_info["order"], "condition"].tolist()
            self.adata, split_info = make_cv_splits_from_order(
                adata=self.adata,
                ordered_conditions=list_A+list_B+list_C, # condition 이름이어야함
                n_folds=k,
                cond_col="condition",
            )


            

            # fold별 폴더 준비
            fold_dirs = prepare_kfold_dirs(self)  # 위에서 정의한 함수

            # 원래 base path 백업
            base_dataset_path = self.dataset_path

            for n in range(k):
                print(f"\n=== Running fold {n+1}/{k} ({self.split}) ===")

                # 1) fold별 dataset_path 설정
                self.dataset_path = fold_dirs[n]
                self.adata.obs['split'] = None
                self.adata.obs.loc[self.adata.obs['condition']=='ctrl','split'] = 'train'
                self.adata.obs.loc[self.adata.obs['condition'].isin(split_info[n]['train_conditions']),'split'] = 'train'
                self.adata.obs.loc[self.adata.obs['condition'].isin(split_info[n]['valid_conditions']),'split'] = 'valid'
                self.adata.obs.loc[self.adata.obs['condition'].isin(split_info[n]['test_conditions']),'split'] = 'test'
                self.adata.write_h5ad(os.path.join(self.dataset_path, 'perturb_processed_metadata.h5ad'))

                # 3) 이제 이 k_n-fold-split에 대해 그래프/데이터로더 생성
                self._build_pyg_and_loaders_for_current_split()

            # 다 돌고 나면 원래 dataset_path 복구
            self.dataset_path = base_dataset_path
        else:
            raise ValueError(f"Unknown split option: {self.split}")



"""
</-------------------------------------------------------->
==========================================================
 Balanced Variant Splitting Framework (TP53 Variant-Seq)
==========================================================

[목적]
- 단일 세포 변이 스크리닝 데이터에서 각 condition(=variant)의 샘플 수가 불균형하기 때문에,
  모델 학습/평가용 3-Fold split을 할 때 fold 간 샘플 수 imbalance를 최소화해야 함.
- 특히 동일한 아미노산 위치(pos)를 공유하는 변이들이 존재하는데,
  이 경우 같은 pos의 변이가 서로 다른 fold에 들어가면 정보 누수 가능성이 있음.
- 따라서 "pos 기반 그룹 제약"을 유지하면서 fold별 총 샘플 수 균형을 최적화하는 split을 생성하는 것이 목적.

[방법]
1) anndata.obs['condition']로부터 각 조건(condition)별 샘플 수를 집계하여 테이블 생성.
   - 변이 개수 (variant_count)
   - 변이 위치(pos) 합 (예: R273C+E287K → 273 + 287)
   - n_samples = 해당 condition의 샘플 수

2) pos 기준으로 조건들을 두 부류로 분류:
   A: pos가 중복되는 그룹 내에서 대표 1개 (train split 안정성 위해 유지)
   B: pos가 유일한 variant (그냥 fold 분배)
   C: pos가 중복된 나머지 variant들

3) A + B + C 순서를 다양한 랜덤 샘플링 방식으로 생성하여
   각 candidate를 3-fold(K=3)로 분배:
      fold f = index % 3
   각 fold의 총 샘플 수(n_samples 합)를 계산.

4) 수천 번 반복하며(예: n_trials=5000)
   fold 간 imbalance = max(fold_sum) - min(fold_sum)
   값이 가장 작은 candidate를 선택.

5) 선택된 best split을 variant-level DataFrame(df_split) 에 fold 라벨로 저장

[결과]
- 동일 pos 변이들이 동일 fold에 묶여 정보 누수를 방지하고,
- 전체 조건을 3개의 fold로 나누었을 때 샘플 수가 가장 균형(balanced)되며,
- 모델 학습/평가에서 안정적인 cross-validation split 구성이 가능해짐.
"""

import re
import numpy as np
import pandas as pd
from anndata import AnnData


def build_condition_table(
    adata: AnnData,
    cond_col: str = "condition",
) -> pd.DataFrame:
    """
    각 condition별로
      - HCT116, U2OS 샘플 수
      - variant_count (변이 개수)
      - pos (position 합)
    계산해서 반환.
    """
    # 각 adata에서 condition value_counts
    adata = adata[~adata.obs[cond_col].str.contains('Ter', na=False)]
    cond_counts = adata.obs[cond_col].value_counts()
    cond_counts = cond_counts.reset_index()
    cond_counts = cond_counts.rename(columns={cond_col: "condition", "count": "n_samples"})

    # variant_count, pos, n_samples 계산
    cond_counts["variant_count"] = cond_counts["condition"].apply(count_non_ctrl_variants)
    cond_counts["pos"] = cond_counts["condition"].apply(get_pos_sum)

    # 기준이 되는 샘플 수: 여기서는 HCT116만 사용 (원래 코드 유지)
    # 필요하면 HCT116+U2OS로 바꿀 수 있음
    cond_counts = cond_counts.rename(columns={'count':'n_samples'})

    return cond_counts

import re
import pandas as pd

import re
import pandas as pd

import re
import pandas as pd

def parse_positions_from_condition(cond) -> list[int]:
    """
    condition 예시:
      'TP53~Y220C'
      'TP53~Y220C+TP53~Y220D'
      'ctrl', 'CTRL_1', None

    반환:
      'TP53~Y220C'                     -> [220]
      'TP53~Y220C+TP53~Y220D'          -> [220, 220]
      'ctrl' / 'CTRL_1' / None         -> []
    """
    # NaN, None 처리
    if cond is None or (isinstance(cond, float) and pd.isna(cond)):
        return []

    cond = str(cond)

    # ctrl 계열은 전부 변이 없음
    if cond.lower().startswith("ctrl"):
        return []

    positions: list[int] = []

    # '+' 로 여러 개 붙어 있을 수 있다고 가정
    parts = [p for p in cond.split("+") if p.strip()]

    for part in parts:
        # 예: 'TP53~Y220C' → "~Y220C"에서 220만 뽑기
        # TP53의 '53'은 절대 잡지 않도록 '~AA###AA' 패턴만 본다
        m = re.search(r"~[A-Z\*](\d+)[A-Z\*]", part)
        if m:
            positions.append(int(m.group(1)))

    return positions


def get_pos_sum(cond):
    """
    # TODO: split 할때, 주의하세요. 
    'TP53~Y220C'                    -> 220
    'TP53~Y220C+TP53~Y220D'         -> 440
    'ctrl' / None                   -> 0
    """
    positions = parse_positions_from_condition(cond)
    return int(sum(positions)) if positions else 0


def make_candidate(
    df: pd.DataFrame,
    rng=None,
    shuffle_inside: bool = True,
    group_by_pos: bool = True,
):
    """
    df: 반드시 'pos'와 'n_samples' 컬럼 포함

    - group_by_pos=True:
        pos 중복 여부에 따라 A, B, C 리스트 만들기
          * 중복 pos: 그 pos 그룹에서 1개는 A, 나머지는 C
          * 유일 pos: B
        반환:
          order = A + B + C
          groups = {"A": [...], "B": [...], "C": [...]}

    - group_by_pos=False:
        pos는 완전히 무시하고,
        모든 variant index를 랜덤 셔플하여 하나의 order로 사용.
        반환:
          order = 랜덤 셔플된 전체 index 리스트
          groups = {"A": order, "B": [], "C": []} (그냥 전부 A 취급)
    """
    if rng is None:
        rng = np.random.default_rng()

    # --- pos 무시하는 simple 모드 ---
    if not group_by_pos:
        idxs = list(df.index)
        rng.shuffle(idxs)
        return idxs, {"A": idxs, "B": [], "C": []}

    # --- 기존: pos 중복 고려하는 모드 ---
    A, B, C = [], [], []

    # 1) 먼저 pos별 인덱스 리스트를 만들고
    pos_groups = []
    for pos, sub in df.groupby("pos"):
        idxs = list(sub.index)
        pos_groups.append((pos, idxs))

    # 2) "해당 pos를 가진 variant 수" 기준으로 내림차순 정렬
    #    → 중복이 많은 pos 그룹이 먼저 오도록
    pos_groups.sort(key=lambda x: len(x[1]), reverse=True)

    # (선택) 같은 크기끼리는 순서를 랜덤하게 섞고 싶으면:
    # rng.shuffle(pos_groups)  # 정렬 전에 섞고, 그 다음 sort(use stable) 해도 됨

    # 3) 정렬된 pos 그룹 순서대로 A/B/C 분배
    for pos, idxs in pos_groups:
        if len(idxs) == 1:
            # 유일 pos → B 그룹
            B.append(idxs[0])
        else:
            # 중복 pos 그룹
            # 어떤 variant가 대표(A)로 갈지는 랜덤
            rng.shuffle(idxs)
            A.append(idxs[0])       # 대표 하나
            C[:0] = idxs[1:]        # 나머지는 C에 flatten으로 추가
            # C.extend(idxs[1:])      # 나머지는 C에 flatten으로 추가

    order = A + B + C
    groups = {"A": A, "B": B, "C": C}
    return order, groups


def eval_candidate(df: pd.DataFrame, order, n_folds: int = 3):
    """
    주어진 order(variant 순서)를 n_folds로 나눌 때,
    각 fold의 n_samples 합계와 imbalance(max-min)를 계산.
    """
    n_samples = df.loc[order, "n_samples"].to_numpy()

    fold_sums = np.zeros(n_folds, dtype=float)
    folds = [[] for _ in range(n_folds)]

    for k, (idx, cnt) in enumerate(zip(order, n_samples)):
        f = k % n_folds  # 0,1,2,0,1,2,... 식으로 분배
        fold_sums[f] += cnt
        folds[f].append(idx)

    imbalance = float(fold_sums.max() - fold_sums.min())
    return imbalance, fold_sums, folds

def find_best_split(
    df: pd.DataFrame,
    n_folds: int = 3,
    n_trials: int = 500,
    random_state: int = 42,
    group_by_pos: bool = True,
):
    """
    여러 번 candidate를 만들고, 각 candidate에 대해
    fold별 n_samples 합의 imbalance(max-min)가 가장 작은 경우를 선택.

    - group_by_pos=True  → pos 중복을 고려(A/B/C 구조 사용)
    - group_by_pos=False → pos 완전 무시, 모든 variant를 그냥 랜덤 셔플
    """
    rng = np.random.default_rng(random_state)

    best_imbalance = None
    best_info = None

    for _ in range(n_trials):
        order, groups = make_candidate(
            df,
            rng=rng,
            shuffle_inside=True,
            group_by_pos=group_by_pos,
        )
        imbalance, fold_sums, folds = eval_candidate(df, order, n_folds=n_folds)

        if (best_imbalance is None) or (imbalance < best_imbalance):
            best_imbalance = imbalance
            best_info = {
                "order": order,
                "groups": groups,
                "fold_sums": fold_sums,
                "folds": folds,
            }

    return best_imbalance, best_info








import re
import numpy as np
import pandas as pd
from anndata import AnnData

import numpy as np
import pandas as pd
from anndata import AnnData

def make_cv_splits_from_order(
    adata: AnnData,
    ordered_conditions,
    n_folds: int = 3,
    cond_col: str = "condition",
    valid_mode: str = "prev_block_last_scaled",
    valid_ratio: float = 1/10,
):
    """
    A+B+C를 합쳐 만든 ordered_conditions 리스트를 기준으로
    리스트를 n_folds개의 연속 구간(block)으로 쪼개고,
    각 fold k에 대해 train/valid/test를 정의.

    valid_mode:
      - "prev_block_last_third":
          test 블록 바로 이전 블록(prev block)의 뒤 1/3을 valid로 사용.
          (나머지는 test/valid가 아닌 모든 condition이 train)
    """
    adata_out = adata.copy()
    print(ordered_conditions)
    # ordered_conditions = list(ordered_conditions)
    m = len(ordered_conditions)
    if m == 0:
        raise ValueError("ordered_conditions가 비어 있습니다.")

    # --- 1) ordered_conditions를 n_folds개 연속 구간으로 나누기 위한 boundary 계산 ---
    # boundaries: [0, ..., m] 길이 n_folds+1
    # 예: m=10, n_folds=3 -> [0, 3, 6, 10]
    boundaries = [(k * m) // n_folds for k in range(n_folds + 1)]

    # --- 2) 각 condition이 어느 block(=fold) 에 속하는지 기록 ---
    cond_to_block = {}  # condition 문자열 -> block index
    for f in range(n_folds):
        start, end = boundaries[f], boundaries[f + 1]
        for cond in ordered_conditions[start:end]:
            cond_to_block[cond] = f

    # --- 3) obs["fold"] 컬럼 생성 (이 condition이 test가 되는 fold 번호) ---
    fold_values = []
    for cond in adata_out.obs[cond_col].astype(str):
        if cond in cond_to_block:
            fold_values.append(cond_to_block[cond])
        else:
            # ordered_conditions에 없는 condition (예: ctrl, Ter 등) 은 -1
            fold_values.append(-1)
    adata_out.obs["fold"] = fold_values

    # --- 4) fold별 train/valid/test 조건 및 cell 인덱스 dict 구성 ---
    split_info = {}
    obs = adata_out.obs

    # "train 개수는 항상 동일"을 위해 첫 fold에서 결정해서 고정
    target_train_conditions = None  # 조건 개수 기준
    for f in range(n_folds):
        # (1) 이 fold에서 test에 해당하는 condition들
        start_test, end_test = boundaries[f], boundaries[f + 1]
        test_conditions = ordered_conditions[start_test:end_test]
        test_len = len(test_conditions)


        """
        규칙:
            - train 개수는 모든 fold에서 동일
            - valid 개수는 test 블록의 크기에 따라 가변적
            - 기본적으로 직전 블록에서 valid를 쪼갤 때,
            (n_folds - 1) * valid_ratio 만큼 쪼개는 것을 base로 사용
        """
        prev_f = (f - 1) % n_folds
        start_prev, end_prev = boundaries[prev_f], boundaries[prev_f + 1]
        prev_block_len = end_prev - start_prev

        if prev_block_len <= 0:
            # 이전 블록에 아무것도 없으면 valid 없음
            valid_len_base = 0
            valid_len = 0
            valid_conditions = []
        else:
            # 1) 기본 valid 길이: (n_folds - 1) * valid_ratio 만큼 쪼개기
            #    예: n_folds=4, valid_ratio=1/6이면 → (4-1)*1/6 = 0.5
            #        prev_block_len=14 → base ≈ 7
            scale = (n_folds - 1) * valid_ratio
            valid_len_base = int(round(prev_block_len * scale))
            # 최소 1개, 최대 prev_block_len 개
            valid_len_base = max(1, min(valid_len_base, prev_block_len))

            # 2) 첫 fold라면 target_train_conditions 결정
            if target_train_conditions is None:
                # 첫 fold에서는 base를 사용
                valid_len = valid_len_base
                target_train_conditions = m - test_len - valid_len
            else:
                # 이후 fold에서는 "train 개수 동일"을 맞추기 위해
                # valid_len을 다시 계산:
                #   train = m - test_len - valid_len == target_train_conditions
                #   → valid_len = m - target_train_conditions - test_len
                valid_len = m - target_train_conditions - test_len
                # prev_block 안에서 가능한 범위로 클램프
                if valid_len < 1:
                    valid_len = 1
                if valid_len > prev_block_len:
                    valid_len = prev_block_len
                # 이렇게 조정하면 어떤 fold에서는 train이 약간 달라질 수도 있지만,
                # prev_block 길이 내에서 최대한 target_train에 맞추는 형태.

            # prev 블록의 "뒤에서" valid_len 만큼 뽑기
            valid_start = end_prev - valid_len
            valid_conditions = ordered_conditions[valid_start:end_prev]

        # train = 전체 - (test ∪ valid)
        test_set = set(test_conditions)
        valid_set = set(valid_conditions)
        train_conditions = [
            c for c in ordered_conditions
            if (c not in test_set) and (c not in valid_set)
        ]

                # (3) 각 set에 해당하는 cell index 선택
        train_mask = obs[cond_col].isin(train_conditions)
        valid_mask = obs[cond_col].isin(valid_conditions)
        test_mask  = obs[cond_col].isin(test_conditions)

        train_idx = obs.index[train_mask].to_list()
        valid_idx = obs.index[valid_mask].to_list()
        test_idx  = obs.index[test_mask].to_list()
        # ==============================================================
        # (4) [검증] 중복 pos에 대한 train/test 분포 규칙 체크
        #      - 중복 pos(같은 pos를 공유하는 variant가 2개 이상)라면
        #        -> train에 정확히 1개
        #        -> valid에는 0개
        #        -> 나머지는 test에만 존재해야 함
        # ==============================================================
        # pos -> 해당 pos를 포함하는 condition 리스트 (split별로 분리)
        pos2conds_train = {}
        pos2conds_valid = {}
        pos2conds_test  = {}

        for cond in train_conditions:
            for pos in parse_positions_from_condition(cond):
                pos2conds_train.setdefault(pos, []).append(cond)

        for cond in valid_conditions:
            for pos in parse_positions_from_condition(cond):
                pos2conds_valid.setdefault(pos, []).append(cond)

        for cond in test_conditions:
            for pos in parse_positions_from_condition(cond):
                pos2conds_test.setdefault(pos, []).append(cond)


        # 전체 pos 집합
        all_pos = (
            set(pos2conds_train.keys())
            | set(pos2conds_valid.keys())
            | set(pos2conds_test.keys())
        )

        # pos별로 "총 몇 개의 variant가 있는지" 계산 (train+valid+test 합)
        pos2total_n = {}
        for pos in all_pos:
            n_tr = len(pos2conds_train.get(pos, []))
            n_va = len(pos2conds_valid.get(pos, []))
            n_te = len(pos2conds_test.get(pos, []))
            pos2total_n[pos] = n_tr + n_va + n_te

        # 중복 pos만 필터링 (총 개수 >= 2)
        duplicate_pos = {pos for pos, n in pos2total_n.items() if n >= 2}

        # 규칙 위반 정보 저장용
        pos_rule_violation = {}

        for pos in sorted(duplicate_pos):
            conds_tr = pos2conds_train.get(pos, [])
            conds_va = pos2conds_valid.get(pos, [])
            conds_te = pos2conds_test.get(pos, [])

            n_tr = len(conds_tr)
            n_va = len(conds_va)
            n_te = len(conds_te)
            total_n = pos2total_n[pos]  # = n_tr + n_va + n_te

            # 원하는 규칙:
            #   - 중복 pos → train에 정확히 1개
            #   - valid에는 0개
            #   - test에는 (총 개수 - 1)개
            ok = (n_tr == 1) and (n_va == 0) and (n_te == total_n - 1)

            if not ok:
                pos_rule_violation[pos] = {
                    "train_conds": conds_tr,
                    "valid_conds": conds_va,
                    "test_conds":  conds_te,
                    "n_train": n_tr,
                    "n_valid": n_va,
                    "n_test":  n_te,
                    "total":   total_n,
                }

        # (선택) 위반 있으면 경고 출력
        if pos_rule_violation:
            print(f"[경고] fold {f}에서 '중복 pos → train=1, valid=0, 나머지 test' 규칙 위반:")
            for pos, info in pos_rule_violation.items():
                print(f"  pos {pos}: {info}")




        split_info[f] = {
            "train_conditions": train_conditions,
            "valid_conditions": valid_conditions,
            "test_conditions":  test_conditions,
            "train_idx": train_idx,
            "valid_idx": valid_idx,
            "test_idx":  test_idx,
            # 검증용 summary
            # "pos_duplicate_rule_violation": pos_rule_violation,
        }

    return adata_out, split_info


"""
# 1) condition 테이블 생성
cond_df = build_condition_table(adata, cond_col="condition")

# 2) ctrl 제외 + variant_count == 1만 사용
cond_df = cond_df[cond_df["condition"] != "ctrl"].copy()
df_split = cond_df[cond_df["variant_count"] == 1].copy()

# 3) best split 탐색
best_imbalance, best_info = find_best_split(
    df_split,
    n_folds=3,
    n_trials=5000,
    random_state=42,
)

print("최적 candidate의 fold별 총 샘플 수:", best_info["fold_sums"])
print("fold 간 샘플 수 차이(최대-최소):", best_imbalance)

# 4) df_split에 fold 라벨 저장
df_split["fold"] = -1
for f, idxs in enumerate(best_info["folds"]):
    df_split.loc[idxs, "fold"] = f

# 5) A/B/C variant 목록이 필요하면
list_A = df_split.loc[best_info["groups"]["A"], "condition"].tolist()
list_B = df_split.loc[best_info["groups"]["B"], "condition"].tolist()
list_C = df_split.loc[best_info["groups"]["C"], "condition"].tolist()

print("A (중복 pos에서 대표 1개):", list_A[:10], "...")
print("B (비중복 pos):", list_B[:10], "...")
print("C (중복 pos의 나머지):", list_C[:10], "...")

ordered_conditions = df_split.loc[best_info["order"], "condition"].tolist()
adata_cv, split_info = make_cv_splits_from_order(
    adata=adata,
    ordered_conditions=ordered_conditions,
    n_folds=3,
    cond_col="condition",
)


"""




import os

def prepare_kfold_dirs(self):
    """
    # k-fold용 폴더 준비 함수

    self.split 이 '3-fold', '5-fold' 처럼 k-fold 형식일 때,
    각 fold마다 dataset_path 하위에 별도 폴더를 생성하고 경로 dict를 반환.

    예:
        self.dataset_path = '/path/to/dataset'
        self.split = '3-fold'

        → 생성되는 폴더:
            /path/to/dataset[0_3-fold]
            /path/to/dataset[1_3-fold]
            /path/to/dataset[2_3-fold]

        → 반환:
            {
                0: '/path/to/dataset[0_3-fold]',
                1: '/path/to/dataset[1_3-fold]',
                2: '/path/to/dataset[2_3-fold]',
            }
    """
    # 1) self.split 파싱: '3-fold' → k=3
    if not isinstance(self.split, str) or not self.split.endswith("-fold"):
        raise ValueError(f"prepare_kfold_dirs: k-fold 형식이 아님: self.split={self.split}")

    k_str = self.split.split("-")[0]
    if not k_str.isdigit():
        raise ValueError(f"prepare_kfold_dirs: fold 수를 해석할 수 없음: self.split={self.split}")

    k = int(k_str)

    # 2) fold별 디렉토리 생성
    base = self.dataset_path
    fold_dirs = {}

    for n in range(k):
        # 요청한 이름 형식: self.dataset_path + '['+ n + '_' + self.split + ']'
        fold_path = f"{base}[{n+1}_{self.split}]"
        os.makedirs(fold_path, exist_ok=True)
        fold_dirs[n] = fold_path

    return fold_dirs
"""

==========================================================
 Balanced Variant Splitting Framework (TP53 Variant-Seq)
==========================================================
<--------------------------------------------------------/>
"""









import torch.nn as nn
from torch_geometric.nn import SGConv

class MLP(torch.nn.Module):
    def __init__(self, sizes, batch_norm=True, last_layer_act="linear"):
        """
        Multi-layer perceptron with optional LayerNorm
        :param sizes: list of layer sizes, e.g. [512, 256, 128]
        :param batch_norm: whether to use LayerNorm
        :param last_layer_act: activation function of the last layer
        """
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers.append(torch.nn.Linear(sizes[s], sizes[s + 1]))
            if batch_norm and s < len(sizes) - 2:  # 마지막 레이어 전까지만 norm 적용
                layers.append(torch.nn.LayerNorm(sizes[s + 1]))
            if s < len(sizes) - 2 or last_layer_act == "relu":
                layers.append(torch.nn.ReLU())

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)





class GEARS_2(torch.nn.Module):
    """
    GEARS 모델 초기화 클래스
    - gene embedding 및 perturbation embedding 초기화
    - GNN 레이어 및 MLP 설정
    - 데이터셋에서 control condition 평균 계산 및 다양한 embedding 모듈 구성
    """
    def __init__(self, ConditionData, device = 'cuda', config = None):
        super(GEARS_2, self).__init__()                                       # 상위 torch.nn.Module 초기화
        self.config = config
        self.gears_mode = self.config.get('gears_mode', 'variant')             # 'gene' 또는 'variant' 모드 설정
        self.device = device                                                  # 사용할 디바이스 (기본: CUDA)
        self.hidden_dim = self.config.get('hidden_dim', 64)                    # 기본 hidden dimension;               default:64
        self.dataloader = ConditionData                                       # 데이터 로더 객체
        self.compression = self.config.get('compression', 'position_embedding')
        if self.compression == 'full_sequence_average':
            self.seq_emb_hidden_dim = max(arr.shape[2] for arr in list(self.dataloader.embedding_cache.values())) # 시퀀스 embedding hidden dim;         default:1280
            self.seq_emb_sequence_dim = max(arr.shape[1] for arr in list(self.dataloader.embedding_cache.values())) # 시퀀스 ;                           default:1280
        elif self.compression == 'position_embedding':
            self.seq_emb_hidden_dim = len(random.choice(list(random.choice(list(self.dataloader.embedding_cache.values())).values()))) # 전체 dimension이 일정해야합니다! Caution!
            self.seq_emb_sequence_dim = len(random.choice(list(random.choice(list(self.dataloader.embedding_cache.values())).values())))
        self.n_genes = len(self.dataloader.gene_list)                         # 유전자 수
        self.n_conditions = len(self.dataloader.condition)                    # 조건 수 # Das ist Go-term
        self.n_gene_layer = self.config['n_gene_layer']                       # gene graph GNN 레이어 수
        self.n_condition_layer = self.config['n_condition_layer']             # condition graph GNN 레이어 수
        # self.param_uncertainty_reg = 1                                      # 불확실성 정규화 계수
        self.uncertainty = self.config['uncertainty']
        self.direction_lambda = self.config['direction_lambda']               # 방향성 손실 가중치;                    default: 1e-1
        self.embedding_merge_method = self.config['embedding_merge_method']
        self.variant_representation = self.config.get('variant_representation', 'ALT')


        # 서열 언어모델의 임베딩 압축 방향 결정
        # 'hidden-wise'로 압축 시, hidden dimension 방향으로 압축되어 -> (1, hidden_dim)
        if self.config['compress_by']=='hidden-wise':                         # 길이 200의 서열의 임베딩이 1280차원일때, 
            self.compress_by = 1
            self.variant_emb_length = int(self.seq_emb_hidden_dim)
        # 'sequence-wise'로 압축 시, sequence dimension 방향으로 압축되어 -> (1, sequence_dim)
        elif self.config['compress_by']=='sequence-wise':
            self.compress_by = 2
            self.variant_emb_length = int(self.seq_emb_sequence_dim)
        # default: hidden dimension 방향 압축 -> (1, hidden_dim)
        else:
            self.compress_by = 1


        ctrl_idx = np.where(self.dataloader.adata.obs['condition'].values == 'ctrl')[0]  # control condition 인덱스
        ctrl_data = self.dataloader.adata.X[ctrl_idx].toarray()  # control 샘플 발현 데이터

        pert_full_id2pert = dict(self.dataloader.adata.obs[['condition_name', 'condition']].values)  # U2OS_TP53~N239D+ctrl_1+1 → TP53~N239D+ctrl 매핑
        self.ctrl_expression = torch.tensor(np.mean(ctrl_data, axis=0), dtype=torch.float32, device=self.device).reshape(-1,)  # control 평균 발현
        
        if 'non_zeros_gene_idx' in self.dataloader.adata.uns:
            self.dict_filter = {
                pert_full_id2pert[i]: j for i, j in self.dataloader.adata.uns['non_zeros_gene_idx'].items() 
                if i in pert_full_id2pert and pert_full_id2pert[i] is not None
            }
            if len(self.dict_filter) == 0:
                print("Warning: dict_filter is empty after initialization.")
        else:
            print("Warning: 'non_zeros_gene_idx' not found in self.adata.uns.")
            self.dict_filter = {}
        # # fixed 20250612: de gene에 대한 loss 계산을 위한 정보
        # if 'top_non_zero_de_20' in self.dataloader.adata.uns:
        #     self.top_de_gene_idx = {
        #         pert_full_id2pert[i]: j for i, j in self.dataloader.adata.uns['top_non_zero_de_20'].items()
        #         if i in pert_full_id2pert and pert_full_id2pert[i] is not None
        #     }
        #     if len(self.top_de_gene_idx) == 0:
        #         print("Warning: top_de_gene_idx is empty after initialization.")
        # else:
        #     print("Warning: 'top_non_zero_de_20' not found in self.adata.uns.")
        #     self.top_de_gene_idx = {}

        self.gene_list = self.dataloader.gene_list  # 유전자 리스트
        self.condition_list = self.dataloader.condition  # 조건 리스트 # Das ist Go-term

        # Gene 노드 임베딩 및 포지션 임베딩 정의
        self.gene_emb = nn.Embedding(self.n_genes, self.hidden_dim)
        self.gene_pos_emb = nn.Embedding(self.n_genes, self.hidden_dim)
        self.G_expression_layers = torch.nn.ModuleList()  # gene graph GNN 레이어 리스트
        for i in range(1, self.n_gene_layer + 1):
            self.G_expression_layers.append(SGConv(self.hidden_dim, self.hidden_dim, 1))
        self.G_expression = self.dataloader.G_expression_edge_index.to(self.device)  # gene edge index
        self.G_expression_weight = self.dataloader.G_expression_edge_weight.to(self.device)  # gene edge weight
        
        # Bilinear 모듈
        # self.condition_variant_bilinear = nn.Bilinear(self.hidden_dim, self.variant_emb_length, self.hidden_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(self.variant_emb_length, self.hidden_dim, bias=True),
        )


        # Condition 노드 임베딩 정의
        self.condition_emb = nn.Embedding(self.n_conditions, self.hidden_dim)
        self.G_condition_layers = torch.nn.ModuleList()  # condition graph GNN 레이어 리스트
        for i in range(1, self.n_condition_layer + 1):
            self.G_condition_layers.append(SGConv(self.hidden_dim, self.hidden_dim, 1))
        self.G_condition = self.dataloader.G_condition_edge_index.to(self.device)  # condition edge index
        self.G_condition_weight = self.dataloader.G_condition_edge_weight.to(self.device)  # condition edge weight

        # MLP 모듈 초기화
        self.position_MLP = MLP([self.hidden_dim, self.hidden_dim, self.hidden_dim])  # 위치 기반 임베딩 후처리
        self.condition_mix_MLP = MLP([self.hidden_dim, self.hidden_dim*2, self.hidden_dim], last_layer_act='linear')  # condition 혼합용 MLP
        self.postpert_MLP = MLP([self.hidden_dim, self.hidden_dim* 2, self.hidden_dim])  # perturbation 후처리 MLP
        # if self.gears_mode == 'variant':
        #     if self.embedding_merge_method == 'cat':
        #         self.variant_condition_mix_MLP = MLP([self.variant_emb_length + self.hidden_dim, 512, 512, self.hidden_dim], last_layer_act='linear')  # variant 정보 포함 , batch_norm=True
        #     elif self.embedding_merge_method == 'no_pert':
        #         self.variant_condition_mix_MLP = MLP([self.variant_emb_length, 512, 512, self.hidden_dim], last_layer_act='linear')  # variant 정보 포함 , batch_norm=True
        #     elif self.embedding_merge_method == 'element':
        #         self.variant_condition_mix_MLP = MLP([self.variant_emb_length, 512, 512, self.hidden_dim], last_layer_act='linear')  # variant 정보 포함 , batch_norm=True
        #     else:
        #         pass
        # 활성화 함수
        self.ReLU = nn.ReLU()

        # 배치 정규화
        self.bn_emb = nn.BatchNorm1d(self.hidden_dim)
        self.bn_pert_base = nn.BatchNorm1d(self.hidden_dim)
        # self.bn_pert_base_trans = nn.BatchNorm1d(self.hidden_dim) # 안쓰임

        # 유전자별 디코더 파라미터 (선형 조합)
        self.weight_1 = nn.Parameter(torch.rand(self.n_genes, self.hidden_dim, 1))
        self.bias_1 = nn.Parameter(torch.rand(self.n_genes, 1))
        nn.init.xavier_normal_(self.weight_1)
        nn.init.xavier_normal_(self.bias_1)

        self.weight_2 = nn.Parameter(torch.rand(1, self.n_genes, self.hidden_dim+1))
        self.bias_2 = nn.Parameter(torch.rand(1, self.n_genes))
        nn.init.xavier_normal_(self.weight_2)
        nn.init.xavier_normal_(self.bias_2)

        # cross-gene state decoder
        self.cross_gene_state = MLP([self.n_genes, self.hidden_dim, self.hidden_dim])
        if self.config['uncertainty']:
            self.uncertainty_w = MLP([self.hidden_dim, self.hidden_dim*2, self.hidden_dim, 1], last_layer_act='linear')
        self.loss_version = self.config['loss_version']
        
    def forward(self, data, visible=[]):
        """
        GEARS 모델의 forward 함수
        - gene embedding + condition embedding + (optional) variant embedding 결합
        - GNN으로 embedding 확장
        - 최종 유전자 발현량 예측값 생성 및 반환

        Parameters
        ----------
        data: torch_geometric.data.Data
            입력 그래프 및 condition/variant 정보 포함
        visible: list
            디버그용 출력 여부

        Returns
        -------
        torch.Tensor or tuple
            유전자 발현 예측값 (및 선택적으로 log-variance 또는 DE subset 예측)
        """
        # 학습 데이터: 
        # data.x: variant 이전의 gene expression data. 유전자 개수만큼의 차원 수
        # data.pert: 타겟 유전자의 인덱스 정보. [idx, idx]
        # data.variant: variant 정보. [Gene~variant, Gene~variant] -> [['TP53~Y200C'],['TP53~Y200C']]
        """
        Section: dataload
        """
        n_graphs = len(data.batch.unique())
        x, condition_idx, variant_condition = data.x, data.pert_idx, data.variant
        # x, condition_idx, variant_condition = data.x, [[2501]*n_graphs], [[data.variant]*n_graphs]

        # intentional_data_leakage 
        # y = data.y

        # [debugging] : 'dataload'가 visible 리스트에 있는 경우
        # if 'dataload' in visible:
        #     print(f"variant_condition: {variant_condition}")

        # Batch size
        n_graphs = len(data.batch.unique())

        # gene embedding
        ## get base gene embeddings
        gene_emb_input = torch.LongTensor(list(range(self.n_genes))).repeat(n_graphs, ).to(self.device)
        # gene_emb_input = torch.arange(self.n_genes, dtype=torch.long, device=self.device).repeat(n_graphs)
        
        """
        Section: gene_embedding
        """
        # 유전자 임베딩 국수 뽑기...
        gene_emb = self.gene_emb(gene_emb_input) # gene 임베딩 벡터 추출
        gene_emb = self.bn_emb(gene_emb)         # 배치 정규화
        gene_emb = self.ReLU(gene_emb)           # 비선형 활성화 적용
        # 삶은 유전자 임베딩 국수!
        # 집가고싶다

        visible_emb = self.config.get("visible_emb", False)
        # [debugging]: 'gene_embedding'이 visible 리스트에 있는 경우
        if 'gene_embedding' in visible:
            plot_embedding_heatmap(
                gene_emb, labels=self.gene_list,
                title=f"Gene Embedding Heatmap [Before GNN] [{gene_emb.shape}]"
            )
            
        # gene positional embedding 구성 및 GNN 적용
        gene_pos_emb_input = torch.LongTensor(list(range(self.n_genes))).repeat(n_graphs, ).to(self.device)
        # if visible == True:
        #     print(f"gene positional embedding input: [{gene_pos_emb_input.shape}]\n")
        gene_pos_emb = self.gene_pos_emb(gene_pos_emb_input)
        # if visible == True:
        #     print(f"gene positional embedding: [{gene_pos_emb.shape}]\n")
        for idx, layer in enumerate(self.G_expression_layers):
            gene_pos_emb = layer(gene_pos_emb, self.G_expression, self.G_expression_weight)  # GNN layer 적용
            if idx < len(self.G_expression_layers) - 1:
                gene_pos_emb = gene_pos_emb.relu()
        gene_emb = gene_emb + 0.2 * gene_pos_emb  # positional 정보 반영
        gene_emb = self.position_MLP(gene_emb)  # MLP 후처리

        # [debugging]: 'gene_embedding'이 visible 리스트에 있는 경우
        if 'gene_embedding' in visible:
            plot_embedding_heatmap(
                gene_emb, labels=self.gene_list,
                title=f"Gene Embedding Heatmap [After GNN] [{gene_emb.shape}]"
            )

        """
        Section: variant_condition_index
        """
        # Section: variant_condition_index
        # (variant embedding, condition embedding, perturbation 적용, decoder 처리)
        # variant embedding 추출 방법
        # OR gears_mode == 'gene'

        if self.gears_mode == 'variant':
            # variant_condition에서 ctrl이 아닌 variant 조건들을 모아 인덱스화
            variant_condition_index = []
            for idx, i in enumerate(variant_condition): # idx, [Gene~variant, Gene~variant]
                for j in i: # Gene~variant
                    if j != 'ctrl': # control condition은 제외
                        variant_condition_index.append([idx, j]) # idx: sample index,  j: variant info
            variant_condition_index = list(zip(*variant_condition_index))

            # [debugging]: 'variant_condition_index'가 visible 리스트에 있는 경우
            """
            [Output Example]
            variant_condition_index[0] (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 29, 30)
            variant_condition_index[1] ('TP53~R196G', 'TP53~I195T', 'TP53~M133T', 'TP53~I232T', 'TP53~S127P', 'TP53~S127P', 'TP53~N239D', 'TP53~R196Q', 'TP53~I232T', 'TP53~N239D', 'TP53~Y220C', 'TP53~P278F', 'TP53~Y220C', 'TP53~Y236H', 'TP53~R196Q', 'TP53~M133T', 'TP53~N239D', 'TP53~N239D', 'TP53~Y236H', 'TP53~N239D', 'TP53~I232T', 'TP53~I232T', 'TP53~N239D', 'TP53~I195T', 'TP53~P278F')
            len(variant_condition_index[0]) 25
            len(variant_condition_index[1]) 25
            variant_condition_index [(0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 29, 30), ('TP53~R196G', 'TP53~I195T', 'TP53~M133T', 'TP53~I232T', 'TP53~S127P', 'TP53~S127P', 'TP53~N239D', 'TP53~R196Q', 'TP53~I232T', 'TP53~N239D', 'TP53~Y220C', 'TP53~P278F', 'TP53~Y220C', 'TP53~Y236H', 'TP53~R196Q', 'TP53~M133T', 'TP53~N239D', 'TP53~N239D', 'TP53~Y236H', 'TP53~N239D', 'TP53~I232T', 'TP53~I232T', 'TP53~N239D', 'TP53~I195T', 'TP53~P278F')]
            """
            if 'variant_condition_index' in visible:
                print(f"variant_condition_index[0] {variant_condition_index[0]}") # sample indices
                print(f"variant_condition_index[1] {variant_condition_index[1]}") # variant info
                print(f"len(variant_condition_index[0]) {len(variant_condition_index[0])}")
                print(f"len(variant_condition_index[1]) {len(variant_condition_index[1])}")
                print(f'variant_condition_index {variant_condition_index}')


            """
            Section: variant_condition_embedding
            """
            # variant 임베딩 도시락 [Empty tensor]
            whole_variant_condition_emb = torch.empty((0, self.hidden_dim), dtype=torch.float32, device=self.device)
            for i, gene_var_ in enumerate(variant_condition_index[1]): # variant info ('TP53~R196G', ..., 'TP53~M133T')
                # {Variant-seq data} | variant 형태가 gene~variant일 경우 처리

                if '~' in gene_var_:
                    gene_, var_ = gene_var_.split('~') # gene과 variant 분리

                    if var_ is None:
                        print(f"Warning: variant {gene_var_} is None")

                    # ALT와 REF 시퀀스 임베딩을 캐시파일에서 로드해옵니다. 캐시파일은 데이터 로드 시 불러옵니다.
                    if self.compression == 'full_sequence_average':
                        embedding_ALT = get_cached_embedding(gene_, var_, self.dataloader.embedding_cache)
                        embedding_REF = get_cached_embedding(gene_, "REF", self.dataloader.embedding_cache)
                    elif self.compression == 'position_embedding':
                        # ALT, DIFF
                        embedding_ALT = get_cached_embedding(gene_, var_, self.dataloader.embedding_cache, self.variant_representation)
                        embedding_REF = get_cached_embedding(gene_, var_, self.dataloader.embedding_cache, self.variant_representation)
                    else:
                        pass
                        # 나중에 경고 메세지 추가

                    # Variant Embedding을 제대로 학습하고 있는 것인지 확인하는 use_dummy_embedding
                    # [Ablation: Variant Embedding] 입니다.
                    use_dummy_embedding = self.config.get("use_dummy_embedding", False)
                    if use_dummy_embedding:
                        if embedding_REF is not None:
                            embedding_REF = np.zeros_like(embedding_REF)
                        if embedding_ALT is not None:
                            embedding_ALT = np.zeros_like(embedding_ALT)

                    # 데이터를 제대로 확인하십시오!!!
                    if (embedding_REF is None) or (embedding_ALT is None):
                        raise ValueError(f"Error: ![{gene_var_}]!\n embedding_REF or embedding_ALT is None .\n embedding_REF={embedding_REF}, embedding_ALT={embedding_ALT}.\n Please check the input or embedding generation process.")

                    # 임베딩 벡터 차원 맞추기를 위한 최대 서열 길이 확인.
                    # max_len = max(embedding_REF.shape[1], embedding_ALT.shape[1])

                    # 변이 TER과 같이 종결 코돈에 의해 번역이 중단되는 경우 Reference와 임베딩 벡터의 차원이 다릅니다.
                    # # ToDo: 생각해보면 임베딩 생성 때 [조기 종결 서열을 입력으로 사용하는 것]과 [Ref 서열에서 0 masking 하는 것] 둘 중에 어떤 방법이 학습에 적절할 지 고민할 필요가 있습니다.
                    # embedding_REF = pad_to_length(torch.tensor(embedding_REF), max_len)
                    # embedding_ALT = pad_to_length(torch.tensor(embedding_ALT), max_len)

                    # Reference Embedding과 Alternate Embedding간의 차이를 Variant Embedding으로 사용합니다.
                    # ToDo: 근데 차이값이 굉장히 굉장히 작아서 임시로 10000을 곱했습니다. 추후 임베딩 값을 적절한 방법으로 정규화하는 작업이 요구됩니다.
                    # self.compression = self.config.get("compression", 'full_sequence') # default: pooling
                    if self.compression == 'full_sequence_average':
                        emb = (embedding_REF - embedding_ALT).mean(axis=self.compress_by) # 차이를 구한 후 Pooling 합니다.
                    elif self.compression == 'position_embedding':
                        emb = embedding_ALT
                    else:
                        # ToDo: 미구현 기능입니다.
                        # pooling을 적용하지 않고 이후 연산으로 고차원의 Variant Embedding 벡터를 넘깁니다.
                        emb = (embedding_REF - embedding_ALT)
                else:
                    # {Perturb-seq data} | variant가 없이 gene만 있는 경우 (e.g. TP53)

                    # REF 시퀀스 임베딩만 캐시파일에서 로드해옵니다.
                    if self.compression == 'full_sequence_average':
                        embedding_REF = get_cached_embedding(gene_var_, "REF", self.dataloader.embedding_cache)
                    if self.compression == 'position_embedding':
                        embedding_REF = get_cached_embedding(gene_var_, "REF", self.dataloader.embedding_cache, self.variant_representation)
                    
                    # 0 벡터 처리!
                    embedding_0 = np.zeros_like(embedding_REF)
                    if embedding_0 is None:
                        raise ValueError(f"Error: ![{gene_var_}]!\n embedding_REF is None .\n embedding_REF={embedding_REF}.\n Please check the input or embedding generation process.")
                    
                    # 기존 전체 임베딩 평균 방법론
                    if self.compression == 'full_sequence_average':
                        emb = embedding_0.mean(axis=self.compress_by) # 평균 벡터 사용
                    # Variant 발생 위치에 해당하는 embedding만 가져오는 방법론 > 사전 처리된 임베딩 파일 사용. embedding_cache_variant_position_[{protein_language_model}]
                    elif self.compression == 'position_embedding':
                        emb = embedding_0
                    else:
                        raise ValueError(f"Error: {self.compression} is not valid method to process variant embedding")

                # # Check Embed
                # print(f"Variant: {gene_var_}")
                # print(f"Embedding: {emb}")
                
                # 불필요한 변환 방지
                if not isinstance(emb, torch.Tensor):
                    emb_tensor = torch.tensor(emb, dtype=torch.float32, device=self.device)
                else:
                    emb_tensor = emb.to(dtype=torch.float32, device=self.device)
                emb_tensor = self.cond_mlp(emb_tensor)
                emb_tensor = emb_tensor.unsqueeze(0)
                # variant 임베딩 텐서를 variant 임베딩 도시락에 추가
                whole_variant_condition_emb = torch.cat((whole_variant_condition_emb, emb_tensor), dim=0)

            # [debugging]: 'variant_condition_enbedding'이 visible 리스트에 있는 경우. batch 내의 variant embedding을 확인합니다.
            if 'variant_condition_embedding' in visible:
                plot_embedding_heatmap(
                    whole_variant_condition_emb, labels=variant_condition_index[1],
                    title=f"Variant Embedding Heatmap [{whole_variant_condition_emb.shape}]"
                )
                # In batch variant embedding stats
                print(f"[Variant Condition Embedding] overall stats: "
                    f"mean={whole_variant_condition_emb.mean().item():.4e}, "
                    f"std={whole_variant_condition_emb.std().item():.4e}, "
                    f"min={whole_variant_condition_emb.min().item():.4e}, "
                    f"max={whole_variant_condition_emb.max().item():.4e}, "
                    f"abs mean={whole_variant_condition_emb.abs().mean().item():.4e}")

                # Check Embedding details
                print(f"Variant Condition Embedding: {whole_variant_condition_emb}")

            # 샘플별 variant embedding 합산
            # [Warning]! 샘플을 기준으로 변이를 모두 합산하는 방식입니다. ToDo: 다중 유전자 조건 예측 시, 유전자 기준으로 합산하도록 수정
            # [Example]? 샘플 내에 변이 유전자가 2개 이상 {TP53~R196G, KRAS~A216V} > R196G + A216V
            variant_track = defaultdict(lambda: torch.zeros(self.hidden_dim, device=self.device)) # 변이 임베딩 도시락 [Empty dict]
            for i in range(len(variant_condition_index[0])): # ex) length: 25
                sample_idx = variant_condition_index[0][i] # ex) i: 0~24
                variant_track[sample_idx] += whole_variant_condition_emb[i] # 동일 샘플인 경우 variant embedding 합산됨. Multi variant case
            
            # Check Embed
            # if visible_emb:
            #     print(f"Whole Variant Embedding Embedding [variant_track]: {variant_track}")






        """
        Section: pert_condition_embedding
        """
        condition_index = [] # pert 인덱스 도시락 [Empty list]
        for idx, i in enumerate(condition_idx): 
            for j in i:
                if j != -1: # -1:ctrl
                    condition_index.append([idx, j]) # batch 내 n번째 sample의 pert_id [n, pert_id],[n, pert_id],[n+1, pert_id],...
        if 'pert_condition_index' in visible:
            print(f"condition_index[0] {condition_index[0]}") # sample indices
            print(f"condition_index[1] {condition_index[1]}") # variant info
            print(f"len(condition_index[0]) {len(condition_index[0])}")
            print(f"len(condition_index[1]) {len(condition_index[1])}")
            print(f'condition_index {condition_index}')
        condition_index = torch.tensor(condition_index).T # pert 인덱스 도시락 [Full tensor], (N, 2) => (2, N), 세로 도시락 > 가로 도시락

        # Pert comdition embedding을 사용
        if self.embedding_merge_method != "no_pert":
            ## Global Condition(pert) 임베딩 가져오기
            condition_emb_input = torch.LongTensor(list(range(self.n_conditions))).to(self.device) # Pert 유전자에 대한 전체 인덱스
            condition_emb = self.condition_emb(condition_emb_input) # Pert 유전자에 대한 Pert comdition embedding 생성

            # Skip Pert Graph
            pert_graph_gnn = self.config.get("pert_graph", False)
            if pert_graph_gnn:
                # condition graph GNN을 통해 임베딩 업데이트
                for idx, layer in enumerate(self.G_condition_layers):
                    condition_emb = layer(condition_emb, self.G_condition, self.G_condition_weight)
                    if idx < len(self.G_condition_layers) - 1:
                        condition_emb = condition_emb.relu()

            # [debugging]: 'pert_condition_embedding'이 visible 리스트에 있는 경우. batch 내의 pert embedding을 확인합니다.
            if 'pert_condition_embedding' in visible:
                plot_embedding_heatmap(
                    condition_emb, labels=condition_index[1],
                    title=f"Pert Conditon Embedding Heatmap [{condition_emb.shape}]"
                )
                print(f"[Pert Condition Embedding] overall stats: "
                    f"mean={condition_emb.mean().item():.4e}, "
                    f"std={condition_emb.std().item():.4e}, "
                    f"min={condition_emb.min().item():.4e}, "
                    f"max={condition_emb.max().item():.4e}, "
                    f"abs mean={condition_emb.abs().mean().item():.4e}")

                print(f"Pert Conditon Embedding: {condition_emb}")


        """
        Section: merge_condition_embedding
        """
        ## perturbation embedding 생성 모듈
        # Variant Embedding을 학습하는 경우, perturbation과 위에서 생성한 variant embedding을 합쳐줍니다.
        if self.gears_mode == 'variant': # Combination Module for variant level perturbation
            
            # Variant Embedding 처리 1단계: gene level의 pert emb와 통합
            gene_emb = gene_emb.reshape(n_graphs, self.n_genes, -1) # (batch, gene, dim)
            pert_track = defaultdict(lambda: []) # sample별 pert 임베딩 도시락 (Empty dict)
            all_embs = [] # Heatmap용 embedding numpy list
            all_names = []  # Heatmap용 sample name list
            for i in range(condition_index.shape[1]):
                sample_idx = condition_index[0][i].item()
                cond_id = condition_index[1][i]

                # Choose how to merge {pert & variant} embedding
                if self.embedding_merge_method == "cat":
                    merge_emb = torch.cat([condition_emb[cond_id], variant_track[sample_idx]], dim=0)
                elif self.embedding_merge_method == "no_pert":
                    merge_emb = variant_track[sample_idx]
                elif self.embedding_merge_method == "dot":
                    pass
                elif self.embedding_merge_method == "element":
                    merge_emb = condition_emb[cond_id] * variant_track[sample_idx]
                elif self.embedding_merge_method == "bilinear":
                    merge_emb = self.condition_variant_bilinear(condition_emb[cond_id], variant_track[sample_idx])
                else:
                    raise ValueError(f"Unknown embedding_merge_method: {self.embedding_merge_method}")

                if sample_idx in pert_track:
                    pert_track[sample_idx] = pert_track[sample_idx] + merge_emb
                else:
                    pert_track[sample_idx] = merge_emb

                # pert_track[sample_idx].append(merge_emb)
                if 'merge_embedding' in visible:
                    all_embs.append(merge_emb.detach().cpu().numpy())
                    all_names.append(str(variant_condition[sample_idx]))

            # [debugging]: 'merge_embedding'이 visible 리스트에 있는 경우. merge embedding 결과를 확인합니다
            if 'merge_embedding' in visible:
                merged_embedding = pd.DataFrame(all_embs, index=all_names)
                plot_embedding_heatmap(
                    merged_embedding.values, labels=merged_embedding.index,
                    title=f"Variant Embedding Heatmap [After Merge] [{all_embs.shape}]"
                )

            """
            Section: transform_variant_embedding
            """
            # Variant Embedding 처리 2단계: 통합된 embed을 MLP 학습. gene embed에 가산 전 작업
            # MLP(Variant Embedding) -> mixed_condition_emb
            # fixed: 20250704
            # mixed_condition_emb = torch.zeros((n_graphs, self.hidden_dim), dtype=torch.float32, device=self.device) # MLP 태운 임베딩을 보관
            # for i in range(condition_index.shape[1]): # 각 샘플 별로 하나씩
            #     sample_idx = condition_index[0][i].item()
            #     embeddings = pert_track[sample_idx]  # perturbation embedding 리스트 [[variant_emb],[variant_emb],...]
            #     if embeddings:
            #         emb_tensor = torch.stack(embeddings, dim=0)  # (n_pert, dim)

            #         # single gene variant 인 경우, variant_condition_mix_MLP가 최소 2개 input을 요구
            #         if emb_tensor.shape[0] == 1 and self.variant_condition_mix_MLP.training:
            #             emb_tensor = emb_tensor.repeat(2, 1)
                    
            #         # [debugging]: 'transform_variant_embedding'이 visible 리스트에 있는 경우. MLP 이전
            #         if 'transform_variant_embedding' in visible:
            #             plot_embedding_heatmap(
            #                 emb_tensor,
            #                 title=f"Variant Embedding Heatmap [Before MLP] [{emb_tensor.shape}]"
            #             )
                    
            #         # variant emb 학습
            #         mixed = self.variant_condition_mix_MLP(emb_tensor)
            #         mixed_condition_emb[sample_idx] = mixed.sum(dim=0, keepdim=True)
                    
            #         # [debugging]: 'transform_variant_embedding'이 visible 리스트에 있는 경우. MLP 이후
            #         if 'transform_variant_embedding' in visible:
            #             plot_embedding_heatmap(
            #                 mixed,
            #                 title=f"Variant Embedding Heatmap [After MLP] [{mixed.shape}]"
            #             )
                    
            #     else:
            #         pass

            # editing..
            if condition_index.shape[0] != 0:
                # comdition mixing
                pert_values = list(pert_track.values())
                if len(pert_values) == 1:
                    # 단일 perturbation 처리
                    emb_total = self.condition_mix_MLP(torch.stack(pert_values * 2))
                else:
                    emb_total = self.condition_mix_MLP(torch.stack(pert_values))

                # Base embedding 업데이트
                for idx, j in enumerate(pert_track.keys()):
                    gene_emb[j] = gene_emb[j] + emb_total[idx]


        # 단순 Gene Perturbation만 사용하는 경우 (= gears_mode가 'gene'인 경우 perturbation 처리)
        else: # Combination Module for gene level perturbation
            gene_emb = gene_emb.reshape(n_graphs, self.n_genes, -1) # (batch 크기, gene 개수, feature 차원)
            if condition_index.shape[0] != 0:
                # pert_track = defaultdict(lambda: 0)  # 기본값 0
                pert_track = {}

                # Perturbation 집계
                ### in case all samples in the batch are controls, then there is no indexing for pert_index.
                for i, j in enumerate(condition_index[0]): # sample indices
                    if j.item() in pert_track:
                        pert_track[j.item()] = pert_track[j.item()] + condition_emb[condition_index[1][i]]
                    else:
                        pert_track[j.item()] = condition_emb[condition_index[1][i]]

                # comdition mixing
                pert_values = list(pert_track.values())
                if len(pert_values) == 1:
                    # 단일 perturbation 처리
                    emb_total = self.condition_mix_MLP(torch.stack(pert_values * 2))
                else:
                    emb_total = self.condition_mix_MLP(torch.stack(pert_values))

                # Base embedding 업데이트
                for idx, j in enumerate(pert_track.keys()):
                    gene_emb[j] = gene_emb[j] + emb_total[idx]

        gene_emb = gene_emb.reshape(n_graphs * self.n_genes, -1)
        gene_emb = self.bn_pert_base(gene_emb)

        ## apply the first MLP
        gene_emb = self.ReLU(gene_emb) # nn.ReLU()
        out = self.postpert_MLP(gene_emb) # decoder shared MLP, MLP([self.hidden_dim, self.hidden_dim*2, self.hidden_dim], last_layer_act='linear')
        out = out.reshape(n_graphs, self.n_genes, -1)
        out = out.unsqueeze(-1) * self.weight_1
        w = torch.sum(out, axis = 2)
        out = w + self.bias_1 # Zu

        # Cross gene
        # Cross gene embedding 계산
        cross_gene_embed = self.cross_gene_state(out.view(n_graphs, self.n_genes, -1).squeeze(2))
        cross_gene_embed = cross_gene_embed.repeat(1, self.n_genes)
        cross_gene_embed = cross_gene_embed.reshape([n_graphs,self.n_genes, -1])
        # cross_gene_embed = cross_gene_embed.unsqueeze(1).expand(-1, self.n_genes, -1) #h^cg

        # Cross gene 결합 및 가중합
        if visible == True:
            print(f"out shape: {out.shape}")
            print(f"cross_gene_embed shape: {cross_gene_embed.shape}")

        cross_gene_out = torch.cat([out, cross_gene_embed], 2)
        cross_gene_out = cross_gene_out * self.weight_2
        cross_gene_out = torch.sum(cross_gene_out, axis=2)
        out = cross_gene_out + self.bias_2
        out = out.reshape(n_graphs * self.n_genes, -1) + x.reshape(-1,1)
        # out = out.view(n_graphs * self.n_genes, -1) + x.view(-1, 1)
        out = torch.split(out.flatten(), self.n_genes)

        ## uncertainty head
        if self.uncertainty:
            out_logvar = self.uncertainty_w(gene_emb)
            out_logvar = torch.split(torch.flatten(out_logvar), self.n_genes)
            return torch.stack(out), torch.stack(out_logvar)
        return torch.stack(out)



from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

def evaluate(loader, model, uncertainty, device):
    """
    Run model in inference mode using a given data loader
    """

    model.eval()
    model.to(device)
    pert_cat = []
    pred = []
    truth = []
    pred_de = []
    truth_de = []
    results = {}
    logvar = []
    
    for itr, batch in enumerate(loader):
        # x, pert_index, variant = batch.x, batch.pert_index, batch.variant
        batch.to(device)
        pert_cat.extend(batch.pert)

        with torch.no_grad():
            if uncertainty:
                p, unc = model(batch)
                logvar.extend(unc.cpu())
            else:
                p = model(batch)
            t = batch.y
            pred.extend(p.cpu())
            truth.extend(t.cpu())
            
            # Differentially expressed genes
            for itr, de_idx in enumerate(batch.de_idx):
                pred_de.append(p[itr, de_idx])
                truth_de.append(t[itr, de_idx])

    # all genes
    results['pert_cat'] = np.array(pert_cat)
    pred = torch.stack(pred)
    truth = torch.stack(truth)
    results['pred']= pred.detach().cpu().numpy()
    results['truth']= truth.detach().cpu().numpy()

    pred_de = torch.stack(pred_de)
    truth_de = torch.stack(truth_de)
    results['pred_de']= pred_de.detach().cpu().numpy()
    results['truth_de']= truth_de.detach().cpu().numpy()
    
    if uncertainty:
        results['logvar'] = torch.stack(logvar).detach().cpu().numpy()
        
    return results


def compute_metrics(results):
    """
    Given results from a model run and the ground truth, compute metrics

    """
    metrics = {}
    metrics_pert = {}

    metric2fct = {
           'mse': mse,
           'pearson': pearsonr
    }
    
    for m in metric2fct.keys():
        metrics[m] = []
        metrics[m + '_de'] = []

    for pert in np.unique(results['pert_cat']):

        metrics_pert[pert] = {}
        p_idx = np.where(results['pert_cat'] == pert)[0]

        for m, fct in metric2fct.items():
            if m == 'pearson':
                val = fct(results['pred'][p_idx].mean(0), results['truth'][p_idx].mean(0))[0]
                if np.isnan(val):
                    val = 0
            else:
                val = fct(results['pred'][p_idx].mean(0), results['truth'][p_idx].mean(0))

            metrics_pert[pert][m] = val
            metrics[m].append(metrics_pert[pert][m])

       
        if pert != 'ctrl':
            
            for m, fct in metric2fct.items():
                if m == 'pearson':
                    val = fct(results['pred_de'][p_idx].mean(0), results['truth_de'][p_idx].mean(0))[0]
                    # if np.isnan(val):
                    #     pred_vec = results['pred_de'][p_idx].mean(0)
                    #     truth_vec = results['truth_de'][p_idx].mean(0)
                    #     print(f"[NaN Pearson] Perturbation: {pert}")
                    #     print(f"Pred DE mean: {pred_vec}")
                    #     print(f"Truth DE mean: {truth_vec}")
                    #     print(f"Var(Pred): {np.var(pred_vec):.6f}, Var(Truth): {np.var(truth_vec):.6f}")
                    #     print("----")
                        
                    #     val = 0
                else:
                    val = fct(results['pred_de'][p_idx].mean(0), results['truth_de'][p_idx].mean(0))
                    
                metrics_pert[pert][m + '_de'] = val
                metrics[m + '_de'].append(metrics_pert[pert][m + '_de'])

        else:
            for m, fct in metric2fct.items():
                metrics_pert[pert][m + '_de'] = 0
    
    for m in metric2fct.keys():
        
        metrics[m] = np.mean(metrics[m])
        metrics[m + '_de'] = np.mean(metrics[m + '_de'])
    
    return metrics, metrics_pert

import wandb
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

def plot_embedding_heatmap(embeddings, labels=None, title='Embedding Heatmap', cmap='vlag'):
    """
    Embedding 행렬을 heatmap으로 시각화합니다.

    Parameters
    ----------
    embeddings : torch.Tensor or np.ndarray
        (n_samples, dim) 형태의 임베딩
    labels : list of str, optional
        샘플 라벨 (y축에 표시됨)
    title : str
        그래프 제목
    cmap : str
        색상 맵

    예시: plot_embedding_heatmap(embeddings, labels=sample_names)
    """

    # 임베딩의 샘플 수와 차원 수를 가져옴
    num_samples, emb_dim = embeddings.shape

    # figure 크기 설정 (샘플 수에 따라 y축 크기를 조정)
    figsize = (10, num_samples / 6)

    # torch.Tensor이면 numpy array로 변환
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    # 새로운 figure 생성
    plt.figure(figsize=figsize)

    # seaborn heatmap으로 시각화
    sns.heatmap(
        embeddings,
        cmap=cmap,
        xticklabels=False,  # x축 라벨은 숨김 (dim이 너무 많으면 가독성 저하)
        yticklabels=labels if labels is not None else True,  # labels가 있으면 사용
        cbar=True  # color bar 표시
    )

    # 제목과 축 라벨 설정
    plt.title(title)
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Samples")

    # 레이아웃 최적화
    plt.tight_layout()

    # 최종 시각화 출력
    plt.show()


def log_embedding_heatmap(model, epoch, mode="train"):
    """
    Forward hook에 저장된 embedding과 nn.Embedding 레이어 weight를 heatmap으로 wandb에 기록

    Parameters
    ----------
    model : GEARS_2
        학습된 모델 (hook_outputs dict 포함)
    epoch : int
        현재 epoch
    mode : str
        "train" or "val"
    """

    def embedding_heatmap_to_wandb_image(embedding, title="Embedding Heatmap", cmap="viridis"):
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()
        plt.figure(figsize=(8, 6))
        sns.heatmap(embedding, cmap=cmap, xticklabels=False, yticklabels=False)
        plt.title(title)
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        img = Image.open(buf)
        return wandb.Image(img, caption=title)

    def embedding_weight_heatmap(embedding_layer, title="Embedding Weight", cmap="viridis"):
        """
        nn.Embedding 레이어의 weight를 heatmap으로 wandb.Image로 변환
        """
        weight = embedding_layer.weight.detach().cpu().numpy()
        plt.figure(figsize=(10, 6))
        sns.heatmap(weight, cmap=cmap, xticklabels=False, yticklabels=False)
        plt.title(title)
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        img = Image.open(buf)
        return wandb.Image(img, caption=title)

    # ✅ variant embedding 학습 모듈
    if "variant_condition_mix_MLP" in model.hook_outputs:
        wandb.log({
            f"{mode}/variant_emb_heatmap_epoch{epoch}": embedding_heatmap_to_wandb_image(
                model.hook_outputs["variant_condition_mix_MLP"],
                title=f"{mode} Variant Embedding (epoch {epoch})"
            )
        })

    # ✅ condition embedding
    if "condition_mix_MLP" in model.hook_outputs:
        wandb.log({
            f"{mode}/condition_emb_heatmap_epoch{epoch}": embedding_heatmap_to_wandb_image(
                model.hook_outputs["condition_mix_MLP"],
                title=f"{mode} Condition Embedding (epoch {epoch})"
            )
        })

    # ✅ gene embedding flow
    for layer_name in ["position_MLP", "postpert_MLP", "cross_gene_state"]:
        if layer_name in model.hook_outputs:
            wandb.log({
                f"{mode}/{layer_name}_heatmap_epoch{epoch}": embedding_heatmap_to_wandb_image(
                    model.hook_outputs[layer_name],
                    title=f"{mode} {layer_name} (epoch {epoch})"
                )
            })

    # ✅ nn.Embedding weight 모니터링
    wandb.log({
        f"{mode}/condition_emb_weight_epoch{epoch}": embedding_weight_heatmap(
            model.condition_emb,
            title=f"{mode} condition_emb weight (epoch {epoch})"
        ),
        f"{mode}/gene_emb_weight_epoch{epoch}": embedding_weight_heatmap(
            model.gene_emb,
            title=f"{mode} gene_emb weight (epoch {epoch})"
        ),
        f"{mode}/gene_pos_emb_weight_epoch{epoch}": embedding_weight_heatmap(
            model.gene_pos_emb,
            title=f"{mode} gene_pos_emb weight (epoch {epoch})"
        ),
    })




def deeper_analysis(adata, test_res, de_column_prefix='rank_genes_groups_cov', most_variable_genes=None):
    # 성능 평가에 사용할 메트릭 정의
    metric2fct = {
        'pearson': pearsonr,
        'mse': mse
    }

    pert_metric = {}

    ## (1) 조건 이름 매핑 및 유전자 정보 정리
    pert2pert_full_id = dict(adata.obs[['condition', 'condition_name']].values)
    geneid2name = dict(zip(adata.var.index.values, adata.var['gene_name']))
    geneid2idx = dict(zip(adata.var.index.values, range(len(adata.var.index.values))))

    ## (2) 각 condition 별 평균 expression 계산
    unique_conditions = adata.obs.condition.unique()
    conditions2index = {cond: np.where(adata.obs.condition == cond)[0] for cond in unique_conditions}
    condition2mean_expression = {cond: np.mean(adata.X[idx], axis=0) for cond, idx in conditions2index.items()}

    pert_list = np.array(list(condition2mean_expression.keys()))
    mean_expression = np.array(list(condition2mean_expression.values())).reshape(len(pert_list), adata.X.shape[1])
    ctrl = mean_expression[np.where(pert_list == 'ctrl')[0]]  # control condition의 평균 expression

    ## (3) HVG가 주어지지 않은 경우: 가장 변화가 큰 유전자 200개 추출
    if most_variable_genes is None:
        most_variable_genes = np.argsort(np.std(mean_expression, axis=0))

    ## (4) perturbation별 평가
    for pert in np.unique(test_res['pert_cat']):
        if pert == 'ctrl':
            continue
        else:
            pert_metric[pert] = {}
    
            # 각 perturbation에 대한 DE 유전자 인덱스 (top N)
            de_idx = [geneid2idx[i] for i in adata.uns['rank_genes_groups_cov_all'][pert2pert_full_id[pert]][:20]]
            de_idx_50 = [geneid2idx[i] for i in adata.uns['rank_genes_groups_cov_all'][pert2pert_full_id[pert]][:50]]
            de_idx_100 = [geneid2idx[i] for i in adata.uns['rank_genes_groups_cov_all'][pert2pert_full_id[pert]][:100]]
            de_idx_200 = [geneid2idx[i] for i in adata.uns['rank_genes_groups_cov_all'][pert2pert_full_id[pert]][:200]]
            de_idx_map = {20: de_idx, 50: de_idx_50, 100: de_idx_100, 200: de_idx_200}
    
            pert_idx = np.where(test_res['pert_cat'] == pert)[0]
            pred_mean = np.mean(test_res['pred_de'][pert_idx], axis=0).reshape(-1,)
            true_mean = np.mean(test_res['truth_de'][pert_idx], axis=0).reshape(-1,)
    
            ## (5) 전체 유전자에 대한 방향성 일치율
            direc_change = np.abs(np.sign(test_res['pred'][pert_idx].mean(0) - ctrl[0]) - np.sign(test_res['truth'][pert_idx].mean(0) - ctrl[0]))
            frac_correct_direction = len(np.where(direc_change == 0)[0]) / len(geneid2name)
            pert_metric[pert]['frac_correct_direction_all'] = frac_correct_direction
    
            ## (6) Top N DE 유전자에 대한 방향성 평가
            for val in [20, 50, 100, 200]:
                d_idx = de_idx_map[val]
                direc_change = np.abs(np.sign(test_res['pred'][pert_idx].mean(0)[d_idx] - ctrl[0][d_idx]) -
                                      np.sign(test_res['truth'][pert_idx].mean(0)[d_idx] - ctrl[0][d_idx]))
                pert_metric[pert][f'frac_correct_direction_{val}'] = (direc_change == 0).sum() / val
    
            ## (7) non-zero DE 유전자에 대한 정밀 분석
            mean = np.mean(test_res['truth_de'][pert_idx], axis=0)
            std = np.std(test_res['truth_de'][pert_idx], axis=0)
            min_ = np.min(test_res['truth_de'][pert_idx], axis=0)
            max_ = np.max(test_res['truth_de'][pert_idx], axis=0)
            q25 = np.quantile(test_res['truth_de'][pert_idx], 0.25, axis=0)
            q75 = np.quantile(test_res['truth_de'][pert_idx], 0.75, axis=0)
            q45 = np.quantile(test_res['truth_de'][pert_idx], 0.45, axis=0)
            q55 = np.quantile(test_res['truth_de'][pert_idx], 0.55, axis=0)
            q40 = np.quantile(test_res['truth_de'][pert_idx], 0.4, axis=0)
            q60 = np.quantile(test_res['truth_de'][pert_idx], 0.6, axis=0)
    
            zero_des = np.intersect1d(np.where(min_ == 0)[0], np.where(max_ == 0)[0])
            nonzero_des = np.setdiff1d(list(range(20)), zero_des)
            if len(nonzero_des) > 0:
                # 방향성 일치율
                direc_change = np.abs(np.sign(pred_mean[nonzero_des] - ctrl[0][de_idx][nonzero_des]) -
                                      np.sign(true_mean[nonzero_des] - ctrl[0][de_idx][nonzero_des]))
                pert_metric[pert]['frac_correct_direction_20_nonzero'] = (direc_change == 0).sum() / len(nonzero_des)

                # 범위 내 예측률
                pert_metric[pert]['frac_in_range'] = np.mean((pred_mean[nonzero_des] >= min_[nonzero_des]) & (pred_mean[nonzero_des] <= max_[nonzero_des]))
                pert_metric[pert]['frac_in_range_45_55'] = np.mean((pred_mean[nonzero_des] >= q45[nonzero_des]) & (pred_mean[nonzero_des] <= q55[nonzero_des]))
                pert_metric[pert]['frac_in_range_40_60'] = np.mean((pred_mean[nonzero_des] >= q40[nonzero_des]) & (pred_mean[nonzero_des] <= q60[nonzero_des]))
                pert_metric[pert]['frac_in_range_25_75'] = np.mean((pred_mean[nonzero_des] >= q25[nonzero_des]) & (pred_mean[nonzero_des] <= q75[nonzero_des]))

                # z-score 오차 분석
                zero_idx = np.where(std > 0)[0]
                sigma = np.abs(pred_mean[zero_idx] - mean[zero_idx]) / std[zero_idx]
                pert_metric[pert]['mean_sigma'] = np.mean(sigma)
                pert_metric[pert]['std_sigma'] = np.std(sigma)
                pert_metric[pert]['frac_sigma_below_1'] = 1 - (sigma > 1).sum() / len(zero_idx)
                pert_metric[pert]['frac_sigma_below_2'] = 1 - (sigma > 2).sum() / len(zero_idx)
    
            ## (8) 변화량 기반 상관성 평가 (pert - ctrl)
            for m, fct in metric2fct.items():
                if m != 'mse':
                    # 전체 유전자
                    val = fct(test_res['pred'][pert_idx].mean(0) - ctrl[0], test_res['truth'][pert_idx].mean(0) - ctrl[0])[0]
                    pert_metric[pert][m + '_delta'] = val if not np.isnan(val) else 0
                    # DE 유전자
                    val = fct(test_res['pred'][pert_idx].mean(0)[de_idx] - ctrl[0][de_idx], test_res['truth'][pert_idx].mean(0)[de_idx] - ctrl[0][de_idx])[0]
                    pert_metric[pert][m + '_delta_de'] = val if not np.isnan(val) else 0
    
            ## (9) fold change gap 분석 (log2 기준 x 아님)
            pert_mean = np.mean(test_res['truth'][pert_idx], axis=0)
            fold_change = pert_mean / ctrl
            fold_change[np.isnan(fold_change)] = 0
            fold_change[np.isinf(fold_change)] = 0
            fold_change[0][np.where(pert_mean < 0.5)[0]] = 0  # 너무 낮은 값 제거
    
            # fold change 기준 조건들: 전체, down < 0.1, up > 10 등
            for label, condition in {
                'all': np.where(fold_change[0] > 0)[0],
                'downreg_0.33': np.intersect1d(np.where(fold_change[0] < 0.333)[0], np.where(fold_change[0] > 0)[0]),
                'downreg_0.1': np.intersect1d(np.where(fold_change[0] < 0.1)[0], np.where(fold_change[0] > 0)[0]),
                'upreg_3': np.where(fold_change[0] > 3)[0],
                'upreg_10': np.where(fold_change[0] > 10)[0]
            }.items():
                if len(condition) > 0:
                    pred_fc = test_res['pred'][pert_idx].mean(0)[condition]
                    true_fc = test_res['truth'][pert_idx].mean(0)[condition]
                    ctrl_fc = ctrl[0][condition]
                    gap = np.abs(pred_fc / ctrl_fc - true_fc / ctrl_fc)
                    pert_metric[pert][f'fold_change_gap_{label}'] = np.mean(gap)
    
            ## (10) HVG 기반 성능 평가 (top N genes)
            for top_k in [1000, 500, 250, 100, 50]:
                top_k_genes = most_variable_genes[-top_k:]  # 상위 top_k 유전자 추출
    
                for m, fct in metric2fct.items():
                    if m != 'mse':
                        # delta: (pred - ctrl) vs (truth - ctrl)
                        val = fct(
                            test_res['pred'][pert_idx].mean(0)[top_k_genes] - ctrl[0][top_k_genes],
                            test_res['truth'][pert_idx].mean(0)[top_k_genes] - ctrl[0][top_k_genes]
                        )[0]
                        pert_metric[pert][f'{m}_delta_top{top_k}_hvg'] = val if not np.isnan(val) else 0
    
                        # absolute value correlation
                        val = fct(
                            test_res['pred'][pert_idx].mean(0)[top_k_genes],
                            test_res['truth'][pert_idx].mean(0)[top_k_genes]
                        )[0]
                        pert_metric[pert][f'{m}_top{top_k}_hvg'] = val if not np.isnan(val) else 0
    
                    else:
                        # MSE는 그대로 사용
                        val = fct(
                            test_res['pred'][pert_idx].mean(0)[top_k_genes],
                            test_res['truth'][pert_idx].mean(0)[top_k_genes]
                        )
                        pert_metric[pert][f'{m}_top{top_k}_hvg'] = val
    
    
            ## (11) Top-K DE 유전자에 대한 정량적 평가
            for k, d_idx in de_idx_map.items():
                for m, fct in metric2fct.items():
                    if m != 'mse':
                        delta_corr = fct(test_res['pred'][pert_idx].mean(0)[d_idx] - ctrl[0][d_idx],
                                         test_res['truth'][pert_idx].mean(0)[d_idx] - ctrl[0][d_idx])[0]
                        corr = fct(test_res['pred'][pert_idx].mean(0)[d_idx], test_res['truth'][pert_idx].mean(0)[d_idx])[0]
                        pert_metric[pert][f'{m}_delta_top{k}_de'] = delta_corr if not np.isnan(delta_corr) else 0
                        pert_metric[pert][f'{m}_top{k}_de'] = corr if not np.isnan(corr) else 0
                    else:
                        mse_val = fct(test_res['pred'][pert_idx].mean(0)[d_idx] - ctrl[0][d_idx], test_res['truth'][pert_idx].mean(0)[d_idx] - ctrl[0][d_idx])
                        pert_metric[pert][f'{m}_top{k}_de'] = mse_val

    return pert_metric


# def loss_fct(pred, y, perts, hvgs, ctrl=None, direction_lambda=1e-1, hvg_weight=2.0, dict_filter=None, loss_version=None, visible=False):
#     """
#     Main MSE Loss function, includes direction loss

#     Args:
#         pred (torch.tensor): predicted gene expression values (batch_size x n_genes)
#         y (torch.tensor): true gene expression values (batch_size x n_genes)
#         perts (list): list of perturbation labels for each sample in batch
#         ctrl (torch.tensor): control expression vector (1 x n_genes)
#         direction_lambda (float): weight of directionality loss term
#         dict_filter (dict): maps perturbation name to selected gene indices for evaluation
#         visible (bool): if True, print debug information
#     """
#     device = pred.device
#     gamma = 2 # Degree of power for autofocus loss (MSE -> (x^2+gamma) = x^4)
#     mse_p = torch.nn.MSELoss() # (not used directly, may be legacy)
#     perts = np.array(perts) # Convert perturbation list to numpy array for indexing
#     losses = torch.tensor(0.0, requires_grad=True, device=device)
#     hvg_losses = torch.tensor(0.0, device=device)
#     autofocus_loss = torch.tensor(0.0, device=device)
#     autofocus_hvg_loss = torch.tensor(0.0, device=device)
#     direction_loss = torch.tensor(0.0, device=device)
#     direction_hvg_loss = torch.tensor(0.0, device=device)
#     # Set of unique perturbations
#     unique_perts = set(perts) # Unique perturbation types in the batch

#     for p in set(perts):
#         pert_idx = np.where(perts == p)[0] # Indices in batch corresponding to perturbation p
#         if p != 'ctrl':
#             retain_idx = dict_filter[p] # Select relevant gene indices for this perturbation
            
#             # hvg_idx = de_filter[p]
#             if visible == True:
#                 print(f"Shape of pred: {pred.shape}")
#                 print(f'pert: {p}, pert index: {pert_idx}, non-zero: {retain_idx}')
#                 print(f"Type of retain_idx: {type(retain_idx)}")
#                 print(f"Data type of retain_idx: {retain_idx.dtype}")
#                 print(f"Sample of retain_idx: {retain_idx[:10]}")
                
#             if loss_version == 2:
#                 # 교집합: retain_idx ∩ hvg_n_idx
#                 retain_idx = sorted(retain_idx)  # 보장
#                 retain_idx_set = set(retain_idx)
#                 hvg_idx_set = set(hvgs)
                
#                 # retain_idx 내에서 HVG인 유전자만 추출
#                 selected_idx = sorted(list(retain_idx_set & hvg_idx_set))
#                 # pred_p는 retain_idx 기준의 gene subset이므로,
#                 # hvg_selected_idx는 selected_idx가 retain_idx 내에서 몇 번째인지 매핑 필요
#                 selected_idx_map = {g: i for i, g in enumerate(selected_idx)}
                
#                 ## 검증용 코드 ##
#                 max_index = pred.shape[1]  # gene 개수
#                 assert all(i < max_index for i in selected_idx), f"selected_idx out of bounds: {selected_idx}"
#                 for g in selected_idx:
#                     if g not in selected_idx_map:
#                         raise ValueError(f"Gene {g} not in selected_idx_map. retain_idx: {selected_idx}")
#                 ##-----------##
                
#                 retain_idx = torch.tensor(retain_idx, device=pred.device)
                
#                 hvg_selected_idx = [selected_idx_map[g] for g in selected_idx if g in hvgs]
#                 selected_idx = torch.tensor(selected_idx, device=pred.device)
#                 hvg_selected_idx = torch.tensor(hvg_selected_idx, device=pred.device)
#                 # 예측값과 정답에서 해당 유전자 인덱스만 추출
#                 pred_p = pred[pert_idx][:, selected_idx]
#                 y_p = y[pert_idx][:, selected_idx]
#                 if visible:
#                     print(f"[DEBUG] pred.shape: {pred.shape}")
#                     print(f"[DEBUG] selected_idx: {selected_idx[:10]} (len={len(selected_idx)})")
#                     print(f"[DEBUG] hvg_selected_idx: {hvg_selected_idx[:10]} (len={len(hvg_selected_idx)})")
#                     print(f"[DEBUG] max_index allowed: {max_index}")
#                     if any(i >= max_index or i < 0 for i in hvg_selected_idx):
#                         raise ValueError(f"[ERROR] hvg_selected_idx has out-of-bounds index: {hvg_selected_idx}")

#             else:
#                 # Filter predictions and labels to retained gene indices
#                 pred_p = pred[pert_idx][:, retain_idx]
#                 y_p = y[pert_idx][:, retain_idx]
#             # if loss_version == 2:
#             #     pred_p = pred_p[:, hvg_idx]
#             #     y_p = y_p[:, hvg_idx]
#             if visible == True:
#                 print(f"Predicted Y: {pred_p} | True Y: {y_p}")
#         else:
#             if loss_version == 2:
#                 selected_idx = list(range(pred.shape[1]))
#                 hvg_selected_idx = list(hvgs)
#             # For control group, use all genes
#             pred_p = pred[pert_idx]
#             y_p = y[pert_idx]

#         # MSE 손실 계산
#         # Base MSE
#         # MSE 손실 계산
#         base_mse = (pred_p - y_p) ** (2 + gamma)
        
#         # 전체 loss는 가중치 적용
#         if loss_version == 2:
#             weights = torch.ones_like(base_mse)
#             if len(hvg_selected_idx) > 0:
#                 hvg_mask = torch.zeros_like(base_mse[0])
#                 hvg_mask[hvg_selected_idx] = hvg_weight - 1.0
#                 weights += hvg_mask.unsqueeze(0)  # shape (1, n_genes)로 broadcasting
#             base_mse = base_mse * weights
        
#         autofocus = torch.sum(base_mse) / pred_p.shape[0] / pred_p.shape[1]
#         autofocus_loss = autofocus_loss + autofocus
#         # losses += autofocus ! in-place 
#         losses = losses + autofocus
        
#         # HVG 전용 loss는 가중치 없이 HVG 유전자만 추출
#         if loss_version == 2:
#             if len(hvg_selected_idx) > 0:
#                 hvg_mse = (pred_p[:, hvg_selected_idx] - y_p[:, hvg_selected_idx]) ** (2 + gamma)
#                 autofocus_hvg = torch.sum(hvg_mse) / pred_p.shape[0] / len(hvg_selected_idx)
#             else:
#                 autofocus_hvg = torch.tensor(0.0, device=pred.device)
        
#             autofocus_hvg_loss = autofocus_hvg_loss + autofocus_hvg
#             hvg_losses = hvg_losses + autofocus_hvg


#         # 방향성 손실 계산
#         # L = L(autofocus) + lambda * L(direction)
#         # L(direction) = sign(Pred - ctrl) - sign(True - ctrl)
#         ## direction loss
#         if loss_version == 2:
#             direction_diff = (torch.sign(y_p - ctrl[selected_idx]) - torch.sign(pred_p - ctrl[selected_idx])) ** 2

#             # 가중치 벡터 (1 + (hvg_weight - 1) at HVG positions)
#             dir_weights = torch.ones_like(direction_diff)
#             if len(hvg_selected_idx) > 0:
#                 mask = torch.zeros_like(direction_diff[0])
#                 mask[hvg_selected_idx] = hvg_weight - 1.0
#                 dir_weights += mask  # broadcasting over batch

#             direction = torch.sum(direction_lambda * direction_diff * dir_weights) / pred_p.shape[0] / pred_p.shape[1]

#             # HVG 전용 direction 손실 계산
#             if len(hvg_selected_idx) > 0:
#                 direction_hvg = torch.sum(direction_lambda * direction_diff[:, hvg_selected_idx]) / pred_p.shape[0] / pred_p.shape[1]
#             else:
#                 direction_hvg = torch.tensor(0.0, device=pred.device)
#             direction_hvg_loss = direction_hvg_loss + direction_hvg
#             hvg_losses =  hvg_losses + direction_hvg


#         else:
#             if (p!= 'ctrl'):
#                 direction = torch.sum(direction_lambda *
#                                     (torch.sign(y_p - ctrl[retain_idx]) -
#                                      torch.sign(pred_p - ctrl[retain_idx]))**2)/\
#                                      pred_p.shape[0]/pred_p.shape[1]

#             else:
#                 direction = torch.sum(direction_lambda * (torch.sign(y_p - ctrl) -
#                                                     torch.sign(pred_p - ctrl))**2)/\
#                                                     pred_p.shape[0]/pred_p.shape[1]
#         direction_loss = direction_loss + direction
#         losses = losses + direction

#     if loss_version == 2:
#         return (
#             losses / len(unique_perts),
#             autofocus_loss / len(unique_perts),
#             direction_loss / len(unique_perts),
#             hvg_losses / len(unique_perts),
#             autofocus_hvg_loss / len(unique_perts),
#             direction_hvg_loss / len(unique_perts)
#         )
        
#     return losses / len(unique_perts), autofocus_loss / len(unique_perts), direction_loss / len(unique_perts)




# def loss_fct(pred, y, perts, hvgs, ctrl=None, direction_lambda=1e-1, direction_alpha = 0.5, direction_method=None, hvg_weight=2.0, dict_filter=None, loss_version=None, visible=False):
#     """
#     Main MSE Loss function, includes direction loss

#     Args:
#         pred (torch.tensor): predicted gene expression values (batch_size x n_genes)
#         y (torch.tensor): true gene expression values (batch_size x n_genes)
#         perts (list): list of perturbation labels for each sample in batch
#         ctrl (torch.tensor): control expression vector (1 x n_genes)
#         direction_lambda (float): weight of directionality loss term
#         dict_filter (dict): maps perturbation name to selected gene indices for evaluation
#         visible (bool): if True, print debug information
#     """
#     device = pred.device
#     gamma = 2 # Degree of power for autofocus loss (MSE -> (x^2+gamma) = x^4)
#     # mse_p = torch.nn.MSELoss() # (not used directly, may be legacy)
#     perts = np.array(perts) # Convert perturbation list to numpy array for indexing
#     losses = torch.tensor(0.0, requires_grad=True, device=device)
#     hvg_losses = torch.tensor(0.0, device=device)
#     autofocus_loss = torch.tensor(0.0, device=device)
#     autofocus_hvg_loss = torch.tensor(0.0, device=device)
#     direction_loss = torch.tensor(0.0, device=device)
#     direction_hvg_loss = torch.tensor(0.0, device=device)
#     # Set of unique perturbations
#     unique_perts = set(perts) # Unique perturbation types in the batch

#     for p in set(perts):
#         pert_idx = np.where(perts == p)[0] # Indices in batch corresponding to perturbation p
#         if p != 'ctrl':
#             retain_idx = dict_filter[p] # Select relevant gene indices for this perturbation
            
#             # hvg_idx = de_filter[p]
#             if visible == True:
#                 print(f"Shape of pred: {pred.shape}")
#                 print(f'pert: {p}, pert index: {pert_idx}, non-zero: {retain_idx}')
#                 print(f"Type of retain_idx: {type(retain_idx)}")
#                 print(f"Data type of retain_idx: {retain_idx.dtype}")
#                 print(f"Sample of retain_idx: {retain_idx[:10]}")


                
#             ##-----------------------------------------------------------------------------------------------##
#             ## HVG PART ##
#             # if loss_version == 2:
#             # 교집합: retain_idx ∩ hvg_n_idx
#             retain_idx = sorted(retain_idx)  # 보장
#             retain_idx_set = set(retain_idx)
#             hvg_idx_set = set(hvgs)
            
#             # retain_idx 내에서 HVG인 유전자만 추출
#             selected_idx = sorted(list(retain_idx_set & hvg_idx_set))
#             # pred_p는 retain_idx 기준의 gene subset이므로,
#             # hvg_selected_idx는 selected_idx가 retain_idx 내에서 몇 번째인지 매핑 필요
#             selected_idx_map = {g: i for i, g in enumerate(selected_idx)}
            
#             ## 검증용 코드 ##
#             max_index = pred.shape[1]  # gene 개수
#             assert all(i < max_index for i in selected_idx), f"selected_idx out of bounds: {selected_idx}"
#             for g in selected_idx:
#                 if g not in selected_idx_map:
#                     raise ValueError(f"Gene {g} not in selected_idx_map. retain_idx: {selected_idx}")
#             ##-----------##
            
#             retain_idx = torch.tensor(retain_idx, device=pred.device)
#             pred_p = pred[pert_idx][:, retain_idx]
#             y_p = y[pert_idx][:, retain_idx]
            
#             hvg_selected_idx = [selected_idx_map[g] for g in selected_idx if g in hvgs]
#             selected_idx = torch.tensor(selected_idx, device=pred.device)
#             hvg_selected_idx = torch.tensor(hvg_selected_idx, device=pred.device)
#             # 예측값과 정답에서 해당 유전자 인덱스만 추출
#             pred_p = pred[pert_idx][:, selected_idx]
#             y_p = y[pert_idx][:, selected_idx]
#             if visible:
#                 print(f"[DEBUG] pred.shape: {pred.shape}")
#                 print(f"[DEBUG] selected_idx: {selected_idx[:10]} (len={len(selected_idx)})")
#                 print(f"[DEBUG] hvg_selected_idx: {hvg_selected_idx[:10]} (len={len(hvg_selected_idx)})")
#                 print(f"[DEBUG] max_index allowed: {max_index}")
#                 if any(i >= max_index or i < 0 for i in hvg_selected_idx):
#                     raise ValueError(f"[ERROR] hvg_selected_idx has out-of-bounds index: {hvg_selected_idx}")
#             ##-----------------------------------------------------------------------------------------------##

            
#             # else:
#             #     # Filter predictions and labels to retained gene indices
#             #     pred_p = pred[pert_idx][:, retain_idx]
#             #     y_p = y[pert_idx][:, retain_idx]
            
#             # if loss_version == 2:
#             #     pred_p = pred_p[:, hvg_idx]
#             #     y_p = y_p[:, hvg_idx]
#             if visible == True:
#                 print(f"Predicted Y: {pred_p} | True Y: {y_p}")
#         else:
#             # if loss_version == 2:
#             selected_idx = list(range(pred.shape[1]))
#             hvg_selected_idx = list(hvgs)
#             # For control group, use all genes
#             pred_p = pred[pert_idx]
#             y_p = y[pert_idx]

#         # MSE 손실 계산
#         base_mse = (pred_p - y_p).pow(2 + gamma)  # Base MSE
        
#         # HVG 가중치 mask (HVG가 있을 때만)
#         if len(hvg_selected_idx) > 0:
#             hvg_mask = torch.zeros_like(base_mse[0])
#             hvg_mask[hvg_selected_idx] = hvg_weight - 1.0
        
#         # 전체 loss 가중치 적용
#         if loss_version == 2:
#             weights = torch.ones_like(base_mse)
#             if len(hvg_selected_idx) > 0:
#                 weights += hvg_mask.unsqueeze(0)  # broadcasting over batch
#             base_mse = base_mse * weights
        
#         # 전체 autofocus loss
#         autofocus = torch.sum(base_mse) / pred_p.shape[0] / pred_p.shape[1]
#         autofocus_loss = autofocus_loss + autofocus # 모니터링용 autofocus loss 기록
#         losses = losses + autofocus
        
#         # HVG 전용 MSE loss (가중치 없이)
#         if len(hvg_selected_idx) > 0:
#             hvg_mse = (pred_p[:, hvg_selected_idx] - y_p[:, hvg_selected_idx]).pow(2 + gamma)
#             autofocus_hvg = torch.sum(hvg_mse) / pred_p.shape[0] / len(hvg_selected_idx)
#         else:
#             autofocus_hvg = torch.tensor(0.0, device=pred.device)
        
#         autofocus_hvg_loss = autofocus_hvg_loss + autofocus_hvg # 모니터링용 HVG autofocus loss 기록
#         hvg_losses = hvg_losses + autofocus_hvg
        
#         # Direction loss 계산
#         if direction_method == 'sign':
#             direction_diff = (torch.sign(y_p - ctrl[selected_idx]) - torch.sign(pred_p - ctrl[selected_idx])).pow(2)
#         elif direction_method == 'tanh':
#             direction_diff = (torch.tanh(y_p - ctrl[selected_idx]) - torch.tanh(pred_p - ctrl[selected_idx])).pow(2)
#         elif direction_method == 'hybrid':
#             diff_sign = (torch.sign(y_p - ctrl[selected_idx]) -
#                          torch.sign(pred_p - ctrl[selected_idx])).pow(2)
#             diff_tanh = (torch.tanh(y_p - ctrl[selected_idx]) -
#                          torch.tanh(pred_p - ctrl[selected_idx])).pow(2)
#             direction_diff = direction_alpha * diff_sign + (1 - direction_alpha) * diff_tanh
        
#         else:
#             raise ValueError(f"Invalid direction_method: {direction_method}")
            
#         if loss_version == 2:
#             dir_weights = torch.ones_like(direction_diff)
#             if len(hvg_selected_idx) > 0:
#                 dir_weights += hvg_mask.unsqueeze(0)  # broadcasting
#             direction = torch.sum(direction_lambda * direction_diff * dir_weights)/ pred_p.shape[0] / pred_p.shape[1]
#         else:
#             direction = torch.sum(direction_lambda * direction_diff)/ pred_p.shape[0] / pred_p.shape[1]
        
        
#         # HVG 전용 Direction loss
#         if len(hvg_selected_idx) > 0:
#             direction_hvg = torch.sum(direction_lambda * direction_diff[:, hvg_selected_idx])/ pred_p.shape[0] / len(hvg_selected_idx)
#         else:
#             direction_hvg = torch.tensor(0.0, device=pred.device)
        
#         direction_hvg_loss = direction_hvg_loss + direction_hvg
#         hvg_losses = hvg_losses + direction_hvg

#         direction_loss = direction_loss + direction
#         losses = losses + direction


#     return (
#         losses / len(unique_perts),
#         autofocus_loss / len(unique_perts),
#         direction_loss / len(unique_perts),
#         hvg_losses / len(unique_perts),
#         autofocus_hvg_loss / len(unique_perts),
#         direction_hvg_loss / len(unique_perts)
#     )
def loss_fct(pred, y, perts, hvgs, ctrl=None,
             direction_lambda=1e-1, direction_alpha=0.5,
             direction_method=None, hvg_weight=2.0,
             dict_filter=None, loss_version=None,
             visible=False):
    """
    이 함수는 예측된 유전자 발현값과 실제값 간의 차이를 기반으로 손실을 계산합니다.
    1. 기본 MSE (autofocus) 손실을 계산하고,
    2. 고변이 유전자(HVG)에 대한 가중치 손실을 별도로 고려하며,
    3. perturbation 방향성 보존 여부에 대한 directionality 손실도 포함합니다.
    옵션에 따라 HVG에 추가 가중치를 주거나, direction 손실 방식(sign/tanh/hybrid)을 선택할 수 있습니다.
    
    Main MSE Loss function with directionality and HVG weighting.

    Args:
        pred (torch.tensor): predicted gene expression (batch x n_genes)
        y (torch.tensor): true gene expression (batch x n_genes)
        perts (list): perturbation labels
        hvgs (list[int]): HVG gene indices
        ctrl (torch.tensor): control gene expression vector (1 x n_genes)
        direction_lambda (float): direction loss weight
        direction_alpha (float): for hybrid method
        direction_method (str): 'sign', 'tanh', or 'hybrid'
        hvg_weight (float): weight multiplier for HVGs
        dict_filter (dict): per perturbation → DE gene indices
        loss_version (int): if == 2, enable HVG weighting
        visible (bool): print debug info
    """

    device = pred.device  # 모델의 디바이스 설정
    gamma = 2  # autofocus 손실의 제곱 계수 (x^2+gamma → x^4처럼 동작)
    perts = np.array(perts)  # list to numpy

    # 전체 손실 및 항목별 손실 초기화
    losses = torch.tensor(0.0, requires_grad=True, device=device)
    hvg_losses = torch.tensor(0.0, device=device)
    autofocus_loss = torch.tensor(0.0, device=device)
    autofocus_hvg_loss = torch.tensor(0.0, device=device)
    direction_loss = torch.tensor(0.0, device=device)
    direction_hvg_loss = torch.tensor(0.0, device=device)

    unique_perts = set(perts)  # perturbation 종류별로 나누어 계산

    for p in unique_perts:
        pert_idx = np.where(perts == p)[0]  # 해당 perturbation 샘플 인덱스

        if p != 'ctrl':
            retain_idx = dict_filter[p]  # 해당 perturbation에 대한 DE 유전자 index
            retain_idx = sorted(retain_idx)
            retain_idx_tensor = torch.tensor(retain_idx, device=pred.device)

            pred_p = pred[pert_idx][:, retain_idx_tensor]  # 예측값 중 retain index만 추출
            y_p = y[pert_idx][:, retain_idx_tensor]        # 실제값도 동일 추출

            # HVG 중 retain_idx에 포함되는 유전자만 필터링
            hvg_idx_set = set(hvgs)
            hvg_selected_genes = sorted(list(set(retain_idx) & hvg_idx_set))  # retain ∩ HVG
            selected_idx_map = {g: i for i, g in enumerate(retain_idx)}  # retain index → local index 매핑
            hvg_selected_idx = [selected_idx_map[g] for g in hvg_selected_genes]
            hvg_selected_idx_tensor = torch.tensor(hvg_selected_idx, device=pred.device)

            if visible:
                print(f"[DEBUG] p: {p}, retain_idx={len(retain_idx)}, HVG={len(hvg_selected_idx)}")

        else:
            # ctrl은 전체 유전자 사용
            retain_idx_tensor = torch.arange(pred.shape[1], device=pred.device)
            pred_p = pred[pert_idx]
            y_p = y[pert_idx]
            hvg_selected_idx_tensor = torch.tensor(sorted(hvgs), device=pred.device)

        # ----- 기본 MSE (autofocus) -----
        base_mse = (pred_p - y_p).pow(2 + gamma)

        if loss_version == 2 and len(hvg_selected_idx_tensor) > 0:
            hvg_mask = torch.zeros_like(base_mse[0])
            hvg_mask[hvg_selected_idx_tensor] = hvg_weight - 1.0  # HVG 가중치 설정
            weights = 1.0 + hvg_mask.unsqueeze(0)  # 전체 weight mask
            base_mse = base_mse * weights  # 가중치 적용

        autofocus = base_mse.mean()  # 평균 MSE
        autofocus_loss = autofocus_loss + autofocus
        losses = losses + autofocus

        # ----- HVG 전용 MSE 계산 -----
        if len(hvg_selected_idx_tensor) > 0:
            hvg_mse = (pred_p[:, hvg_selected_idx_tensor] - y_p[:, hvg_selected_idx_tensor]).pow(2 + gamma)
            autofocus_hvg = hvg_mse.mean()
        else:
            autofocus_hvg = torch.tensor(0.0, device=pred.device)

        autofocus_hvg_loss = autofocus_hvg_loss + autofocus_hvg
        hvg_losses = hvg_losses + autofocus_hvg

        # ----- 방향성 손실 (directional loss) -----
        ctrl_slice = ctrl[retain_idx_tensor]  # control 발현값 추출

        if direction_method == 'sign':
            direction_diff = (torch.sign(y_p - ctrl_slice) - torch.sign(pred_p - ctrl_slice)).pow(2)
        elif direction_method == 'tanh':
            direction_diff = (torch.tanh(y_p - ctrl_slice) - torch.tanh(pred_p - ctrl_slice)).pow(2)
        elif direction_method == 'hybrid':
            diff_sign = (torch.sign(y_p - ctrl_slice) - torch.sign(pred_p - ctrl_slice)).pow(2)
            diff_tanh = (torch.tanh(y_p - ctrl_slice) - torch.tanh(pred_p - ctrl_slice)).pow(2)
            direction_diff = direction_alpha * diff_sign + (1 - direction_alpha) * diff_tanh
        else:
            raise ValueError(f"Invalid direction_method: {direction_method}")

        # HVG 가중치 적용
        if loss_version == 2 and len(hvg_selected_idx_tensor) > 0:
            dir_weights = torch.ones_like(direction_diff)
            dir_weights = dir_weights + hvg_mask.unsqueeze(0)
            direction = (direction_lambda * direction_diff * dir_weights).mean()
        else:
            direction = (direction_lambda * direction_diff).mean()

        direction_loss = direction_loss + direction
        losses = losses + direction

        # ----- HVG 전용 방향성 손실 -----
        if len(hvg_selected_idx_tensor) > 0:
            direction_hvg = (direction_lambda * direction_diff[:, hvg_selected_idx_tensor]).mean()
        else:
            direction_hvg = torch.tensor(0.0, device=pred.device)

        direction_hvg_loss = direction_hvg_loss + direction_hvg
        hvg_losses = hvg_losses + direction_hvg

    # ----- 최종 평균 손실 반환 -----
    return (
        losses / len(unique_perts),              # 전체 loss
        autofocus_loss / len(unique_perts),      # 기본 MSE
        direction_loss / len(unique_perts),      # direction 손실
        hvg_losses / len(unique_perts),          # HVG 관련 전체 손실 (MSE + direction)
        autofocus_hvg_loss / len(unique_perts),  # HVG 전용 MSE
        direction_hvg_loss / len(unique_perts)   # HVG 전용 direction 손실
    )




    
def uncertainty_loss_fct(pred, logvar, y, perts, reg = 0.1, ctrl = None,
                         direction_lambda = 1e-3, dict_filter = None):
    """
    Uncertainty loss function

    Args:
        pred (torch.tensor): predicted values
        logvar (torch.tensor): log variance
        y (torch.tensor): true values
        perts (list): list of perturbations
        reg (float): regularization parameter
        ctrl (str): control perturbation
        direction_lambda (float): direction loss weight hyperparameter
        dict_filter (dict): dictionary of perturbations to conditions

    """
    gamma = 2                     
    perts = np.array(perts)
    losses = torch.tensor(0.0, requires_grad=True).to(pred.device)
    for p in set(perts):
        if p!= 'ctrl':
            retain_idx = dict_filter[p]
            pred_p = pred[np.where(perts==p)[0]][:, retain_idx]
            y_p = y[np.where(perts==p)[0]][:, retain_idx]
            logvar_p = logvar[np.where(perts==p)[0]][:, retain_idx]
        else:
            pred_p = pred[np.where(perts==p)[0]]
            y_p = y[np.where(perts==p)[0]]
            logvar_p = logvar[np.where(perts==p)[0]]
                         
        # uncertainty based loss
        losses += torch.sum((pred_p - y_p)**(2 + gamma) + reg * torch.exp(
            -logvar_p)  * (pred_p - y_p)**(2 + gamma))/pred_p.shape[0]/pred_p.shape[1]
                         
        # direction loss                 
        if p!= 'ctrl':
            losses += torch.sum(direction_lambda *
                                (torch.sign(y_p - ctrl[retain_idx]) -
                                 torch.sign(pred_p - ctrl[retain_idx]))**2)/\
                                 pred_p.shape[0]/pred_p.shape[1]
        else:
            losses += torch.sum(direction_lambda *
                                (torch.sign(y_p - ctrl) -
                                 torch.sign(pred_p - ctrl))**2)/\
                                 pred_p.shape[0]/pred_p.shape[1]
            
    return losses/(len(set(perts)))


def save_checkpoint(model, optimizer, epoch, loss, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at epoch {epoch} to: {path}")

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M")


import math
import torch
import numpy as np
def compute_avg_loss_on_loader(model, loader, device, config, visible_cfg=None):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch.to(device)
            y = batch.y
            if model.uncertainty:
                pred, logvar = model(batch)
                loss = uncertainty_loss_fct(
                    pred, logvar, y, batch.pert,
                    reg=model.param_uncertainty_reg,
                    ctrl=model.ctrl_expression,
                    dict_filter=model.dict_filter,
                    direction_lambda=model.param_direction_lambda,
                )
            else:
                visible = visible_cfg if visible_cfg is not None else []
                pred = model(batch, visible)
                loss, *_ = loss_fct(
                    pred, y, batch.pert, model.dataloader.hvg_n_idx,
                    ctrl=model.ctrl_expression,
                    dict_filter=model.dict_filter,
                    loss_version=model.loss_version,
                    direction_lambda=config['direction_lambda'],
                    direction_alpha=config['direction_alpha'],
                    direction_method=config['direction_method'],
                    hvg_weight=config['hvg_weight'],
                    visible=visible
                )
            total += float(loss.item())
            n += 1
    return total / max(n, 1)

class BestSaver:
    def __init__(self, monitor: str = "val_pearson", mode: str = "max", min_delta: float = 0.0):
        assert mode in ["min", "max"]
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self.best_value = -math.inf if mode == "max" else math.inf
        self.best_epoch = -1
        self.best_path = None

    def is_better(self, current: float) -> bool:
        if self.mode == "max":
            return current > (self.best_value + self.min_delta)
        else:
            return current < (self.best_value - self.min_delta)

class EarlyStopping:
    def __init__(self, patience: int = 20, mode: str = "max", min_delta: float = 0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best = -math.inf if mode == "max" else math.inf
        self.should_stop = False

    def step(self, value: float):
        improved = (value > self.best + self.min_delta) if self.mode == "max" else (value < self.best - self.min_delta)
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
