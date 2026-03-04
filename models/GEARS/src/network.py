from multiprocessing import Pool  # 병렬 처리 모듈
from tqdm import tqdm  # 진행률 표시 라이브러리
import pandas as pd  # 데이터프레임 처리
import numpy as np  # 수치 연산
import os  # 경로 및 파일 시스템 처리
from config import BASE_DIR, DATA_DIR, RAW_DATA_PATH, GEARS_DATA_PATH, OUTPUT_DIR  # 설정 값 로드

def get_edge_list(args, minimum=0.1):
    """
    Get gene ontology edge list
    """
    """
    두 유전자 간의 GO-term 유사도를 기반으로 edge list를 구성하는 함수입니다.
    유사도가 최소 기준(minimum) 이상일 경우 edge로 추가됩니다.
    """
    g1, gene2go = args  # 기준 유전자(g1)와 전체 gene-to-GO 매핑 딕셔너리
    edge_list = []  # edge 리스트 초기화
    for g2 in gene2go.keys():  # 모든 g2 유전자에 대해
        score = len(gene2go[g1].intersection(gene2go[g2])) / len(
            gene2go[g1].union(gene2go[g2]))  # Jaccard 유사도 계산
        if score > minimum:  # 최소 유사도 기준 이상이면
            edge_list.append((g1, g2, score))  # edge 추가
    return edge_list  # 해당 유전자 g1의 edge 목록 반환


def make_condition_edges(method, data_name, minimum=0.1, num_workers=0, save=True):
    """
    description: network를 구성
        method: network generation method.
            ex) "go", "expression"
        num_workers: more than 0 workers to activate 
    output:
        source	target	    importance
    0	node1	node1	    1.000000
    1	node1   node2	    0.181818
    2	node1   node3	    0.121951
    3	node1   node4	    0.108108
    4	node1   node5	    0.125000
    """
    """
    유전자 간 관계 그래프(edge list)를 구성하는 함수입니다.
    method에 따라 GO-term 기반 또는 expression 기반 네트워크를 생성하고, 필요 시 저장합니다.
    """

    filename = GEARS_DATA_PATH + '/data/edges_for_condition_' + method + '_' + data_name + '.csv'  # 결과 파일 경로 설정

    if os.path.exists(filename):  # 결과 파일이 이미 존재할 경우
        print("Now loading exist data from directory: ", filename)  # 로드 안내 출력
        return pd.read_csv(filename)  # 파일에서 데이터 로드 후 반환

    if method == "go":  # GO-term 기반 네트워크 생성 시
        # GO-term 파일 로드
        with open(os.path.join(GEARS_DATA_PATH, '/gene2go_all.pkl'), 'rb') as f:
            gene_go_all = pickle.load(f)  # 전체 gene-GO 매핑 로드
        gene_go_all = {i: gene_go_all[i] for i in gene_condition_names}  # condition 관련 유전자만 필터링

        print('Creating GO-term based graph, this can take a few minutes')  # 시작 메시지 출력

        if num_workers > 0:  # 멀티프로세싱 사용
            with Pool(num_workers) as p:  # 프로세스 풀 생성
                all_edge_list = list(
                    tqdm(p.imap(get_edge_list, ((g, gene_go_all) for g in gene_go_all.keys())),
                         total=len(gene_go_all.keys())))  # 병렬로 edge 리스트 생성
                edge_list = []  # 최종 edge 리스트
                for i in all_edge_list:
                    edge_list = edge_list + i  # 모든 결과를 하나의 리스트로 병합
                df_edge_list = pd.DataFrame(edge_list).rename(columns={0: 'source', 1: 'target', 2: 'importance'})  # DataFrame 생성 및 컬럼명 지정

        else:  # 단일 프로세스로 처리
            edge_list = []  # edge 리스트 초기화
            for g1 in tqdm(gene2go.keys()):  # 모든 유전자에 대해 반복
                for g2 in gene2go.keys():
                    score = len(np.intersect1d(gene2go[g1], gene2go[g2])) / len(np.union1d(gene2go[g1], gene2go[g2]))  # Jaccard 유사도
                    edge_list.append((g1, g2, score))  # edge 추가
            edge_list_filter = [i for i in edge_list if i[2] > minimum]  # 최소 유사도 기준 필터링
            df_edge_list = pd.DataFrame(edge_list_filter).rename(columns={0: 'source', 1: 'target', 2: 'importance'})  # DataFrame 생성

    if method == "expression":  # expression 기반 네트워크 생성
        print('Creating Expression based graph, this can take a few minutes')  # 안내 메시지 출력
        # 구현 예정 (placeholder)
        pass

    if save:  # 저장 옵션이 켜져 있을 경우
        print('Saving edge_list to file')  # 저장 안내 출력
        df_edge_list.to_csv(filename, index=False)  # CSV 파일로 저장

    return df_edge_list  # edge 리스트 반환


def pearson_correlation(x, y):
    """
    두 행렬 x, y의 column-wise Pearson correlation을 계산하는 함수입니다.
    결과는 -1 ~ 1 범위로 클리핑되어 반환됩니다.
    """
    xv = x - x.mean(axis=0)  # x의 각 열 평균을 빼서 중심화
    yv = y - y.mean(axis=0)  # y도 동일하게 중심화
    xvss = (xv * xv).sum(axis=0)  # x의 분산 요소 계산
    yvss = (yv * yv).sum(axis=0)  # y의 분산 요소 계산
    result = np.matmul(xv.transpose(), yv) / np.sqrt(np.outer(xvss, yvss))  # Pearson correlation 공식 계산
    return np.maximum(np.minimum(result, 1.0), -1.0)  # 값 범위를 -1 ~ 1로 제한하여 반환



