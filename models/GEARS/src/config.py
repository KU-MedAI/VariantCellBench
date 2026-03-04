# config.py

import os

# 루트 디렉토리 (예: 현재 파일 기준 상위 디렉토리)
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# yennefer dir: /home/jovyan/work/variantseq


###----------Yennefer 환경용----------###
# # 데이터 경로
# # 최상위 데이터경로
# DATA_DIR = os.path.join(BASE_DIR, 'database')
# RAW_DATA_PATH = os.path.join(DATA_DIR, 'rawdata')
# GEARS_DATA_PATH = os.path.join(DATA_DIR, 'gears')
# # PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed_data.csv')

# # 출력 경로
# OUTPUT_DIR = os.path.join(BASE_DIR, 'log')
# ERROR_DIR = os.path.join(BASE_DIR, 'variantGEARS/dump')

# /NFS_DATA/samsung/database/gears




###----------SNA 환경용----------###
# 데이터 경로
# 최상위 데이터경로
BASE_DIR = '/NFS_DATA/samsung'
DATA_DIR = os.path.join(BASE_DIR, 'database')
RAW_DATA_PATH = os.path.join(BASE_DIR, 'rawdata/Variant-seq')
GEARS_DATA_PATH = os.path.join(DATA_DIR, 'gears')
# PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed_data.csv')

# 출력 경로
OUTPUT_DIR = os.path.join(BASE_DIR, 'variantGEARS/log')
ERROR_DIR = os.path.join(BASE_DIR, 'variantGEARS/error')




###----------슈파두파컴퓨타 환경용----------###
# environment = 'KISTI'
# if environment == 'KISTI':
#     BASE_DIR = '/scratch/x3317a07/samsung'
#     DATA_DIR = os.path.join(BASE_DIR, 'database')
#     RAW_DATA_PATH = os.path.join(BASE_DIR, 'rawdata/Variant-seq/kim2023')
#     GEARS_DATA_PATH = os.path.join(DATA_DIR, 'gears')
#     # PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed_data.csv')

#     # 출력 경로
#     OUTPUT_DIR = os.path.join(BASE_DIR, 'variantGEARS/log')
#     ERROR_DIR = os.path.join(BASE_DIR, 'variantGEARS/error')