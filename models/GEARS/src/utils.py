import os
import sys
import logging
import traceback
from config import DATA_DIR, RAW_DATA_PATH, OUTPUT_DIR, ERROR_DIR

def enable_error_logging(log_filename="error.log"):
    """
    logging 모듈을 활용한 에러 로깅 설정.
    KeyboardInterrupt는 별도로 처리.
    """
    log_path = os.path.join(ERROR_DIR, log_filename)

    # 로깅 설정
    logging.basicConfig(
        level=logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)  # 콘솔 출력도 병행
        ]
    )

    def custom_excepthook(exctype, value, tb):
        if exctype == KeyboardInterrupt:
            print("\n[중단됨] 사용자가 실행을 중단했습니다 (KeyboardInterrupt).")
            return
        # 전체 traceback 문자열 생성
        error_msg = ''.join(traceback.format_exception(exctype, value, tb))
        logging.error("Uncaught exception:\n%s", error_msg)

    sys.excepthook = custom_excepthook


def setup_logger(log_filename="error.log"):
    log_path = os.path.join(ERROR_DIR, log_filename)
    os.makedirs(ERROR_DIR, exist_ok=True)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )