import os
import re
from datetime import datetime
from typing import Optional
import pandas as pd
import plotly.graph_objects as go


# --------------------------------------------
# 공통 유틸
# --------------------------------------------
def read_loss_log(csv_path: str) -> pd.DataFrame:
    """
    학습 로그 CSV를 읽어 DataFrame 반환.
    epoch 컬럼이 없으면 1..N 순서로 생성.
    """
    df = pd.read_csv(csv_path)
    if 'epoch' not in df.columns:
        df.insert(0, 'epoch', range(1, len(df) + 1))
    return df

def _make_line_plot(df: pd.DataFrame, columns, title: str, yaxis_title: str, y_range=None) -> go.Figure:
    """
    지정된 컬럼들을 epoch 축에 대해 라인 플롯으로 시각화.
    df에 존재하는 컬럼만 그리며, 하나도 없으면 에러 발생.
    """
    if 'epoch' not in df.columns:
        df = df.copy()
        df.insert(0, 'epoch', range(1, len(df) + 1))

    fig = go.Figure()
    plotted = 0
    for col in columns:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df['epoch'],
                y=df[col],
                mode='lines+markers',
                name=col
            ))
            plotted += 1

    if plotted == 0:
        raise ValueError(f"요청한 컬럼 중 DataFrame에 존재하는 것이 없습니다: {columns}")

    fig.update_layout(
        title=title,
        xaxis_title='epoch',
        yaxis_title=yaxis_title,
        template='plotly_white',
        legend_title='metric',
        hovermode='x unified'
    )
    if y_range is not None:
        fig.update_yaxes(range=y_range)
    return fig

# --------------------------------------------
# 1) training loss
# --------------------------------------------
def plot_training_loss(df: pd.DataFrame) -> go.Figure:
    """
    plot title: training loss
    columns: avg_loss, avg_autofocus_loss, avg_direction_loss
    """
    cols = ["avg_loss", "avg_autofocus_loss", "avg_direction_loss"]
    return _make_line_plot(df, cols, "training loss", "loss")

# --------------------------------------------
# 2) training loss (HVG)
# --------------------------------------------
def plot_training_loss_hvg(df: pd.DataFrame) -> go.Figure:
    """
    plot title: training loss (HVG)
    columns: avg_hvg_loss, avg_hvg_autofocus_loss, avg_hvg_direction_loss
    """
    cols = ["avg_hvg_loss", "avg_hvg_autofocus_loss", "avg_hvg_direction_loss"]
    return _make_line_plot(df, cols, "training loss (HVG)", "loss")

# --------------------------------------------
# 3) MSE
# --------------------------------------------
def plot_mse(df: pd.DataFrame) -> go.Figure:
    """
    plot title: MSE
    columns: train_mse, val_mse
    """
    cols = ["train_mse", "val_mse"]
    return _make_line_plot(df, cols, "MSE", "MSE")

# --------------------------------------------
# 4) MSE DE
# --------------------------------------------
def plot_mse_de(df: pd.DataFrame) -> go.Figure:
    """
    plot title: MSE DE
    columns: train_mse_de, val_mse_de
    """
    cols = ["train_mse_de", "val_mse_de"]
    return _make_line_plot(df, cols, "MSE DE", "MSE")

# --------------------------------------------
# 5) MSE DE non-dropout
# --------------------------------------------
def plot_mse_de_non_dropout(df: pd.DataFrame) -> go.Figure:
    """
    plot title: MSE DE non-dropout
    columns: train_mse_de_non_dropout, val_mse_de_non_dropout
    """
    cols = ["train_mse_de_non_dropout", "val_mse_de_non_dropout"]
    return _make_line_plot(df, cols, "MSE DE non-dropout", "MSE")

# --------------------------------------------
# 6) PCC
# --------------------------------------------
def plot_pcc(df: pd.DataFrame) -> go.Figure:
    """
    plot title: PCC
    columns: train_pearson, val_pearson
    """
    cols = ["train_pearson", "val_pearson"]
    return _make_line_plot(df, cols, "PCC", "Pearson r", y_range=[-1, 1])

# --------------------------------------------
# 7) PCC DE
# --------------------------------------------
def plot_pcc_de(df: pd.DataFrame) -> go.Figure:
    """
    plot title: PCC DE
    columns: train_pearson_de, val_pearson_de
    """
    cols = ["train_pearson_de", "val_pearson_de"]
    return _make_line_plot(df, cols, "PCC DE", "Pearson r", y_range=[-1, 1])

# --------------------------------------------
# 8) PCC DE non-dropout
# --------------------------------------------
def plot_pcc_de_non_dropout(df: pd.DataFrame) -> go.Figure:
    """
    plot title: PCC DE non-dropout
    columns: train_pearson_de_non_dropout, val_pearson_de_non_dropout
    """
    cols = ["train_pearson_de_non_dropout", "val_pearson_de_non_dropout"]
    return _make_line_plot(df, cols, "PCC DE non-dropout", "Pearson r", y_range=[-1, 1])

def _slugify(text: str, allow_unicode: bool = False) -> str:
    """파일명으로 쓰기 안전하게 슬러그화."""
    text = text.strip()
    if not allow_unicode:
        text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'[\\/*?:"<>|]+', '-', text)  # 금지문자 → -
    text = re.sub(r'\s+', '_', text)            # 공백 → _
    text = re.sub(r'[_-]+', '_', text)          # 중복 구분자 정리
    return text.strip('._-') or "plot"

def save_plot_png(
    fig: go.Figure,
    checkpoint_dir: str,
    filename: Optional[str] = None,
    width: int = 1400,
    height: int = 900,
    scale: int = 2,
    overwrite: bool = False,
) -> str:
    """
    Plotly Figure를 PNG로 저장하고 저장된 파일 경로를 반환.

    Args:
        fig: Plotly Figure
        checkpoint_dir: 저장할 디렉토리 경로 (없으면 생성)
        filename: 파일명(확장자 생략 가능). None이면 title 기반 자동 생성
        width, height: PNG 픽셀 크기
        scale: 해상도 배수(2~3 권장)
        overwrite: True면 동일 파일명 덮어쓰기, False면 넘버링

    Returns:
        저장된 파일의 전체 경로 (str)

    Raises:
        RuntimeError: kaleido 미설치 등으로 저장 실패 시
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 기본 파일명 생성 (figure title + 타임스탬프)
    if filename is None:
        title = (fig.layout.title.text if getattr(fig.layout, "title", None) else "") or "plot"
        base = _slugify(title)
    else:
        base = _slugify(os.path.splitext(filename)[0])

    fname = base + ".png"
    out_path = os.path.join(checkpoint_dir, fname)

    # 덮어쓰기 방지 시 넘버링
    if not overwrite:
        i = 1
        while os.path.exists(out_path):
            out_path = os.path.join(checkpoint_dir, f"{base}_{i}.png")
            i += 1

    # 저장
    try:
        fig.write_image(out_path, format="png", width=width, height=height, scale=scale)
    except Exception as e:
        # kaleido가 없으면 보통 여기서 오류가 납니다.
        raise RuntimeError(
            f"PNG 저장에 실패했습니다. kaleido가 설치되어 있는지 확인하세요 "
            f"(pip install -U kaleido). 원인: {e}"
        ) from e

    return out_path

def save_all_standard_plots(df, checkpoint_dir: str):
    makers = [
        ("training_loss", plot_training_loss),
        ("training_loss_hvg", plot_training_loss_hvg),
        ("mse", plot_mse),
        ("mse_de", plot_mse_de),
        # ("mse_de_non_dropout", plot_mse_de_non_dropout),
        ("pcc", plot_pcc),
        ("pcc_de", plot_pcc_de),
        # ("pcc_de_non_dropout", plot_pcc_de_non_dropout),
    ]
    saved = []
    for name, fn in makers:
        fig = fn(df)
        saved.append(save_plot_png(fig, checkpoint_dir, filename=f"{name}.png", overwrite = True))
    return saved

# 예시
# paths = save_all_standard_plots(df, "./checkpoints/run_0421")
# print(paths)

