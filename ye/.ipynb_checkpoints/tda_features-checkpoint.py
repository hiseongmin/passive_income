"""
TDA Trading System - TDA Feature Extraction
Topological Data Analysis 파이프라인
"""
import numpy as np
import pandas as pd
from ripser import ripser
from tqdm import tqdm

from config import TDA_WINDOW, TDA_DIM, TDA_DELAY


def time_delay_embedding(series: np.ndarray, dim: int = 3, delay: int = 1) -> np.ndarray:
    """
    Takens' Theorem을 이용한 Time Delay Embedding
    1D 시계열을 고차원 point cloud로 변환

    Args:
        series: 1D 시계열 데이터
        dim: 임베딩 차원 (기본값: 3)
        delay: 지연 간격 (기본값: 1)

    Returns:
        (n, dim) 형태의 point cloud

    Example:
        [1,2,3,4,5] with dim=3 → [[1,2,3], [2,3,4], [3,4,5]]
    """
    n = len(series) - (dim - 1) * delay
    if n <= 0:
        return np.array([]).reshape(0, dim)

    embedded = np.zeros((n, dim))
    for i in range(dim):
        embedded[:, i] = series[i * delay : i * delay + n]
    return embedded


def compute_persistence(point_cloud: np.ndarray) -> list:
    """
    Ripser를 이용한 Persistence Diagram 계산

    Args:
        point_cloud: (n, dim) 형태의 point cloud

    Returns:
        persistence diagrams 리스트
        - dgms[0]: H0 (connected components)
        - dgms[1]: H1 (loops/holes)
    """
    if len(point_cloud) < 3:
        # 최소 3개 포인트 필요
        return [np.array([[0, 0]]), np.array([[0, 0]])]

    result = ripser(point_cloud, maxdim=1)
    return result['dgms']


def extract_tda_features(price_window: np.ndarray) -> dict:
    """
    가격 윈도우에서 TDA 피처 추출

    Args:
        price_window: 가격 시계열 (예: 20개 캔들)

    Returns:
        TDA 피처 딕셔너리:
        - h0_amplitude: H0 최대 lifetime
        - h0_mean_life: H0 평균 lifetime
        - h0_std_life: H0 lifetime 표준편차
        - h0_num_points: H0 포인트 수
        - h0_entropy: H0 persistence entropy
        - h1_*: H1에 대해 동일한 피처들
    """
    # 정규화
    std = price_window.std()
    if std < 1e-8:
        std = 1e-8
    normalized = (price_window - price_window.mean()) / std

    # Time Delay Embedding
    point_cloud = time_delay_embedding(normalized, dim=TDA_DIM, delay=TDA_DELAY)

    if len(point_cloud) < 3:
        # 빈 피처 반환
        return _empty_tda_features()

    # Persistence Diagram 계산
    dgms = compute_persistence(point_cloud)

    features = {}
    for dim_idx, dgm in enumerate(dgms):
        prefix = f'h{dim_idx}_'

        # Lifetime 계산 (death - birth)
        lifetimes = dgm[:, 1] - dgm[:, 0]
        lifetimes = lifetimes[np.isfinite(lifetimes)]

        if len(lifetimes) > 0:
            features[prefix + 'amplitude'] = lifetimes.max()
            features[prefix + 'mean_life'] = lifetimes.mean()
            features[prefix + 'std_life'] = lifetimes.std() if len(lifetimes) > 1 else 0
            features[prefix + 'num_points'] = len(lifetimes)

            # Persistence Entropy
            total = lifetimes.sum()
            if total > 1e-8:
                probs = lifetimes / total
                entropy = -np.sum(probs * np.log(probs + 1e-8))
            else:
                entropy = 0
            features[prefix + 'entropy'] = entropy
        else:
            features[prefix + 'amplitude'] = 0
            features[prefix + 'mean_life'] = 0
            features[prefix + 'std_life'] = 0
            features[prefix + 'num_points'] = 0
            features[prefix + 'entropy'] = 0

    return features


def _empty_tda_features() -> dict:
    """빈 TDA 피처 반환"""
    features = {}
    for dim_idx in range(2):  # H0, H1
        prefix = f'h{dim_idx}_'
        features[prefix + 'amplitude'] = 0
        features[prefix + 'mean_life'] = 0
        features[prefix + 'std_life'] = 0
        features[prefix + 'num_points'] = 0
        features[prefix + 'entropy'] = 0
    return features


def add_tda_features_to_df(df: pd.DataFrame, window_size: int = None) -> pd.DataFrame:
    """
    DataFrame에 TDA 피처 추가

    Args:
        df: OHLCV 데이터가 포함된 DataFrame
        window_size: TDA 분석 윈도우 크기 (기본값: config.TDA_WINDOW)

    Returns:
        TDA 피처가 추가된 DataFrame
    """
    if window_size is None:
        window_size = TDA_WINDOW

    closes = df['close'].values
    tda_features_list = []

    for i in tqdm(range(len(df)), desc="TDA 피처 추출"):
        if i < window_size:
            # 윈도우가 부족한 경우 빈 피처
            tda_features_list.append(_empty_tda_features())
        else:
            window = closes[i - window_size:i]
            tda_features_list.append(extract_tda_features(window))

    tda_df = pd.DataFrame(tda_features_list)

    # 원본 DataFrame과 병합
    result = pd.concat([df.reset_index(drop=True), tda_df], axis=1)

    return result
