"""
TDA Trading System - Data Loader
데이터 로딩 및 전처리
"""
import os
import pandas as pd
import numpy as np

from config import DATA_DIR, FILES, EXCLUDE_COLS


def load_data(file_key: str) -> pd.DataFrame:
    """
    데이터 파일 로딩

    Args:
        file_key: FILES 딕셔너리 키
            - 'train_long', 'train_short'
            - 'backtest1_long', 'backtest1_short'
            - 'backtest2_long', 'backtest2_short'

    Returns:
        로딩된 DataFrame
    """
    if file_key not in FILES:
        raise ValueError(f"Unknown file key: {file_key}. Available: {list(FILES.keys())}")

    file_path = os.path.join(DATA_DIR, FILES[file_key])

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)
    print(f"[{file_key}] 로드 완료: {len(df):,} rows")

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    피처 컬럼 추출 (EXCLUDE_COLS 제외)

    Args:
        df: DataFrame

    Returns:
        피처 컬럼명 리스트
    """
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    return feature_cols


def prepare_xy(df: pd.DataFrame, feature_cols: list) -> tuple:
    """
    학습용 X, y 데이터 준비

    Args:
        df: DataFrame
        feature_cols: 피처 컬럼명 리스트

    Returns:
        (X, y) 튜플
        - X: 피처 배열 (inf/nan 처리됨)
        - y: 타겟 배열 (trigger)
    """
    X = df[feature_cols].values
    y = df['trigger'].values

    # inf, nan 처리
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    return X, y


def load_all_data() -> dict:
    """
    모든 데이터 파일 로딩

    Returns:
        딕셔너리 형태로 모든 데이터 반환
    """
    data = {}
    for key in FILES.keys():
        data[key] = load_data(key)
    return data
