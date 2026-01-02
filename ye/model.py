"""
TDA Trading System - Model Training
모델 학습 및 평가
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from config import N_ESTIMATORS, MAX_DEPTH, MIN_SAMPLES_LEAF, RANDOM_STATE
from data_loader import get_feature_columns, prepare_xy


def create_model() -> RandomForestClassifier:
    """
    RandomForest 모델 생성

    Returns:
        RandomForestClassifier 인스턴스
    """
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight='balanced',  # 불균형 데이터 처리
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


def train_model(df: pd.DataFrame, name: str = 'model') -> tuple:
    """
    모델 학습

    Args:
        df: TDA 피처가 포함된 DataFrame
        name: 모델 이름 (출력용)

    Returns:
        (model, feature_cols) 튜플
    """
    feature_cols = get_feature_columns(df)
    X, y = prepare_xy(df, feature_cols)

    # Train/Val 분할
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # 모델 생성 및 학습
    model = create_model()
    model.fit(X_train, y_train)

    # 평가
    train_acc = model.score(X_train, y_train)
    val_acc = model.score(X_val, y_val)

    print(f"\n=== {name.upper()} 모델 ===")
    print(f"학습 데이터: {len(X_train):,} / 검증 데이터: {len(X_val):,}")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Val Accuracy: {val_acc:.4f}")

    # 피처 중요도
    print_feature_importance(model, feature_cols, top_n=10)

    return model, feature_cols


def predict_proba(model: RandomForestClassifier, df: pd.DataFrame,
                  feature_cols: list) -> np.ndarray:
    """
    예측 확률 반환

    Args:
        model: 학습된 모델
        df: 예측할 DataFrame
        feature_cols: 피처 컬럼 리스트

    Returns:
        trigger=1 확률 배열
    """
    X, _ = prepare_xy(df, feature_cols)
    probs = model.predict_proba(X)[:, 1]
    return probs


def print_feature_importance(model: RandomForestClassifier, feature_cols: list,
                             top_n: int = 10):
    """
    피처 중요도 출력

    Args:
        model: 학습된 모델
        feature_cols: 피처 컬럼 리스트
        top_n: 상위 N개 출력
    """
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop {top_n} 피처:")
    for i, row in importance.head(top_n).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
