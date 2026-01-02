"""
비트코인 TDA(위상 데이터 분석) 기반 트리거 탐지 및 시각화

참고 논문:
1. arXiv 2411.13881: Using TDA to predict stock index movement
2. Springer s00521-024-10787-x: Enhancing financial time series forecasting through TDA

5분봉 데이터로 학습 → 15분봉 차트로 시각화 (5일씩 6행)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 데이터 로드 및 리샘플링
# ============================================================

def load_data(filepath):
    """5분봉 데이터 로드"""
    df = pd.read_csv(filepath)
    df['open_time'] = pd.to_datetime(df['open_time'])
    df.set_index('open_time', inplace=True)
    return df

def resample_to_15m(df):
    """5분봉 → 15분봉 리샘플링"""
    df_15m = df.resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'buy_volume': 'sum',
        'sell_volume': 'sum',
        'volume_delta': 'sum',
        'cvd': 'last'
    }).dropna()
    return df_15m

# ============================================================
# 2. Takens 임베딩 (직접 구현)
# ============================================================

def takens_embedding(time_series, dimension=3, delay=1):
    """
    Takens 시간 지연 임베딩

    Parameters:
    - time_series: 1D 시계열
    - dimension: 임베딩 차원
    - delay: 시간 지연

    Returns:
    - point_cloud: (n_points, dimension) 형태의 포인트 클라우드
    """
    n = len(time_series)
    n_points = n - (dimension - 1) * delay

    if n_points <= 0:
        return None

    point_cloud = np.zeros((n_points, dimension))
    for i in range(dimension):
        point_cloud[:, i] = time_series[i * delay : i * delay + n_points]

    return point_cloud

# ============================================================
# 3. Persistent Homology (간소화 버전)
# ============================================================

def compute_persistence_features(point_cloud):
    """
    포인트 클라우드에서 위상적 특성 추출 (간소화 버전)

    - H0: 연결 성분 (클러스터링 기반)
    - 복잡도: 거리 분포의 엔트로피
    """
    if point_cloud is None or len(point_cloud) < 3:
        return {'entropy': 0, 'complexity': 0, 'spread': 0}

    # 쌍별 거리 계산
    distances = pdist(point_cloud)

    if len(distances) == 0:
        return {'entropy': 0, 'complexity': 0, 'spread': 0}

    # 거리 분포 기반 특성
    dist_normalized = distances / (distances.max() + 1e-8)

    # 히스토그램 기반 엔트로피
    hist, _ = np.histogram(dist_normalized, bins=20, density=True)
    hist = hist + 1e-10  # 0 방지
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log(hist))

    # 복잡도: 거리의 표준편차
    complexity = np.std(distances)

    # 확산도: 최대-최소 거리 비율
    spread = distances.max() / (distances.min() + 1e-8)

    return {
        'entropy': entropy,
        'complexity': complexity,
        'spread': np.log1p(spread)
    }

# ============================================================
# 4. TDA 트리거 탐지
# ============================================================

def compute_tda_triggers(df_5m, window_size=60, embedding_dim=3, delay=1, use_oi=False):
    """
    5분봉 데이터로 TDA 특성 계산 및 트리거 탐지

    Parameters:
    - df_5m: 5분봉 데이터프레임
    - window_size: 슬라이딩 윈도우 크기 (60 = 5시간)
    - embedding_dim: Takens 임베딩 차원
    - delay: 시간 지연
    - use_oi: OI(Open Interest) 포함 여부

    Returns:
    - tda_df: TDA 특성 데이터프레임
    """
    prices = df_5m['close'].values
    n = len(prices)

    # OI 데이터 준비 (사용시)
    if use_oi and 'open_interest' in df_5m.columns:
        oi = df_5m['open_interest'].fillna(method='ffill').fillna(0).values
        print("  [OI 포함 모드]")
    else:
        oi = None
        print("  [Close만 사용 모드]")

    results = {
        'entropy': [],
        'complexity': [],
        'spread': [],
        'price_volatility': []
    }
    valid_indices = []

    print(f"TDA 특성 계산 중... (총 {n - window_size + 1}개 윈도우)")

    for i in range(window_size, n + 1):
        window = prices[i - window_size:i]

        # 정규화
        window_norm = (window - window.mean()) / (window.std() + 1e-8)

        # OI 포함시 2D 시계열로 확장
        if oi is not None:
            oi_window = oi[i - window_size:i]
            oi_norm = (oi_window - oi_window.mean()) / (oi_window.std() + 1e-8)
            # 2D 포인트 클라우드 생성 (price, oi)
            combined = np.column_stack([window_norm, oi_norm])
            point_cloud = combined  # 직접 2D 포인트 클라우드로 사용
        else:
            # Takens 임베딩 (기존 방식)
            point_cloud = takens_embedding(window_norm, dimension=embedding_dim, delay=delay)

        # 위상적 특성 추출
        features = compute_persistence_features(point_cloud)

        results['entropy'].append(features['entropy'])
        results['complexity'].append(features['complexity'])
        results['spread'].append(features['spread'])
        results['price_volatility'].append(np.std(window))

        valid_indices.append(i - 1)

        if (i - window_size) % 500 == 0:
            print(f"  진행: {i - window_size + 1}/{n - window_size + 1}")

    # 데이터프레임 생성
    tda_df = pd.DataFrame(results, index=df_5m.index[valid_indices])

    print(f"TDA 특성 계산 완료: {len(tda_df)}개 시점")

    return tda_df

def detect_triggers(tda_df, df_5m, threshold_percentile=85):
    """
    TDA 특성 변화 기반 트리거 지점 탐지
    """
    # 변화량 계산
    tda_df = tda_df.copy()
    tda_df['entropy_change'] = tda_df['entropy'].diff().abs()
    tda_df['complexity_change'] = tda_df['complexity'].diff().abs()
    tda_df['spread_change'] = tda_df['spread'].diff().abs()

    # Z-score 정규화
    for col in ['entropy_change', 'complexity_change', 'spread_change']:
        values = tda_df[col].dropna()
        if len(values) > 0:
            tda_df[col + '_z'] = (tda_df[col] - values.mean()) / (values.std() + 1e-8)

    # 복합 트리거 점수
    tda_df['trigger_score'] = (
        tda_df['entropy_change_z'].fillna(0).abs() * 0.4 +
        tda_df['complexity_change_z'].fillna(0).abs() * 0.3 +
        tda_df['spread_change_z'].fillna(0).abs() * 0.3
    )

    # 임계값 기반 트리거 탐지
    valid_scores = tda_df['trigger_score'].dropna()
    if len(valid_scores) == 0:
        return pd.DataFrame(), tda_df

    threshold = np.percentile(valid_scores, threshold_percentile)
    trigger_mask = tda_df['trigger_score'] > threshold

    # 최소 간격 적용 (연속 트리거 방지, 최소 6개 봉 = 30분)
    trigger_indices = tda_df[trigger_mask].index.tolist()
    filtered_triggers = []
    last_trigger = None

    for idx in trigger_indices:
        if last_trigger is None:
            filtered_triggers.append(idx)
            last_trigger = idx
        elif (idx - last_trigger).total_seconds() >= 1800:  # 30분
            filtered_triggers.append(idx)
            last_trigger = idx

    # 트리거 정보 수집
    triggers = []
    for idx in filtered_triggers:
        try:
            loc = df_5m.index.get_loc(idx)

            # 과거 가격 모멘텀으로 방향 판단 (12개 봉 = 1시간 전)
            # 실시간 데이터처럼 해당 시점까지의 데이터만 사용
            if loc >= 12:
                past_return = (df_5m.iloc[loc]['close'] - df_5m.iloc[loc - 12]['close']) / df_5m.iloc[loc - 12]['close']
                direction = 'long' if past_return > 0.001 else ('short' if past_return < -0.001 else 'neutral')
            else:
                direction = 'neutral'

            triggers.append({
                'time': idx,
                'price': df_5m.loc[idx, 'close'],
                'direction': direction,
                'score': tda_df.loc[idx, 'trigger_score']
            })
        except:
            continue

    print(f"트리거 탐지 완료: {len(triggers)}개")

    return pd.DataFrame(triggers), tda_df

# ============================================================
# 5. 시각화 (5일씩 6행)
# ============================================================

def plot_15m_chart_6rows(df_15m, triggers_df, tda_df, save_path=None, n_rows=10):
    """
    15분봉 차트를 N행으로 시각화
    """
    # 날짜별로 그룹화
    df_15m = df_15m.copy()
    df_15m['date'] = df_15m.index.date

    unique_dates = df_15m['date'].unique()
    total_days = len(unique_dates)

    # N개 구간으로 분할
    days_per_row = max(1, total_days // n_rows)

    fig, axes = plt.subplots(n_rows, 1, figsize=(24, 4 * n_rows))
    fig.suptitle(f'BTCUSDT 15m Chart with TDA Triggers ({days_per_row} Days per Row)',
                 fontsize=16, fontweight='bold', y=0.995)

    for row in range(n_rows):
        ax = axes[row]

        # 해당 행의 날짜 범위
        start_day_idx = row * days_per_row
        end_day_idx = min((row + 1) * days_per_row, total_days)

        if start_day_idx >= total_days:
            ax.set_visible(False)
            continue

        row_dates = unique_dates[start_day_idx:end_day_idx]
        row_data = df_15m[df_15m['date'].isin(row_dates)]

        if len(row_data) == 0:
            ax.set_visible(False)
            continue

        # 캔들스틱 그리기
        candle_width = 0.006  # 15분봉 너비

        for idx, row_candle in row_data.iterrows():
            color = '#26a69a' if row_candle['close'] >= row_candle['open'] else '#ef5350'

            # 몸통
            bottom = min(row_candle['open'], row_candle['close'])
            height = abs(row_candle['close'] - row_candle['open'])
            if height < 0.1:
                height = 0.1

            rect = Rectangle(
                (mdates.date2num(idx) - candle_width/2, bottom),
                candle_width, height,
                facecolor=color, edgecolor=color, alpha=0.9
            )
            ax.add_patch(rect)

            # 심지
            ax.plot([mdates.date2num(idx), mdates.date2num(idx)],
                   [row_candle['low'], row_candle['high']],
                   color=color, linewidth=0.8)

        # 트리거 마커 표시
        row_triggers = triggers_df[
            (triggers_df['time'] >= row_data.index.min()) &
            (triggers_df['time'] <= row_data.index.max())
        ]

        for _, trigger in row_triggers.iterrows():
            if trigger['direction'] == 'long':
                marker_color = '#00ff00'
                marker = '^'
                y_offset = row_data['low'].min() - (row_data['high'].max() - row_data['low'].min()) * 0.02
                text_offset = -15
            elif trigger['direction'] == 'short':
                marker_color = '#ff00ff'
                marker = 'v'
                y_offset = row_data['high'].max() + (row_data['high'].max() - row_data['low'].min()) * 0.02
                text_offset = 15
            else:
                marker_color = '#ffff00'
                marker = 'o'
                y_offset = trigger['price']
                text_offset = 10

            ax.scatter(trigger['time'], trigger['price'],
                      c=marker_color, marker=marker, s=120,
                      edgecolors='white', linewidths=1.5, zorder=5)

            # 세로선 표시
            ax.axvline(trigger['time'], color=marker_color, alpha=0.3,
                      linestyle='--', linewidth=1)

        # 축 설정
        ax.set_xlim(row_data.index.min(), row_data.index.max())
        price_range = row_data['high'].max() - row_data['low'].min()
        ax.set_ylim(row_data['low'].min() - price_range * 0.05,
                   row_data['high'].max() + price_range * 0.05)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))

        # 날짜 범위 표시
        date_range = f"{row_dates[0]} ~ {row_dates[-1]}"
        ax.set_ylabel(f'Price\n{date_range}', fontsize=10)
        ax.grid(True, alpha=0.3)

        # 트리거 개수 표시
        n_long = len(row_triggers[row_triggers['direction'] == 'long'])
        n_short = len(row_triggers[row_triggers['direction'] == 'short'])
        ax.text(0.02, 0.95, f'Long: {n_long} | Short: {n_short}',
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
               color='white')

    # 범례
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#00ff00',
               markersize=12, label='Long Trigger'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='#ff00ff',
               markersize=12, label='Short Trigger'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ffff00',
               markersize=10, label='Neutral')
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(top=0.97)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"차트 저장 완료: {save_path}")

    plt.show()

    return fig

# ============================================================
# 메인 실행
# ============================================================

if __name__ == "__main__":
    # 1. 데이터 로드
    filepath = "/notebooks/ye/anal/BTCUSDT_perp_5m_30d_full.csv"
    print("=" * 60)
    print("1. 데이터 로드")
    print("=" * 60)

    df_5m = load_data(filepath)
    print(f"5분봉 데이터: {len(df_5m)}개")
    print(f"기간: {df_5m.index.min()} ~ {df_5m.index.max()}")
    print(f"컬럼: {df_5m.columns.tolist()}")

    # 15분봉 리샘플링 (공통)
    df_15m = resample_to_15m(df_5m)
    print(f"15분봉 데이터: {len(df_15m)}개")

    # ============================================================
    # [A] OI 없는 버전 (Close만 사용)
    # ============================================================
    print("\n" + "=" * 60)
    print("[A] TDA 분석 - Close만 사용 (OI 제외)")
    print("=" * 60)

    tda_df_no_oi = compute_tda_triggers(
        df_5m,
        window_size=60,
        embedding_dim=3,
        delay=1,
        use_oi=False
    )

    triggers_no_oi, tda_df_no_oi = detect_triggers(tda_df_no_oi, df_5m, threshold_percentile=85)

    print("\n[A] 트리거 요약 (OI 제외):")
    print(f"  - Long: {len(triggers_no_oi[triggers_no_oi['direction'] == 'long'])}개")
    print(f"  - Short: {len(triggers_no_oi[triggers_no_oi['direction'] == 'short'])}개")
    print(f"  - Neutral: {len(triggers_no_oi[triggers_no_oi['direction'] == 'neutral'])}개")
    print(f"  - 총: {len(triggers_no_oi)}개")

    # ============================================================
    # [B] OI 포함 버전
    # ============================================================
    print("\n" + "=" * 60)
    print("[B] TDA 분석 - Close + OI 사용")
    print("=" * 60)

    tda_df_with_oi = compute_tda_triggers(
        df_5m,
        window_size=60,
        embedding_dim=3,
        delay=1,
        use_oi=True
    )

    triggers_with_oi, tda_df_with_oi = detect_triggers(tda_df_with_oi, df_5m, threshold_percentile=85)

    print("\n[B] 트리거 요약 (OI 포함):")
    print(f"  - Long: {len(triggers_with_oi[triggers_with_oi['direction'] == 'long'])}개")
    print(f"  - Short: {len(triggers_with_oi[triggers_with_oi['direction'] == 'short'])}개")
    print(f"  - Neutral: {len(triggers_with_oi[triggers_with_oi['direction'] == 'neutral'])}개")
    print(f"  - 총: {len(triggers_with_oi)}개")

    # ============================================================
    # 비교 분석
    # ============================================================
    print("\n" + "=" * 60)
    print("비교 분석")
    print("=" * 60)
    print(f"OI 제외 트리거: {len(triggers_no_oi)}개")
    print(f"OI 포함 트리거: {len(triggers_with_oi)}개")
    print(f"차이: {abs(len(triggers_with_oi) - len(triggers_no_oi))}개")

    # 겹치는 트리거 확인
    if len(triggers_no_oi) > 0 and len(triggers_with_oi) > 0:
        overlap = 0
        for _, t1 in triggers_no_oi.iterrows():
            for _, t2 in triggers_with_oi.iterrows():
                if abs((t1['time'] - t2['time']).total_seconds()) < 1800:  # 30분 이내
                    overlap += 1
                    break
        print(f"겹치는 트리거 (30분 이내): {overlap}개")

    # ============================================================
    # 시각화 (20행, OI 제외 버전)
    # ============================================================
    print("\n" + "=" * 60)
    print("시각화 - OI 제외 버전 (20행)")
    print("=" * 60)

    fig = plot_15m_chart_6rows(
        df_15m,
        triggers_no_oi,
        tda_df_no_oi,
        save_path="/notebooks/ye/anal/btc_tda_NO_OI.png",
        n_rows=20
    )

    # ============================================================
    # 시각화 (20행, OI 포함 버전)
    # ============================================================
    print("\n" + "=" * 60)
    print("시각화 - OI 포함 버전 (20행)")
    print("=" * 60)

    fig2 = plot_15m_chart_6rows(
        df_15m,
        triggers_with_oi,
        tda_df_with_oi,
        save_path="/notebooks/ye/anal/btc_tda_WITH_OI.png",
        n_rows=20
    )

    print("\n" + "=" * 60)
    print("분석 완료!")
    print("=" * 60)
    print(f"\n저장된 파일:")
    print(f"  - OI 제외: btc_tda_NO_OI.png")
    print(f"  - OI 포함: btc_tda_WITH_OI.png")
