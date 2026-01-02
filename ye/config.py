"""
TDA Trading System - Configuration
설정값 중앙 관리
"""
import os

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "ydata")

# 데이터 파일
FILES = {
    'train_long': 'BTCUSDT_perp_15m_training_labeled.csv',
    'train_short': 'BTCUSDT_perp_15m_training_short_labeled.csv',
    'backtest1_long': 'BTCUSDT_perp_15m_backtest1_labeled.csv',
    'backtest1_short': 'BTCUSDT_perp_15m_backtest1_short_labeled.csv',
    'backtest2_long': 'BTCUSDT_perp_15m_backtest2_labeled.csv',
    'backtest2_short': 'BTCUSDT_perp_15m_backtest2_short_labeled.csv',
}

# TDA 파라미터
TDA_WINDOW = 20      # 분석 윈도우 크기 (캔들 수)
TDA_DIM = 3          # Time delay embedding 차원
TDA_DELAY = 1        # Delay 간격

# 트레이딩 파라미터
TP_PCT = 0.02        # Take Profit: 2%
SL_PCT = 0.01        # Stop Loss: 1%
HOLD_CANDLES = 4     # 최대 보유 캔들 수
COOLDOWN = 4         # 진입 후 쿨다운 캔들 수

# 모델 파라미터
N_ESTIMATORS = 200
MAX_DEPTH = 15
MIN_SAMPLES_LEAF = 10
RANDOM_STATE = 42

# 제외할 컬럼 (피처로 사용하지 않음)
EXCLUDE_COLS = [
    'timestamp', 'open', 'high', 'low', 'close', 'volume',
    'buy_volume', 'sell_volume', 'cvd', 'open_interest',
    'trigger', 'event_id', 'open_time', 'volume_delta'
]

# 백테스트 파라미터
INITIAL_CAPITAL = 10000  # 초기 자본
THRESHOLDS = [0.3, 0.4, 0.5]  # 테스트할 임계값
LEVERAGES = [1, 5, 10]  # 테스트할 레버리지
