프롬프트

1.데이터셋 준비와 첫 모델 작성
~/.claude/plans/ticklish-mapping-kettle.md

# TDA 기반 선물 거래 트리거 예측 시스템

## Opus용 프롬프트

```
당신은 암호화폐 선물 거래를 위한 TDA(Topological Data Analysis) 기반 트리거 예측 시스템을 구현해야 합니다.

## 프로젝트 개요

BTCUSDT 선물 데이터에서 향후 2시간 내 2% 가격 변동이 발생하는 시점을 15분 전에 미리 예측하는 딥러닝 모델을 구축합니다.

## 데이터 정보

### 파일 위치
- 5분봉: /notebooks/data/BTCUSDT_perp_etf_to_90d_ago.csv (181,441 rows)
- 1시간봉: /notebooks/data/BTCUSDT_perp_1h_etf_to_90d_ago.csv (15,121 rows)

### 컬럼 구조
open_time, open, high, low, close, volume, buy_volume, sell_volume, volume_delta, cvd, open_interest

## Part 1: 트레이닝 데이터 준비

### TRIGGER 컬럼 생성 로직
1. 각 5분봉 캔들에서 향후 24개 캔들(2시간)을 스캔
2. close 가격 기준 처음으로 ±2% 변동이 발생하는 지점 찾기
3. 해당 지점부터 15분(3캔들) 전까지 TRIGGER=1 마킹
4. DIRECTION 컬럼 추가: 0=UP (+2%), 1=DOWN (-2%), 2=NONE
5. **IMMINENCE 컬럼 추가**: 트리거까지 남은 시간의 역수 (0~1, 1=바로 직전)

```python
# TRIGGER + IMMINENCE 생성 알고리즘
def generate_trigger_labels(df, threshold=0.02, look_forward=24, pre_trigger=3):
    """
    Args:
        df: 5분봉 데이터프레임
        threshold: 변동 기준 (0.02 = 2%)
        look_forward: 탐색할 미래 캔들 수 (24 = 2시간)
        pre_trigger: 트리거 전 마킹할 캔들 수 (3 = 15분)

    Returns:
        TRIGGER: 0 or 1
        IMMINENCE: 0~1 (트리거까지 남은 거리의 역수)
        DIRECTION: 0=UP, 1=DOWN, 2=NONE
    """
    n = len(df)
    trigger = np.zeros(n)
    imminence = np.zeros(n)
    direction = np.full(n, 2)  # 기본값 NONE

    for i in range(n):
        current_price = df.iloc[i]['close']

        for j in range(i+1, min(i+1+look_forward, n)):
            future_price = df.iloc[j]['close']
            change = (future_price - current_price) / current_price

            if abs(change) >= threshold:
                trigger_point = j
                dir_val = 0 if change > 0 else 1  # 0=UP, 1=DOWN

                # 15분 전부터 트리거 시점까지 마킹
                for k in range(max(0, trigger_point - pre_trigger), trigger_point + 1):
                    trigger[k] = 1
                    direction[k] = dir_val
                    # IMMINENCE: 트리거까지 남은 캔들 수의 역수 (0~1)
                    # k가 trigger_point면 1.0, pre_trigger 캔들 전이면 0.25
                    candles_to_trigger = trigger_point - k
                    imminence[k] = 1.0 - (candles_to_trigger / (pre_trigger + 1))
                break

    df['TRIGGER'] = trigger.astype(int)
    df['IMMINENCE'] = imminence
    df['DIRECTION'] = direction.astype(int)
    return df
```

### 데이터 저장 경로 (원본 보존)
- **원본**: `/notebooks/data/` (수정하지 않음)
- **가공 데이터**: `/notebooks/sa/data/` (새로 생성)

```python
# 데이터 처리 및 저장
import os

# 원본 로드
df_5m = pd.read_csv('/notebooks/data/BTCUSDT_perp_etf_to_90d_ago.csv')
df_1h = pd.read_csv('/notebooks/data/BTCUSDT_perp_1h_etf_to_90d_ago.csv')

# TRIGGER/IMMINENCE/DIRECTION 생성
df_5m = generate_trigger_labels(df_5m)

# 저장 경로 생성
os.makedirs('/notebooks/sa/data', exist_ok=True)

# 가공된 데이터 저장
df_5m.to_csv('/notebooks/sa/data/BTCUSDT_perp_5m_labeled.csv', index=False)
df_1h.to_csv('/notebooks/sa/data/BTCUSDT_perp_1h.csv', index=False)
```

### 멀티 타임프레임 정렬
- 5분봉을 primary로 사용
- 각 5분봉에 해당하는 1시간봉 데이터 매핑 (floor to hour)

## Part 2: 모델 아키텍처 (LSTM + N-BEATS + TDA)

### 입력 구성
- **6시간 윈도우**: 5분봉 72개 + 1시간봉 6개
- **TDA 특성**: Takens embedding → Persistence diagram → entropy, amplitude, num_points
- **시장 미세구조**: volume_delta, cvd, open_interest 등

### TDA 특성 추출 (기존 코드 참조)
파일: /notebooks/.Trash-0/files/binance-data-collector/ml/features/tda_features.py

```python
# Takens time delay embedding
def time_delay_embedding(x, dim=3, tau=1):
    n = len(x) - (dim - 1) * tau
    embedded = np.zeros((n, dim))
    for i in range(dim):
        embedded[:, i] = x[i * tau: i * tau + n]
    return embedded

# Persistence diagram → Features
def extract_tda_features(window, embedding_dim=3, embedding_tau=1):
    point_cloud = time_delay_embedding(window, dim=embedding_dim, tau=embedding_tau)
    diagrams = ripser(point_cloud, maxdim=1)['dgms']
    h1_diagram = diagrams[1]

    entropy = persistence_entropy(h1_diagram)
    amplitude = persistence_amplitude(h1_diagram)
    num_points = persistence_num_points(h1_diagram)

    return entropy, amplitude, num_points
```

### 모델 클래스 구현

```python
class TriggerPredictionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # LSTM Encoders
        self.lstm_5m = nn.LSTM(input_size=5, hidden_size=128, num_layers=2,
                               batch_first=True, bidirectional=True, dropout=0.2)
        self.lstm_1h = nn.LSTM(input_size=5, hidden_size=64, num_layers=2,
                               batch_first=True, bidirectional=True, dropout=0.2)

        # TDA & Microstructure Encoders
        self.tda_encoder = nn.Sequential(
            nn.Linear(9, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(64, 64), nn.GELU()
        )
        self.micro_encoder = nn.Sequential(
            nn.Linear(12, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(64, 64), nn.GELU()
        )

        # Temporal Fusion (256 + 128 + 64 + 64 = 512 → 256)
        self.temporal_fusion = nn.Sequential(
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2)
        )

        # N-BEATS Blocks (3개)
        self.nbeats_blocks = nn.ModuleList([
            NBeatsBlock(256, 256, theta_size=32) for _ in range(3)
        ])

        # Output Heads (수정됨: 확률 + 임박도)
        # 1. Trigger 확률 (Sigmoid → 0~1)
        self.trigger_head = nn.Sequential(
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 1), nn.Sigmoid()
        )

        # 2. Imminence Score (임박도: 0=먼 미래, 1=바로 직전)
        self.imminence_head = nn.Sequential(
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 1), nn.Sigmoid()
        )

        # 3. Direction (UP/DOWN/NONE)
        self.direction_head = nn.Sequential(
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 3)
        )

    def forward(self, x_5m, x_1h, tda_features, micro_features):
        # LSTM encoding
        _, (h_5m, _) = self.lstm_5m(x_5m)  # (batch, 256)
        _, (h_1h, _) = self.lstm_1h(x_1h)  # (batch, 128)

        # Static feature encoding
        tda_enc = self.tda_encoder(tda_features)    # (batch, 64)
        micro_enc = self.micro_encoder(micro_features)  # (batch, 64)

        # Fusion
        fused = self.temporal_fusion(torch.cat([h_5m, h_1h, tda_enc, micro_enc], dim=-1))

        # N-BEATS processing
        residual = fused
        for block in self.nbeats_blocks:
            backcast, _ = block(residual)
            residual = residual - backcast

        final_repr = fused + residual

        # Output (확률값으로 변경)
        trigger_prob = self.trigger_head(final_repr)      # (batch, 1) 0~1 확률
        imminence = self.imminence_head(final_repr)       # (batch, 1) 0~1 임박도
        direction_logits = self.direction_head(final_repr)  # (batch, 3)

        return trigger_prob, imminence, direction_logits
```

### 출력 해석

```python
# 추론 시
trigger_prob = model_output[0]    # 0.85 = 85% 확률로 트리거 발생
imminence = model_output[1]       # 0.7 = 15분 중 약 10.5분 후 발생 예상
direction = softmax(model_output[2])  # [0.8, 0.15, 0.05] = UP 80%

# 트레이딩 의사결정 예시
if trigger_prob > 0.7 and imminence > 0.5:
    position_size = base_size * trigger_prob  # 확률에 비례한 사이징
    if direction == 'UP':
        open_long(size=position_size)
    else:
        open_short(size=position_size)
```

### 손실 함수 (BCE + MSE + CE)

```python
class TriggerLoss(nn.Module):
    def __init__(self, trigger_weight=1.0, imminence_weight=0.5, direction_weight=0.3):
        super().__init__()
        self.trigger_weight = trigger_weight
        self.imminence_weight = imminence_weight
        self.direction_weight = direction_weight

    def forward(self, trigger_prob, imminence, direction_logits,
                trigger_target, imminence_target, direction_target):
        """
        Args:
            trigger_prob: (batch, 1) - 예측 확률 0~1
            imminence: (batch, 1) - 예측 임박도 0~1
            direction_logits: (batch, 3) - 방향 로짓
            trigger_target: (batch, 1) - 실제 트리거 0 or 1
            imminence_target: (batch, 1) - 실제 임박도 0~1
            direction_target: (batch,) - 실제 방향 0/1/2
        """
        # 1. Trigger Loss (Binary Cross Entropy)
        trigger_loss = F.binary_cross_entropy(
            trigger_prob, trigger_target.float(), reduction='mean'
        )

        # 2. Imminence Loss (MSE, only for TRIGGER=True samples)
        mask = trigger_target.squeeze() == 1
        if mask.sum() > 0:
            imminence_loss = F.mse_loss(
                imminence[mask], imminence_target[mask], reduction='mean'
            )
        else:
            imminence_loss = torch.tensor(0.0, device=trigger_prob.device)

        # 3. Direction Loss (Cross Entropy, only for TRIGGER=True samples)
        if mask.sum() > 0:
            direction_loss = F.cross_entropy(
                direction_logits[mask], direction_target[mask], reduction='mean'
            )
        else:
            direction_loss = torch.tensor(0.0, device=trigger_prob.device)

        total_loss = (
            self.trigger_weight * trigger_loss +
            self.imminence_weight * imminence_loss +
            self.direction_weight * direction_loss
        )

        return total_loss, {
            'trigger_loss': trigger_loss.item(),
            'imminence_loss': imminence_loss.item(),
            'direction_loss': direction_loss.item()
        }
```

## 참조할 기존 코드

1. TDA 특성: `/notebooks/.Trash-0/files/binance-data-collector/ml/features/tda_features.py`
2. N-BEATS: `/notebooks/.Trash-0/files/binance-data-collector/ml/models/nbeats.py`
3. 시장 미세구조: `/notebooks/.Trash-0/files/binance-data-collector/ml/features/market_microstructure.py`
4. 특성 결합: `/notebooks/.Trash-0/files/binance-data-collector/ml/features/feature_combiner.py`

## 요구사항

1. `/notebooks/sa/` 폴더에 모든 코드 작성
2. **원본 데이터 보존**: `/notebooks/data/` 수정 금지, `/notebooks/sa/data/`에 가공 데이터 저장
3. 클래스 불균형 처리 (TRIGGER=1은 약 15-25%)
4. Train/Val/Test = 70/15/15 분할
5. 평가지표:
   - TRIGGER: Precision, Recall, F1-Score, AUC-ROC
   - IMMINENCE: MAE, MSE (TRIGGER=1인 샘플에서만)
   - DIRECTION: Accuracy (TRIGGER=1인 샘플에서만)
6. 체크포인트 저장 및 로깅 구현

## 의존성

torch>=2.0.0, numpy, pandas, scikit-learn, ripser, persim, matplotlib, tqdm, tensorboard
```

---

## 모델 플로우 다이어그램

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              INPUT LAYER (6시간 윈도우)                              │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌───────────┐  ┌───────────────┐│
│  │   5분봉 OHLCV       │  │   1시간봉 OHLCV     │  │TDA 특성   │  │시장 미세구조  ││
│  │   (72 x 5)          │  │   (6 x 5)           │  │  (9)      │  │    (12)       ││
│  │                     │  │                     │  │           │  │               ││
│  │ • Open              │  │ • Open              │  │• entropy  │  │• volume_delta ││
│  │ • High              │  │ • High              │  │• amplitude│  │• cvd          ││
│  │ • Low               │  │ • Low               │  │• num_pts  │  │• open_interest││
│  │ • Close             │  │ • Close             │  │  (x3      │  │• buy_sell_imb ││
│  │ • Volume            │  │ • Volume            │  │  configs) │  │• atr_pct      ││
│  └──────────┬──────────┘  └──────────┬──────────┘  └─────┬─────┘  └───────┬───────┘│
│             │                        │                   │                │        │
└─────────────┼────────────────────────┼───────────────────┼────────────────┼────────┘
              │                        │                   │                │
              ▼                        ▼                   ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              ENCODER LAYER                                          │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌───────────┐  ┌───────────────┐│
│  │   Bi-LSTM (5m)      │  │   Bi-LSTM (1h)      │  │TDA MLP    │  │Micro MLP      ││
│  │   2 layers          │  │   2 layers          │  │64 units   │  │64 units       ││
│  │   hidden=128        │  │   hidden=64         │  │           │  │               ││
│  │   dropout=0.2       │  │   dropout=0.2       │  │LayerNorm  │  │LayerNorm      ││
│  │                     │  │                     │  │GELU       │  │GELU           ││
│  └──────────┬──────────┘  └──────────┬──────────┘  └─────┬─────┘  └───────┬───────┘│
│             │                        │                   │                │        │
│             ▼                        ▼                   ▼                ▼        │
│        (batch, 256)            (batch, 128)         (batch, 64)     (batch, 64)    │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     │
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           TEMPORAL FUSION                                           │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│                    ┌─────────────────────────────────────────┐                     │
│                    │         Concatenation                   │                     │
│                    │   256 + 128 + 64 + 64 = 512 dims        │                     │
│                    └──────────────────┬──────────────────────┘                     │
│                                       ▼                                             │
│                    ┌─────────────────────────────────────────┐                     │
│                    │         Dense Layer (512 → 256)         │                     │
│                    │         LayerNorm + GELU + Dropout      │                     │
│                    └──────────────────┬──────────────────────┘                     │
│                                       │                                             │
│                                  (batch, 256)                                       │
│                                       │                                             │
└───────────────────────────────────────┼─────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           N-BEATS BLOCKS (x3)                                       │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│    ┌────────────────┐    ┌────────────────┐    ┌────────────────┐                  │
│    │  N-BEATS #1    │    │  N-BEATS #2    │    │  N-BEATS #3    │                  │
│    │                │    │                │    │                │                  │
│    │  4-layer FC    │───▶│  4-layer FC    │───▶│  4-layer FC    │                  │
│    │  hidden=256    │    │  hidden=256    │    │  hidden=256    │                  │
│    │  theta=32      │    │  theta=32      │    │  theta=32      │                  │
│    │                │    │                │    │                │                  │
│    │  ┌──────────┐  │    │  ┌──────────┐  │    │  ┌──────────┐  │                  │
│    │  │Backcast  │  │    │  │Backcast  │  │    │  │Backcast  │  │                  │
│    │  │(residual)│  │    │  │(residual)│  │    │  │(residual)│  │                  │
│    │  └──────────┘  │    │  └──────────┘  │    │  └──────────┘  │                  │
│    └────────────────┘    └────────────────┘    └────────────────┘                  │
│                                                                                     │
│    residual = input - backcast (각 블록에서 차감)                                    │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              OUTPUT HEADS                                           │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌───────────────────────┐  ┌───────────────────────┐  ┌───────────────────────┐   │
│  │    TRIGGER HEAD       │  │   IMMINENCE HEAD      │  │   DIRECTION HEAD      │   │
│  │                       │  │                       │  │                       │   │
│  │  Dense(256 → 128)     │  │  Dense(256 → 64)      │  │  Dense(256 → 64)      │   │
│  │  GELU + Dropout       │  │  GELU + Dropout       │  │  GELU + Dropout       │   │
│  │  Dense(128 → 1)       │  │  Dense(64 → 1)        │  │  Dense(64 → 3)        │   │
│  │  Sigmoid              │  │  Sigmoid              │  │  Softmax              │   │
│  └───────────┬───────────┘  └───────────┬───────────┘  └───────────┬───────────┘   │
│              │                          │                          │               │
│              ▼                          ▼                          ▼               │
│         (batch, 1)                 (batch, 1)                 (batch, 3)           │
│          0 ~ 1                      0 ~ 1                    [UP,DOWN,NONE]        │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                    │                          │                          │
                    ▼                          ▼                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              OUTPUT LAYER                                           │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌───────────────────────┐  ┌───────────────────────┐  ┌───────────────────────┐   │
│  │   TRIGGER 확률        │  │   IMMINENCE 점수      │  │   DIRECTION 확률      │   │
│  │   (batch, 1)          │  │   (batch, 1)          │  │   (batch, 3)          │   │
│  │                       │  │                       │  │                       │   │
│  │   0.85 = 85% 확률     │  │   0.75 = 곧 발생      │  │   [0.8, 0.15, 0.05]   │   │
│  │   트리거 발생 예상    │  │   (15분 중 3.75분 후) │  │   UP=80%              │   │
│  └───────────────────────┘  └───────────────────────┘  └───────────────────────┘   │
│                                                                                     │
│    최종 활용:                                                                       │
│    • confidence = trigger_prob × imminence  (종합 신뢰도)                          │
│    • if confidence > 0.6: enter_position(direction, size=confidence × base)        │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 입출력 상세 명세

### INPUT (입력)

| 입력 이름 | Shape | 설명 |
|-----------|-------|------|
| `x_5m` | (batch, 72, 5) | 6시간 5분봉 OHLCV 시퀀스 |
| `x_1h` | (batch, 6, 5) | 6시간 1시간봉 OHLCV 시퀀스 |
| `tda_features` | (batch, 9) | TDA 특성 (3 config x 3 features) |
| `micro_features` | (batch, 12) | 시장 미세구조 특성 |

### TDA Features 상세

| 인덱스 | 특성 | 설명 |
|--------|------|------|
| 0-2 | Config 1 (dim=3, tau=1) | entropy, amplitude, num_points |
| 3-5 | Config 2 (dim=5, tau=2) | entropy, amplitude, num_points |
| 6-8 | Config 3 (dim=7, tau=3) | entropy, amplitude, num_points |

### Microstructure Features 상세

| 인덱스 | 특성 | 설명 |
|--------|------|------|
| 0 | atr_pct | ATR 백분율 (변동성) |
| 1 | volatility_std | 가격 표준편차 |
| 2 | relative_volume | 상대 거래량 |
| 3 | buy_sell_imbalance | 매수/매도 불균형 |
| 4-5 | cvd, cvd_derivative | CVD 및 변화율 |
| 6-7 | open_interest, oi_change | OI 및 변화율 |
| 8-11 | momentum, efficiency, velocity, regime | 기타 지표 |

### OUTPUT (출력)

| 출력 이름 | Shape | 설명 |
|-----------|-------|------|
| `trigger_prob` | (batch, 1) | 트리거 발생 확률 0~1 |
| `imminence` | (batch, 1) | 임박도 0~1 (1=바로 직전) |
| `direction_logits` | (batch, 3) | [P(UP), P(DOWN), P(NONE)] |

### 예측 결과 해석

```python
# 추론 시
trigger_prob = output[0]      # 0.85 = 85% 확률
imminence = output[1]         # 0.75 = 트리거까지 약 3.75분 (= 0.25 * 15분 남음)
direction = softmax(output[2])  # [0.8, 0.15, 0.05]

# 트레이딩 의사결정
confidence = trigger_prob * imminence  # 종합 신뢰도
if confidence > 0.6:
    position_size = base_size * confidence
    enter_position(direction='LONG' if direction.argmax() == 0 else 'SHORT')
```

---

## 프로젝트 구조

```
/notebooks/sa/
├── data/                              # 가공된 데이터 (원본 보존)
│   ├── BTCUSDT_perp_5m_labeled.csv   # TRIGGER/IMMINENCE/DIRECTION 추가됨
│   └── BTCUSDT_perp_1h.csv           # 1시간봉 복사본
├── data_preparation/
│   ├── trigger_generator.py          # TRIGGER/IMMINENCE 컬럼 생성
│   ├── multi_timeframe.py            # 5분/1시간 정렬
│   └── dataset.py                    # PyTorch Dataset
├── features/
│   ├── tda_features.py               # TDA 특성 추출
│   ├── market_microstructure.py      # 시장 미세구조
│   └── feature_combiner.py           # 특성 결합
├── models/
│   ├── trigger_model.py              # LSTM+NBEATS+TDA 모델
│   └── nbeats_blocks.py              # N-BEATS 블록
├── training/
│   ├── train_trigger.py              # 훈련 스크립트
│   └── loss.py                       # 손실 함수 (BCE+MSE+CE)
├── evaluation/
│   └── metrics.py                    # 평가 지표
├── checkpoints/                       # 저장된 모델
└── requirements.txt
```

---

## 구현 순서

1. **데이터 준비**: trigger_generator.py → TRIGGER/DIRECTION 컬럼 생성
2. **특성 추출**: 기존 TDA 코드 복사 및 수정
3. **모델 구현**: TriggerPredictionModel 클래스
4. **훈련 파이프라인**: DataLoader, Loss, Optimizer 설정
5. **평가**: F1-Score, AUC-ROC 기반 성능 측정