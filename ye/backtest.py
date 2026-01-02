"""
TDA Trading System - Backtest
트레이딩 시뮬레이션 및 성과 분석
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

from config import TP_PCT, SL_PCT, HOLD_CANDLES, COOLDOWN, INITIAL_CAPITAL


def simulate_single(df: pd.DataFrame, probs: np.ndarray, direction: str,
                    threshold: float = 0.5, tp_pct: float = None,
                    sl_pct: float = None, hold_candles: int = None,
                    cooldown: int = None) -> List[Dict]:
    """
    단일 방향 트레이딩 시뮬레이션

    Args:
        df: OHLCV DataFrame
        probs: 예측 확률 배열
        direction: 'long' 또는 'short'
        threshold: 진입 임계값
        tp_pct: Take Profit 비율
        sl_pct: Stop Loss 비율
        hold_candles: 최대 보유 캔들 수
        cooldown: 진입 후 쿨다운

    Returns:
        거래 리스트 (각 거래는 딕셔너리)
    """
    if tp_pct is None:
        tp_pct = TP_PCT
    if sl_pct is None:
        sl_pct = SL_PCT
    if hold_candles is None:
        hold_candles = HOLD_CANDLES
    if cooldown is None:
        cooldown = COOLDOWN

    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values

    trades = []
    cooldown_counter = 0

    for i in range(len(df) - hold_candles):
        if cooldown_counter > 0:
            cooldown_counter -= 1
            continue

        if probs[i] >= threshold:
            entry_price = closes[i]
            entry_time = df.iloc[i].get('open_time', i)

            if direction == 'long':
                tp_price = entry_price * (1 + tp_pct)
                sl_price = entry_price * (1 - sl_pct)
            else:  # short
                tp_price = entry_price * (1 - tp_pct)
                sl_price = entry_price * (1 + sl_pct)

            result = None
            exit_idx = None

            for j in range(1, hold_candles + 1):
                idx = i + j
                if idx >= len(df):
                    break

                if direction == 'long':
                    # Long: SL 먼저 체크 (low가 SL 이하)
                    if lows[idx] <= sl_price:
                        result = ('loss', -sl_pct)
                        exit_idx = idx
                        break
                    # TP 체크 (high가 TP 이상)
                    if highs[idx] >= tp_price:
                        result = ('win', tp_pct)
                        exit_idx = idx
                        break
                else:  # short
                    # Short: SL 먼저 체크 (high가 SL 이상)
                    if highs[idx] >= sl_price:
                        result = ('loss', -sl_pct)
                        exit_idx = idx
                        break
                    # TP 체크 (low가 TP 이하)
                    if lows[idx] <= tp_price:
                        result = ('win', tp_pct)
                        exit_idx = idx
                        break

            # MAX_HOLD 도달 시 시장가 청산
            if result is None:
                exit_idx = min(i + hold_candles, len(df) - 1)
                exit_price = closes[exit_idx]
                if direction == 'long':
                    pnl = (exit_price - entry_price) / entry_price
                else:
                    pnl = (entry_price - exit_price) / entry_price
                result = ('win' if pnl > 0 else 'loss', pnl)

            exit_time = df.iloc[exit_idx].get('open_time', exit_idx)

            trades.append({
                'entry_idx': i,
                'exit_idx': exit_idx,
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_price': entry_price,
                'direction': direction,
                'result': result[0],
                'pnl': result[1],
                'actual_trigger': df.iloc[i]['trigger']
            })

            cooldown_counter = cooldown

    return trades


def simulate_combined(df_long: pd.DataFrame, df_short: pd.DataFrame,
                      long_probs: np.ndarray, short_probs: np.ndarray,
                      threshold: float = 0.5, **kwargs) -> List[Dict]:
    """
    Long + Short 통합 시뮬레이션

    동일 시점에 Long/Short 신호가 동시에 발생하면 더 높은 확률의 방향 선택

    Returns:
        시간순 정렬된 거래 리스트
    """
    long_trades = simulate_single(df_long, long_probs, 'long', threshold, **kwargs)
    short_trades = simulate_single(df_short, short_probs, 'short', threshold, **kwargs)

    # 시간순 정렬
    all_trades = long_trades + short_trades
    all_trades.sort(key=lambda x: x['entry_idx'])

    return all_trades


def calculate_equity_curve(trades: List[Dict], initial_capital: float = None,
                           leverage: float = 1, compound: bool = True) -> pd.DataFrame:
    """
    누적 자산 곡선 계산

    Args:
        trades: 거래 리스트
        initial_capital: 초기 자본
        leverage: 레버리지
        compound: True=복리, False=단리

    Returns:
        자산 곡선 DataFrame (columns: idx, equity, pnl)
    """
    if initial_capital is None:
        initial_capital = INITIAL_CAPITAL

    if not trades:
        return pd.DataFrame({
            'idx': [0],
            'equity': [initial_capital],
            'pnl': [0]
        })

    equity = initial_capital
    curve = []

    for trade in trades:
        pnl_pct = trade['pnl'] * leverage

        if compound:
            # 복리: 현재 자산 기준
            pnl_amount = equity * pnl_pct
        else:
            # 단리: 초기 자산 기준
            pnl_amount = initial_capital * pnl_pct

        equity += pnl_amount

        curve.append({
            'idx': trade['exit_idx'],
            'entry_idx': trade['entry_idx'],
            'equity': equity,
            'pnl': pnl_amount,
            'direction': trade['direction'],
            'result': trade['result']
        })

    return pd.DataFrame(curve)


def calculate_metrics(equity_curve: pd.DataFrame, initial_capital: float = None) -> Dict:
    """
    성과 지표 계산

    Returns:
        딕셔너리:
        - final: 최종 자산
        - peak: 최고 자산
        - min: 최저 자산
        - mdd: 최대 낙폭 (%)
        - total_return: 총 수익률 (%)
    """
    if initial_capital is None:
        initial_capital = INITIAL_CAPITAL

    if len(equity_curve) == 0:
        return {
            'final': initial_capital,
            'peak': initial_capital,
            'min': initial_capital,
            'mdd': 0,
            'total_return': 0
        }

    equities = equity_curve['equity'].values
    final = equities[-1]
    peak = equities.max()
    min_equity = equities.min()

    # MDD 계산
    running_max = np.maximum.accumulate(equities)
    drawdowns = (running_max - equities) / running_max
    mdd = drawdowns.max() * 100

    total_return = (final - initial_capital) / initial_capital * 100

    return {
        'final': final,
        'peak': peak,
        'min': min_equity,
        'mdd': mdd,
        'total_return': total_return,
        'peak_return': (peak - initial_capital) / initial_capital * 100
    }


def analyze_trades(trades: List[Dict]) -> Dict:
    """
    거래 분석

    Returns:
        분석 결과 딕셔너리
    """
    if not trades:
        return {
            'n_trades': 0,
            'win_rate': 0,
            'avg_pnl': 0,
            'total_pnl': 0,
            'long_trades': 0,
            'short_trades': 0
        }

    df = pd.DataFrame(trades)

    wins = (df['result'] == 'win').sum()
    total = len(df)
    win_rate = wins / total if total > 0 else 0

    return {
        'n_trades': total,
        'win_rate': win_rate,
        'avg_pnl': df['pnl'].mean(),
        'total_pnl': df['pnl'].sum(),
        'long_trades': (df['direction'] == 'long').sum(),
        'short_trades': (df['direction'] == 'short').sum(),
        'trigger_hit_rate': df['actual_trigger'].mean()
    }


def run_full_backtest(trades: List[Dict], leverage: float, compound: bool,
                      initial_capital: float = None) -> Dict:
    """
    전체 백테스트 실행 및 결과 반환

    Returns:
        trades, equity_curve, metrics, analysis 포함 딕셔너리
    """
    if initial_capital is None:
        initial_capital = INITIAL_CAPITAL

    equity_curve = calculate_equity_curve(trades, initial_capital, leverage, compound)
    metrics = calculate_metrics(equity_curve, initial_capital)
    analysis = analyze_trades(trades)

    return {
        'trades': trades,
        'equity_curve': equity_curve,
        'metrics': metrics,
        'analysis': analysis,
        'leverage': leverage,
        'compound': compound
    }


def print_backtest_results(results: Dict, name: str = "Backtest"):
    """
    백테스트 결과 출력
    """
    metrics = results['metrics']
    analysis = results['analysis']
    leverage = results['leverage']
    compound_str = "복리" if results['compound'] else "단리"

    print(f"\n{'='*50}")
    print(f"{name} | {leverage}x {compound_str}")
    print(f"{'='*50}")
    print(f"거래 수: {analysis['n_trades']} (Long: {analysis['long_trades']}, Short: {analysis['short_trades']})")
    print(f"승률: {analysis['win_rate']:.2%}")
    print(f"평균 PnL: {analysis['avg_pnl']:.4%}")
    print(f"{'─'*50}")
    print(f"초기 자본: ${INITIAL_CAPITAL:,.0f}")
    print(f"최종 자산: ${metrics['final']:,.0f} ({metrics['total_return']:+.1f}%)")
    print(f"최고점: ${metrics['peak']:,.0f} ({metrics['peak_return']:+.1f}%)")
    print(f"최저점: ${metrics['min']:,.0f}")
    print(f"MDD: {metrics['mdd']:.1f}%")
