"""
TDA Trading System - Visualization
시각화 모듈
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import List, Dict

from config import BASE_DIR, INITIAL_CAPITAL


# 한글 폰트 설정 (필요시)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def plot_equity_curve(equity_curve: pd.DataFrame, title: str,
                      save_path: str = None, initial_capital: float = None):
    """
    누적 자산 곡선 그래프

    Args:
        equity_curve: 자산 곡선 DataFrame
        title: 그래프 제목
        save_path: 저장 경로 (None이면 저장 안함)
        initial_capital: 초기 자본
    """
    if initial_capital is None:
        initial_capital = INITIAL_CAPITAL

    fig, ax = plt.subplots(figsize=(14, 6))

    equities = equity_curve['equity'].values
    indices = equity_curve['idx'].values

    # 자산 곡선
    ax.plot(indices, equities, 'b-', linewidth=1.5, label='Equity')

    # 초기 자본 라인
    ax.axhline(y=initial_capital, color='gray', linestyle='--',
               alpha=0.7, label=f'Initial: ${initial_capital:,.0f}')

    # Peak 마커
    peak_idx = equities.argmax()
    peak_value = equities[peak_idx]
    ax.scatter(indices[peak_idx], peak_value, color='green', s=150,
               marker='*', zorder=5, label=f'Peak: ${peak_value:,.0f}')

    # Min 마커
    min_idx = equities.argmin()
    min_value = equities[min_idx]
    ax.scatter(indices[min_idx], min_value, color='red', s=100,
               marker='v', zorder=5, label=f'Min: ${min_value:,.0f}')

    # 최종 값
    final_value = equities[-1]
    ax.scatter(indices[-1], final_value, color='blue', s=100,
               marker='o', zorder=5, label=f'Final: ${final_value:,.0f}')

    ax.set_xlabel('Candle Index')
    ax.set_ylabel('Equity ($)')
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"차트 저장: {save_path}")

    plt.close()


def plot_trades_on_price(df: pd.DataFrame, trades: List[Dict], title: str,
                         save_path: str = None, max_trades: int = 50):
    """
    가격 차트 위에 진입/청산 포인트 표시

    Args:
        df: OHLCV DataFrame
        trades: 거래 리스트
        title: 그래프 제목
        save_path: 저장 경로
        max_trades: 표시할 최대 거래 수
    """
    fig, ax = plt.subplots(figsize=(16, 8))

    # 가격 차트
    closes = df['close'].values
    ax.plot(closes, 'gray', linewidth=0.8, alpha=0.7, label='Close Price')

    # 거래 표시 (최대 max_trades개)
    display_trades = trades[:max_trades] if len(trades) > max_trades else trades

    for trade in display_trades:
        entry_idx = trade['entry_idx']
        exit_idx = trade['exit_idx']
        direction = trade['direction']
        result = trade['result']

        # 진입점
        if direction == 'long':
            ax.scatter(entry_idx, closes[entry_idx], color='green', s=80,
                       marker='^', zorder=5)
        else:
            ax.scatter(entry_idx, closes[entry_idx], color='red', s=80,
                       marker='v', zorder=5)

        # 청산점
        exit_color = 'green' if result == 'win' else 'red'
        ax.scatter(exit_idx, closes[exit_idx], color=exit_color, s=60,
                   marker='x', zorder=5)

        # 진입-청산 연결선
        line_color = 'green' if result == 'win' else 'red'
        ax.plot([entry_idx, exit_idx], [closes[entry_idx], closes[exit_idx]],
                color=line_color, linestyle='--', alpha=0.5, linewidth=1)

    ax.set_xlabel('Candle Index')
    ax.set_ylabel('Price')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # 범례
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='green',
               markersize=10, label='Long Entry'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='red',
               markersize=10, label='Short Entry'),
        Line2D([0], [0], marker='x', color='green', markersize=10,
               linestyle='None', label='Win Exit'),
        Line2D([0], [0], marker='x', color='red', markersize=10,
               linestyle='None', label='Loss Exit'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"차트 저장: {save_path}")

    plt.close()


def plot_comparison(results_dict: Dict, title: str, save_path: str = None):
    """
    레버리지/복리 비교 차트

    Args:
        results_dict: {label: result} 형태의 딕셔너리
        title: 그래프 제목
        save_path: 저장 경로
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    labels = list(results_dict.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

    # 왼쪽: 자산 곡선 비교
    ax1 = axes[0]
    for i, (label, result) in enumerate(results_dict.items()):
        curve = result['equity_curve']
        ax1.plot(curve['idx'], curve['equity'], color=colors[i],
                 linewidth=1.5, label=label)

    ax1.axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Candle Index')
    ax1.set_ylabel('Equity ($)')
    ax1.set_title('Equity Curves')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 오른쪽: 성과 지표 비교 (막대 그래프)
    ax2 = axes[1]

    x = np.arange(len(labels))
    width = 0.35

    finals = [results_dict[l]['metrics']['total_return'] for l in labels]
    peaks = [results_dict[l]['metrics']['peak_return'] for l in labels]

    bars1 = ax2.bar(x - width/2, finals, width, label='Final Return %', color='blue', alpha=0.7)
    bars2 = ax2.bar(x + width/2, peaks, width, label='Peak Return %', color='green', alpha=0.7)

    ax2.set_xlabel('Strategy')
    ax2.set_ylabel('Return (%)')
    ax2.set_title('Performance Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # 값 표시
    for bar in bars1:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=7)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"차트 저장: {save_path}")

    plt.close()


def plot_summary_table(all_results: Dict, save_path: str = None):
    """
    결과 요약 테이블 이미지 생성

    Args:
        all_results: 모든 백테스트 결과
        save_path: 저장 경로
    """
    fig, ax = plt.subplots(figsize=(12, len(all_results) * 0.5 + 2))
    ax.axis('off')

    # 테이블 데이터 준비
    headers = ['Setting', 'Trades', 'Win Rate', 'Final', 'Peak', 'MDD']
    rows = []

    for label, result in all_results.items():
        metrics = result['metrics']
        analysis = result['analysis']
        rows.append([
            label,
            f"{analysis['n_trades']}",
            f"{analysis['win_rate']:.1%}",
            f"${metrics['final']:,.0f} ({metrics['total_return']:+.1f}%)",
            f"${metrics['peak']:,.0f} ({metrics['peak_return']:+.1f}%)",
            f"{metrics['mdd']:.1f}%"
        ])

    table = ax.table(cellText=rows, colLabels=headers, loc='center',
                     cellLoc='center', colColours=['lightblue']*len(headers))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    plt.title('Backtest Results Summary', fontsize=14, fontweight='bold', y=0.95)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"테이블 저장: {save_path}")

    plt.close()
