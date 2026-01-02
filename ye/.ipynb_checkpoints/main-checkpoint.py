"""
TDA Trading System - Main Execution
메인 실행 스크립트
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 현재 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    THRESHOLDS, LEVERAGES, INITIAL_CAPITAL, BASE_DIR,
    TP_PCT, SL_PCT, HOLD_CANDLES, COOLDOWN
)
from data_loader import load_data
from tda_features import add_tda_features_to_df
from model import train_model, predict_proba
from backtest import (
    simulate_single, simulate_combined, run_full_backtest,
    print_backtest_results, analyze_trades
)
from visualize import (
    plot_equity_curve, plot_trades_on_price,
    plot_comparison, plot_summary_table
)


def main():
    print("=" * 60)
    print("TDA Trading System v2.0")
    print("=" * 60)

    # ========================================
    # 1. 데이터 로딩
    # ========================================
    print("\n[1/5] 데이터 로딩...")

    train_long = load_data('train_long')
    train_short = load_data('train_short')
    backtest1_long = load_data('backtest1_long')
    backtest1_short = load_data('backtest1_short')
    backtest2_long = load_data('backtest2_long')
    backtest2_short = load_data('backtest2_short')

    # ========================================
    # 2. TDA 피처 추출
    # ========================================
    print("\n[2/5] TDA 피처 추출...")

    train_long = add_tda_features_to_df(train_long)
    train_short = add_tda_features_to_df(train_short)
    backtest1_long = add_tda_features_to_df(backtest1_long)
    backtest1_short = add_tda_features_to_df(backtest1_short)
    backtest2_long = add_tda_features_to_df(backtest2_long)
    backtest2_short = add_tda_features_to_df(backtest2_short)

    # ========================================
    # 3. 모델 학습
    # ========================================
    print("\n[3/5] 모델 학습...")

    long_model, long_features = train_model(train_long, "LONG")
    short_model, short_features = train_model(train_short, "SHORT")

    # ========================================
    # 4. 백테스트 실행
    # ========================================
    print("\n[4/5] 백테스트 실행...")

    # 예측 확률 계산
    bt1_long_probs = predict_proba(long_model, backtest1_long, long_features)
    bt1_short_probs = predict_proba(short_model, backtest1_short, short_features)
    bt2_long_probs = predict_proba(long_model, backtest2_long, long_features)
    bt2_short_probs = predict_proba(short_model, backtest2_short, short_features)

    # 결과 저장
    all_results = {}

    for threshold in THRESHOLDS:
        print(f"\n{'─'*50}")
        print(f"Threshold: {threshold}")
        print(f"{'─'*50}")

        # Backtest1 거래 생성
        bt1_trades = simulate_combined(
            backtest1_long, backtest1_short,
            bt1_long_probs, bt1_short_probs,
            threshold=threshold
        )

        # Backtest2 거래 생성
        bt2_trades = simulate_combined(
            backtest2_long, backtest2_short,
            bt2_long_probs, bt2_short_probs,
            threshold=threshold
        )

        # 거래 분석
        bt1_analysis = analyze_trades(bt1_trades)
        bt2_analysis = analyze_trades(bt2_trades)

        print(f"\nBacktest1: {bt1_analysis['n_trades']} trades, "
              f"Win Rate: {bt1_analysis['win_rate']:.1%}")
        print(f"Backtest2: {bt2_analysis['n_trades']} trades, "
              f"Win Rate: {bt2_analysis['win_rate']:.1%}")

        # 레버리지별 테스트
        for leverage in LEVERAGES:
            for compound in [True, False]:
                comp_str = "복리" if compound else "단리"
                label = f"Th{threshold}_Lev{leverage}x_{comp_str}"

                # Backtest2 결과
                result = run_full_backtest(bt2_trades, leverage, compound)
                all_results[label] = result

                # 결과 출력
                if threshold == 0.5:  # 0.5 임계값만 상세 출력
                    print_backtest_results(result, f"BT2 {leverage}x {comp_str}")

    # ========================================
    # 5. 시각화
    # ========================================
    print("\n[5/5] 시각화 생성...")

    # Threshold 0.5 결과만 시각화
    viz_results = {k: v for k, v in all_results.items() if k.startswith('Th0.5')}

    # 비교 차트
    plot_comparison(
        viz_results,
        "Backtest2 Performance Comparison (Threshold 0.5)",
        os.path.join(BASE_DIR, "comparison_chart.png")
    )

    # 개별 자산 곡선
    for label, result in viz_results.items():
        plot_equity_curve(
            result['equity_curve'],
            f"Equity Curve - {label}",
            os.path.join(BASE_DIR, f"equity_{label}.png")
        )

    # 요약 테이블
    plot_summary_table(
        viz_results,
        os.path.join(BASE_DIR, "summary_table.png")
    )

    # 거래 차트 (10x 단리 예시)
    key_result = all_results.get('Th0.5_Lev10x_단리')
    if key_result and key_result['trades']:
        plot_trades_on_price(
            backtest2_long,
            key_result['trades'],
            "Trade Points - Backtest2 (Threshold 0.5, 10x)",
            os.path.join(BASE_DIR, "trades_chart.png")
        )

    # ========================================
    # 최종 결과 요약
    # ========================================
    print("\n" + "=" * 60)
    print("최종 결과 요약 (Threshold 0.5, Backtest2)")
    print("=" * 60)

    print(f"\n{'Setting':<20} {'Final':>15} {'Peak':>15} {'MDD':>10}")
    print("-" * 60)

    for label in viz_results:
        m = all_results[label]['metrics']
        short_label = label.replace('Th0.5_', '')
        print(f"{short_label:<20} "
              f"${m['final']:>10,.0f} ({m['total_return']:+5.1f}%) "
              f"${m['peak']:>10,.0f} ({m['peak_return']:+5.1f}%) "
              f"{m['mdd']:>8.1f}%")

    print("\n완료!")
    print(f"차트 저장 위치: {BASE_DIR}")


if __name__ == "__main__":
    main()
