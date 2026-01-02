"""
Backtest Visualizer

Creates comprehensive visualizations for backtest results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from .position import Trade


class BacktestVisualizer:
    """
    Creates visualizations for backtest results.
    """

    # Color scheme
    COLORS = {
        'long_entry': '#2ECC71',      # Green
        'short_entry': '#E74C3C',     # Red
        'tp_exit': '#3498DB',         # Blue
        'sl_exit': '#F39C12',         # Orange
        'equity_line': '#2980B9',     # Dark blue
        'equity_fill': '#AED6F1',     # Light blue
        'drawdown_fill': '#E74C3C',   # Red
        'positive_pnl': '#27AE60',    # Green
        'negative_pnl': '#C0392B',    # Dark red
        'price_line': '#7F8C8D',      # Gray
    }

    def __init__(self, figsize: Tuple[int, int] = (16, 24)):
        self.figsize = figsize
        plt.style.use('seaborn-v0_8-whitegrid')

    def create_full_report(
        self,
        df_5m: pd.DataFrame,
        trades: List[Trade],
        equity_curve: List[Tuple[pd.Timestamp, float]],
        metrics: Dict,
        output_path: str,
        tda_features: Optional[np.ndarray] = None
    ) -> None:
        """
        Create comprehensive visualization report.

        Args:
            df_5m: 5-minute OHLCV DataFrame
            trades: List of completed trades
            equity_curve: List of (timestamp, equity) tuples
            metrics: Performance metrics dictionary
            output_path: Path to save the report image
        """
        fig = plt.figure(figsize=self.figsize)

        # Adjust grid based on whether TDA features are provided
        if tda_features is not None:
            gs = GridSpec(7, 2, height_ratios=[2, 1.2, 1.2, 1, 1, 1, 0.8], hspace=0.35, wspace=0.3)
        else:
            gs = GridSpec(6, 2, height_ratios=[2.5, 1.5, 1, 1, 1, 0.8], hspace=0.3, wspace=0.3)

        # 1. Price chart with signals (full width)
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_price_with_signals(ax1, df_5m, trades)

        # 2. TDA Features (if provided)
        if tda_features is not None:
            ax_tda = fig.add_subplot(gs[1, :], sharex=ax1)
            self._plot_tda_features(ax_tda, df_5m, tda_features, trades)
            row_offset = 2
        else:
            row_offset = 1

        # Equity curve
        ax2 = fig.add_subplot(gs[row_offset, 0])
        self._plot_equity_curve(ax2, equity_curve, metrics)

        # Drawdown
        ax3 = fig.add_subplot(gs[row_offset, 1])
        self._plot_drawdown(ax3, equity_curve)

        # Trade PnL distribution
        ax4 = fig.add_subplot(gs[row_offset + 1, 0])
        self._plot_pnl_distribution(ax4, trades)

        # Win/Loss by direction
        ax5 = fig.add_subplot(gs[row_offset + 1, 1])
        self._plot_direction_performance(ax5, trades, metrics)

        # Exit reasons pie chart
        ax6 = fig.add_subplot(gs[row_offset + 2, 0])
        self._plot_exit_reasons(ax6, metrics)

        # Cumulative PnL
        ax7 = fig.add_subplot(gs[row_offset + 2, 1])
        self._plot_cumulative_pnl(ax7, trades)

        # Trade duration histogram
        ax8 = fig.add_subplot(gs[row_offset + 3, 0])
        self._plot_duration_histogram(ax8, trades)

        # Monthly returns heatmap (simplified)
        ax9 = fig.add_subplot(gs[row_offset + 3, 1])
        self._plot_monthly_returns(ax9, trades)

        # 10. Metrics summary table (full width)
        ax10 = fig.add_subplot(gs[row_offset + 4, :])
        self._plot_metrics_table(ax10, metrics)

        # Title
        fig.suptitle('Backtest Report - Trigger-Based Trading Strategy',
                     fontsize=14, fontweight='bold', y=0.995)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()

    def _plot_price_with_signals(
        self,
        ax,
        df_5m: pd.DataFrame,
        trades: List[Trade]
    ) -> None:
        """Plot price chart with entry/exit markers."""
        # Convert timestamps
        df_5m = df_5m.copy()
        df_5m['open_time'] = pd.to_datetime(df_5m['open_time'])

        # Plot price line
        ax.plot(df_5m['open_time'], df_5m['close'],
                color=self.COLORS['price_line'], alpha=0.7, linewidth=0.5,
                label='Price')

        # Plot trades
        for trade in trades:
            entry_time = pd.to_datetime(trade.entry_time)
            exit_time = pd.to_datetime(trade.exit_time)

            # Entry marker
            if trade.direction == 'LONG':
                ax.scatter(entry_time, trade.entry_price,
                          c=self.COLORS['long_entry'], marker='^',
                          s=80, zorder=5, label='_nolegend_')
            else:
                ax.scatter(entry_time, trade.entry_price,
                          c=self.COLORS['short_entry'], marker='v',
                          s=80, zorder=5, label='_nolegend_')

            # Exit marker
            if trade.exit_reason == 'TP':
                exit_color = self.COLORS['tp_exit']
            else:
                exit_color = self.COLORS['sl_exit']

            ax.scatter(exit_time, trade.exit_price,
                      c=exit_color, marker='o', s=50, zorder=5, label='_nolegend_')

            # Connect entry to exit
            line_color = self.COLORS['long_entry'] if trade.direction == 'LONG' else self.COLORS['short_entry']
            ax.plot([entry_time, exit_time],
                   [trade.entry_price, trade.exit_price],
                   color=line_color, alpha=0.3, linewidth=1)

        # Custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=self.COLORS['price_line'], linewidth=1, label='Price'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor=self.COLORS['long_entry'],
                   markersize=10, label='Long Entry'),
            Line2D([0], [0], marker='v', color='w', markerfacecolor=self.COLORS['short_entry'],
                   markersize=10, label='Short Entry'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=self.COLORS['tp_exit'],
                   markersize=8, label='TP Exit'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=self.COLORS['sl_exit'],
                   markersize=8, label='SL Exit'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=8)

        ax.set_title('Price Chart with Trade Signals', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    def _plot_equity_curve(
        self,
        ax,
        equity_curve: List[Tuple],
        metrics: Dict
    ) -> None:
        """Plot equity curve over time."""
        if not equity_curve:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            return

        times = [pd.to_datetime(e[0]) for e in equity_curve]
        values = [e[1] for e in equity_curve]

        ax.plot(times, values, color=self.COLORS['equity_line'], linewidth=1.5)
        ax.fill_between(times, values[0], values,
                       color=self.COLORS['equity_fill'], alpha=0.3)
        ax.axhline(y=values[0], color='gray', linestyle='--', alpha=0.5, linewidth=1)

        # Annotate final value
        ax.annotate(f"${values[-1]:,.0f}\n({metrics['total_return_pct']:+.1f}%)",
                   xy=(times[-1], values[-1]),
                   fontsize=9, fontweight='bold',
                   color=self.COLORS['positive_pnl'] if values[-1] > values[0] else self.COLORS['negative_pnl'])

        ax.set_title('Equity Curve', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Equity ($)')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    def _plot_drawdown(self, ax, equity_curve: List[Tuple]) -> None:
        """Plot drawdown chart."""
        if not equity_curve:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            return

        times = [pd.to_datetime(e[0]) for e in equity_curve]
        values = [e[1] for e in equity_curve]

        # Calculate drawdown
        peak = values[0]
        drawdowns = []
        for v in values:
            if v > peak:
                peak = v
            drawdowns.append((peak - v) / peak * 100)

        ax.fill_between(times, 0, drawdowns,
                       color=self.COLORS['drawdown_fill'], alpha=0.5)
        ax.plot(times, drawdowns, color=self.COLORS['drawdown_fill'], linewidth=1)

        max_dd = max(drawdowns)
        ax.axhline(y=max_dd, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax.annotate(f'Max: {max_dd:.1f}%', xy=(times[0], max_dd),
                   fontsize=9, color='red')

        ax.set_title('Drawdown (%)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Drawdown %')
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    def _plot_pnl_distribution(self, ax, trades: List[Trade]) -> None:
        """Plot PnL distribution histogram."""
        if not trades:
            ax.text(0.5, 0.5, 'No trades', ha='center', va='center')
            return

        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        bins = 20
        if wins:
            ax.hist(wins, bins=bins, color=self.COLORS['positive_pnl'],
                   alpha=0.7, label=f'Wins ({len(wins)})')
        if losses:
            ax.hist(losses, bins=bins, color=self.COLORS['negative_pnl'],
                   alpha=0.7, label=f'Losses ({len(losses)})')

        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.axvline(x=np.mean(pnls), color='blue', linestyle='--',
                  linewidth=1, label=f'Mean: ${np.mean(pnls):.2f}')

        ax.set_title('PnL Distribution', fontsize=11, fontweight='bold')
        ax.set_xlabel('PnL ($)')
        ax.set_ylabel('Frequency')
        ax.legend(fontsize=8)

    def _plot_direction_performance(
        self,
        ax,
        trades: List[Trade],
        metrics: Dict
    ) -> None:
        """Plot long vs short performance."""
        directions = ['Long', 'Short']
        counts = [metrics['long_trades'], metrics['short_trades']]
        pnls = [metrics['long_pnl'], metrics['short_pnl']]
        win_rates = [metrics['long_win_rate'] * 100, metrics['short_win_rate'] * 100]

        x = np.arange(len(directions))
        width = 0.35

        # Bar chart for PnL
        colors = [self.COLORS['positive_pnl'] if p > 0 else self.COLORS['negative_pnl'] for p in pnls]
        bars = ax.bar(x, pnls, width, color=colors, alpha=0.8)

        # Add win rate annotations
        for i, (bar, wr, cnt) in enumerate(zip(bars, win_rates, counts)):
            height = bar.get_height()
            ax.annotate(f'WR: {wr:.0f}%\nn={cnt}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       ha='center', va='bottom' if height > 0 else 'top',
                       fontsize=9)

        ax.set_title('Long vs Short Performance', fontsize=11, fontweight='bold')
        ax.set_xlabel('Direction')
        ax.set_ylabel('Total PnL ($)')
        ax.set_xticks(x)
        ax.set_xticklabels(directions)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    def _plot_exit_reasons(self, ax, metrics: Dict) -> None:
        """Plot exit reasons pie chart."""
        labels = []
        sizes = []
        colors = []

        if metrics['tp_exits'] > 0:
            labels.append(f"TP ({metrics['tp_exits']})")
            sizes.append(metrics['tp_exits'])
            colors.append(self.COLORS['tp_exit'])

        if metrics['sl_exits'] > 0:
            labels.append(f"SL ({metrics['sl_exits']})")
            sizes.append(metrics['sl_exits'])
            colors.append(self.COLORS['sl_exit'])

        if metrics['other_exits'] > 0:
            labels.append(f"Other ({metrics['other_exits']})")
            sizes.append(metrics['other_exits'])
            colors.append('#95A5A6')

        if sizes:
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%',
                  startangle=90, textprops={'fontsize': 9})
        else:
            ax.text(0.5, 0.5, 'No trades', ha='center', va='center')

        ax.set_title('Exit Reasons', fontsize=11, fontweight='bold')

    def _plot_cumulative_pnl(self, ax, trades: List[Trade]) -> None:
        """Plot cumulative PnL over trades."""
        if not trades:
            ax.text(0.5, 0.5, 'No trades', ha='center', va='center')
            return

        cumulative = np.cumsum([t.pnl for t in trades])

        ax.plot(range(1, len(cumulative) + 1), cumulative,
               color=self.COLORS['equity_line'], linewidth=1.5)
        ax.fill_between(range(1, len(cumulative) + 1), 0, cumulative,
                       where=(cumulative >= 0), color=self.COLORS['positive_pnl'], alpha=0.3)
        ax.fill_between(range(1, len(cumulative) + 1), 0, cumulative,
                       where=(cumulative < 0), color=self.COLORS['negative_pnl'], alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        ax.set_title('Cumulative PnL', fontsize=11, fontweight='bold')
        ax.set_xlabel('Trade Number')
        ax.set_ylabel('Cumulative PnL ($)')

    def _plot_duration_histogram(self, ax, trades: List[Trade]) -> None:
        """Plot trade duration histogram."""
        if not trades:
            ax.text(0.5, 0.5, 'No trades', ha='center', va='center')
            return

        durations = [t.duration_candles * 5 / 60 for t in trades]  # Convert to hours

        ax.hist(durations, bins=20, color=self.COLORS['equity_line'],
               alpha=0.7, edgecolor='white')
        ax.axvline(x=np.mean(durations), color='red', linestyle='--',
                  linewidth=1.5, label=f'Mean: {np.mean(durations):.1f}h')

        ax.set_title('Trade Duration', fontsize=11, fontweight='bold')
        ax.set_xlabel('Duration (hours)')
        ax.set_ylabel('Frequency')
        ax.legend(fontsize=8)

    def _plot_monthly_returns(self, ax, trades: List[Trade]) -> None:
        """Plot monthly returns bar chart."""
        if not trades:
            ax.text(0.5, 0.5, 'No trades', ha='center', va='center')
            return

        # Group by month
        monthly_pnl = {}
        for trade in trades:
            month_key = pd.to_datetime(trade.exit_time).strftime('%Y-%m')
            monthly_pnl[month_key] = monthly_pnl.get(month_key, 0) + trade.pnl

        months = list(monthly_pnl.keys())
        pnls = list(monthly_pnl.values())
        colors = [self.COLORS['positive_pnl'] if p > 0 else self.COLORS['negative_pnl'] for p in pnls]

        ax.bar(months, pnls, color=colors, alpha=0.8)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        ax.set_title('Monthly Returns', fontsize=11, fontweight='bold')
        ax.set_xlabel('Month')
        ax.set_ylabel('PnL ($)')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    def _plot_metrics_table(self, ax, metrics: Dict) -> None:
        """Plot metrics summary table."""
        ax.axis('off')

        # Select key metrics for display
        table_data = [
            ['Total Return', f"{metrics['total_return_pct']:.2f}%",
             'Win Rate', f"{metrics['win_rate']*100:.1f}%",
             'Sharpe Ratio', f"{metrics['sharpe_ratio']:.2f}"],
            ['Total Trades', f"{metrics['total_trades']}",
             'Profit Factor', f"{metrics['profit_factor']:.2f}",
             'Max Drawdown', f"{metrics['max_drawdown_pct']:.2f}%"],
            ['Total PnL', f"${metrics['total_pnl']:.2f}",
             'Avg Trade', f"${metrics['avg_trade_pnl']:.2f}",
             'Total Fees', f"${metrics['total_fees']:.2f}"],
        ]

        table = ax.table(
            cellText=table_data,
            loc='center',
            cellLoc='center',
            colWidths=[0.12, 0.1, 0.12, 0.1, 0.12, 0.1]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        # Style header-like cells (odd columns)
        for i in range(3):
            for j in [0, 2, 4]:
                table[(i, j)].set_facecolor('#E8E8E8')
                table[(i, j)].set_text_props(fontweight='bold')

    def _plot_tda_features(
        self,
        ax,
        df_5m: pd.DataFrame,
        tda_features: np.ndarray,
        trades: List[Trade]
    ) -> None:
        """
        Plot TDA features on a separate chart.

        TDA features (9 total):
        - [0, 1, 2]: config0 (dim=3, tau=1) - entropy, amplitude, num_points
        - [3, 4, 5]: config1 (dim=5, tau=2) - entropy, amplitude, num_points
        - [6, 7, 8]: config2 (dim=7, tau=3) - entropy, amplitude, num_points
        """
        df_5m = df_5m.copy()
        df_5m['open_time'] = pd.to_datetime(df_5m['open_time'])
        times = df_5m['open_time'].values

        # Create twin axes for different scales
        ax2 = ax.twinx()

        # Plot entropy (average of 3 configs) - left axis
        entropy_avg = (tda_features[:, 0] + tda_features[:, 3] + tda_features[:, 6]) / 3
        ax.plot(times, entropy_avg, color='#9B59B6', alpha=0.8, linewidth=0.8, label='Entropy (avg)')
        ax.fill_between(times, 0, entropy_avg, color='#9B59B6', alpha=0.2)

        # Plot amplitude (average of 3 configs) - right axis
        amplitude_avg = (tda_features[:, 1] + tda_features[:, 4] + tda_features[:, 7]) / 3
        ax2.plot(times, amplitude_avg, color='#E67E22', alpha=0.8, linewidth=0.8, label='Amplitude (avg)')

        # Mark trade entry points on TDA chart
        for trade in trades:
            entry_time = pd.to_datetime(trade.entry_time)
            # Find closest index
            idx = np.argmin(np.abs(df_5m['open_time'] - entry_time))
            if idx < len(entropy_avg):
                color = self.COLORS['long_entry'] if trade.direction == 'LONG' else self.COLORS['short_entry']
                marker = '^' if trade.direction == 'LONG' else 'v'
                ax.scatter(entry_time, entropy_avg[idx], c=color, marker=marker, s=60, zorder=5)

        # Labels and legend
        ax.set_ylabel('Entropy', color='#9B59B6', fontsize=9)
        ax2.set_ylabel('Amplitude', color='#E67E22', fontsize=9)
        ax.tick_params(axis='y', labelcolor='#9B59B6')
        ax2.tick_params(axis='y', labelcolor='#E67E22')

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

        ax.set_title('TDA Features (Topological Data Analysis)', fontsize=11, fontweight='bold')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
