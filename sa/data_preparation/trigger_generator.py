"""
Trigger Label Generator for Trading Signal Prediction

Generates TRIGGER, IMMINENCE, and DIRECTION labels for training data.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from tqdm import tqdm


class TriggerGenerator:
    """
    Generates trigger labels for price movement prediction.

    TRIGGER: 1 if a 2% price movement occurs within 2 hours, 0 otherwise
    IMMINENCE: 0~1 score indicating how soon the trigger will occur (1 = imminent)
    DIRECTION: 0=UP, 1=DOWN, 2=NONE
    """

    def __init__(
        self,
        threshold_pct: float = 0.02,
        look_forward_candles: int = 24,
        pre_trigger_candles: int = 3
    ):
        """
        Args:
            threshold_pct: Price change threshold (0.02 = 2%)
            look_forward_candles: Number of candles to look ahead (24 = 2 hours for 5-min)
            pre_trigger_candles: Number of candles before trigger to mark (3 = 15 min for 5-min)
        """
        self.threshold_pct = threshold_pct
        self.look_forward_candles = look_forward_candles
        self.pre_trigger_candles = pre_trigger_candles

    def generate_triggers(self, df: pd.DataFrame, show_progress: bool = True) -> pd.DataFrame:
        """
        Generate TRIGGER, IMMINENCE, and DIRECTION columns.

        Args:
            df: DataFrame with 'close' column
            show_progress: Whether to show progress bar

        Returns:
            DataFrame with added columns: TRIGGER, IMMINENCE, DIRECTION
        """
        df = df.copy()
        n = len(df)

        trigger = np.zeros(n, dtype=np.int32)
        imminence = np.zeros(n, dtype=np.float32)
        direction = np.full(n, 2, dtype=np.int32)  # Default: NONE

        close_prices = df['close'].values

        iterator = range(n)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating triggers")

        for i in iterator:
            current_price = close_prices[i]

            # Look forward to find first trigger point
            for j in range(i + 1, min(i + 1 + self.look_forward_candles, n)):
                future_price = close_prices[j]
                change = (future_price - current_price) / current_price

                if abs(change) >= self.threshold_pct:
                    trigger_point = j
                    dir_val = 0 if change > 0 else 1  # 0=UP, 1=DOWN

                    # Mark pre-trigger zone (from pre_trigger candles before to trigger point)
                    start_idx = max(0, trigger_point - self.pre_trigger_candles)
                    for k in range(start_idx, trigger_point + 1):
                        trigger[k] = 1
                        direction[k] = dir_val

                        # Calculate imminence score
                        candles_to_trigger = trigger_point - k
                        imminence[k] = 1.0 - (candles_to_trigger / (self.pre_trigger_candles + 1))

                    break

        df['TRIGGER'] = trigger
        df['IMMINENCE'] = imminence
        df['DIRECTION'] = direction

        return df

    def get_label_statistics(self, df: pd.DataFrame) -> dict:
        """
        Get statistics about the generated labels.

        Args:
            df: DataFrame with TRIGGER, IMMINENCE, DIRECTION columns

        Returns:
            Dictionary with label statistics
        """
        total = len(df)
        trigger_count = df['TRIGGER'].sum()

        stats = {
            'total_samples': total,
            'trigger_count': int(trigger_count),
            'trigger_ratio': trigger_count / total if total > 0 else 0,
            'non_trigger_count': int(total - trigger_count),
            'class_imbalance_ratio': (total - trigger_count) / trigger_count if trigger_count > 0 else float('inf'),
        }

        # Direction distribution for TRIGGER=1 samples
        trigger_df = df[df['TRIGGER'] == 1]
        if len(trigger_df) > 0:
            stats['up_count'] = int((trigger_df['DIRECTION'] == 0).sum())
            stats['down_count'] = int((trigger_df['DIRECTION'] == 1).sum())
            stats['up_ratio'] = stats['up_count'] / len(trigger_df)
            stats['down_ratio'] = stats['down_count'] / len(trigger_df)
            stats['avg_imminence'] = float(trigger_df['IMMINENCE'].mean())

        return stats


def process_and_save_data(
    input_path_5m: str,
    input_path_1h: str,
    output_dir: str,
    threshold_pct: float = 0.02,
    look_forward_candles: int = 24,
    pre_trigger_candles: int = 3
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Process raw data and save labeled data to output directory.

    Args:
        input_path_5m: Path to 5-minute candle data
        input_path_1h: Path to 1-hour candle data
        output_dir: Directory to save processed data
        threshold_pct: Price change threshold
        look_forward_candles: Look-ahead window
        pre_trigger_candles: Pre-trigger marking window

    Returns:
        Tuple of (df_5m_labeled, df_1h, statistics)
    """
    import os

    print("Loading data...")
    df_5m = pd.read_csv(input_path_5m)
    df_1h = pd.read_csv(input_path_1h)

    print(f"5-min data shape: {df_5m.shape}")
    print(f"1-hour data shape: {df_1h.shape}")

    # Generate trigger labels
    generator = TriggerGenerator(
        threshold_pct=threshold_pct,
        look_forward_candles=look_forward_candles,
        pre_trigger_candles=pre_trigger_candles
    )

    print("\nGenerating trigger labels...")
    df_5m_labeled = generator.generate_triggers(df_5m)

    # Get statistics
    stats = generator.get_label_statistics(df_5m_labeled)

    print("\n=== Label Statistics ===")
    print(f"Total samples: {stats['total_samples']:,}")
    print(f"TRIGGER=1: {stats['trigger_count']:,} ({stats['trigger_ratio']:.2%})")
    print(f"TRIGGER=0: {stats['non_trigger_count']:,}")
    print(f"Class imbalance ratio: {stats['class_imbalance_ratio']:.2f}:1")

    if 'up_count' in stats:
        print(f"\nDirection distribution (TRIGGER=1 only):")
        print(f"  UP: {stats['up_count']:,} ({stats['up_ratio']:.2%})")
        print(f"  DOWN: {stats['down_count']:,} ({stats['down_ratio']:.2%})")
        print(f"  Avg Imminence: {stats['avg_imminence']:.3f}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save processed data
    output_5m_path = os.path.join(output_dir, 'BTCUSDT_perp_5m_labeled.csv')
    output_1h_path = os.path.join(output_dir, 'BTCUSDT_perp_1h.csv')

    print(f"\nSaving labeled data to {output_5m_path}...")
    df_5m_labeled.to_csv(output_5m_path, index=False)

    print(f"Copying 1-hour data to {output_1h_path}...")
    df_1h.to_csv(output_1h_path, index=False)

    print("Done!")

    return df_5m_labeled, df_1h, stats


if __name__ == "__main__":
    # Default paths
    INPUT_5M = '/notebooks/data/BTCUSDT_perp_etf_to_90d_ago.csv'
    INPUT_1H = '/notebooks/data/BTCUSDT_perp_1h_etf_to_90d_ago.csv'
    OUTPUT_DIR = '/notebooks/sa/data'

    df_5m, df_1h, stats = process_and_save_data(
        input_path_5m=INPUT_5M,
        input_path_1h=INPUT_1H,
        output_dir=OUTPUT_DIR
    )
