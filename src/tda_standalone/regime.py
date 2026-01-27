"""
Regime Detection Module.

Uses K-means clustering on H1-based features to identify market regimes.
These regime labels serve as self-supervised auxiliary targets.

Regime interpretations (based on H1 analysis):
- Regime 0: "Simple" - low entropy, low persistence, low betti (trending market)
- Regime 1: "Chaotic" - high entropy (choppy, avoid trading)
- Regime 2: "Cycling" - high persistence (mean-reversion opportunity)
- Regime 3: "Complex" - high betti count (consolidation)
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import pickle
from pathlib import Path


class RegimeLabeler:
    """
    Computes regime labels using K-means on H1-based features.

    Uses:
    - H1 Betti mean (bins 50-69)
    - H1 Entropy (index 101)
    - H1 Persistence (index 103)
    """

    def __init__(self, n_regimes: int = 4, random_state: int = 42):
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.kmeans: Optional[KMeans] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_fitted = False
        self.regime_stats: Optional[dict] = None

    def _extract_h1_features(self, tda_features: np.ndarray, betti_bins: int = 20) -> np.ndarray:
        """
        Extract H1-based features for regime clustering.

        Args:
            tda_features: (N, 214) TDA features
            betti_bins: Number of Betti bins to use for mean (default 20)

        Returns:
            (N, 3) array of [h1_betti_mean, h1_entropy, h1_persistence]
        """
        # H1 Betti mean (first `betti_bins` bins)
        h1_betti = tda_features[:, 50:50+betti_bins]
        h1_betti_mean = h1_betti.mean(axis=1, keepdims=True)

        # H1 Entropy
        h1_entropy = tda_features[:, 101:102]

        # H1 Persistence
        h1_persistence = tda_features[:, 103:104]

        return np.concatenate([h1_betti_mean, h1_entropy, h1_persistence], axis=1)

    def fit(self, tda_features: np.ndarray) -> 'RegimeLabeler':
        """
        Fit K-means on training TDA features.

        Args:
            tda_features: (N, 214) training TDA features

        Returns:
            self
        """
        # Extract H1 features
        h1_features = self._extract_h1_features(tda_features)

        # Scale features
        self.scaler = StandardScaler()
        h1_scaled = self.scaler.fit_transform(h1_features)

        # Fit K-means
        self.kmeans = KMeans(
            n_clusters=self.n_regimes,
            random_state=self.random_state,
            n_init=10,
        )
        labels = self.kmeans.fit_predict(h1_scaled)

        # Compute regime statistics
        self._compute_regime_stats(h1_features, labels)

        self.is_fitted = True
        return self

    def _compute_regime_stats(self, h1_features: np.ndarray, labels: np.ndarray):
        """Compute statistics for each regime."""
        self.regime_stats = {}

        feature_names = ['h1_betti_mean', 'h1_entropy', 'h1_persistence']

        for regime in range(self.n_regimes):
            mask = labels == regime
            regime_features = h1_features[mask]

            self.regime_stats[regime] = {
                'count': mask.sum(),
                'percentage': 100 * mask.sum() / len(labels),
                'features': {
                    name: {
                        'mean': regime_features[:, i].mean(),
                        'std': regime_features[:, i].std(),
                    }
                    for i, name in enumerate(feature_names)
                }
            }

        # Assign interpretations based on feature values
        self._assign_regime_interpretations()

    def _assign_regime_interpretations(self):
        """Assign descriptive names to regimes based on their characteristics."""
        # Find regime with highest entropy -> "Chaotic"
        # Find regime with highest persistence -> "Cycling"
        # Find regime with highest betti -> "Complex"
        # Remaining -> "Simple"

        entropy_values = [
            self.regime_stats[r]['features']['h1_entropy']['mean']
            for r in range(self.n_regimes)
        ]
        persistence_values = [
            self.regime_stats[r]['features']['h1_persistence']['mean']
            for r in range(self.n_regimes)
        ]
        betti_values = [
            self.regime_stats[r]['features']['h1_betti_mean']['mean']
            for r in range(self.n_regimes)
        ]

        # Default names
        for r in range(self.n_regimes):
            self.regime_stats[r]['name'] = f"Regime_{r}"
            self.regime_stats[r]['interpretation'] = "Unknown"

        # Assign based on dominant characteristic
        chaotic_regime = np.argmax(entropy_values)
        cycling_regime = np.argmax(persistence_values)
        complex_regime = np.argmax(betti_values)

        self.regime_stats[chaotic_regime]['name'] = "Chaotic"
        self.regime_stats[chaotic_regime]['interpretation'] = "High entropy, unpredictable - avoid trading"

        if cycling_regime != chaotic_regime:
            self.regime_stats[cycling_regime]['name'] = "Cycling"
            self.regime_stats[cycling_regime]['interpretation'] = "High persistence, mean-reversion opportunity"

        if complex_regime not in [chaotic_regime, cycling_regime]:
            self.regime_stats[complex_regime]['name'] = "Complex"
            self.regime_stats[complex_regime]['interpretation'] = "High loop count, consolidation"

        # Find Simple (lowest overall activity)
        for r in range(self.n_regimes):
            if self.regime_stats[r]['name'].startswith("Regime_"):
                self.regime_stats[r]['name'] = "Simple"
                self.regime_stats[r]['interpretation'] = "Low complexity, trending market"

    def transform(self, tda_features: np.ndarray) -> np.ndarray:
        """
        Compute regime labels for TDA features.

        Args:
            tda_features: (N, 214) or (214,) TDA features

        Returns:
            (N,) or scalar regime labels
        """
        if not self.is_fitted:
            raise RuntimeError("RegimeLabeler must be fitted before transform")

        single_sample = tda_features.ndim == 1
        if single_sample:
            tda_features = tda_features.reshape(1, -1)

        h1_features = self._extract_h1_features(tda_features)
        h1_scaled = self.scaler.transform(h1_features)
        labels = self.kmeans.predict(h1_scaled)

        if single_sample:
            return labels[0]
        return labels

    def fit_transform(self, tda_features: np.ndarray) -> np.ndarray:
        """Fit and return regime labels."""
        self.fit(tda_features)
        return self.transform(tda_features)

    def get_regime_info(self, regime_id: int) -> dict:
        """Get information about a specific regime."""
        if not self.is_fitted:
            raise RuntimeError("RegimeLabeler must be fitted first")
        return self.regime_stats[regime_id]

    def print_regime_summary(self):
        """Print summary of all regimes."""
        if not self.is_fitted:
            raise RuntimeError("RegimeLabeler must be fitted first")

        print("\n" + "=" * 60)
        print("Regime Summary")
        print("=" * 60)

        for r in range(self.n_regimes):
            stats = self.regime_stats[r]
            print(f"\n{stats['name']} (Regime {r}):")
            print(f"  Count: {stats['count']:,} ({stats['percentage']:.1f}%)")
            print(f"  Interpretation: {stats['interpretation']}")
            print(f"  Features:")
            for name, values in stats['features'].items():
                print(f"    {name}: {values['mean']:.3f} +/- {values['std']:.3f}")

    def save(self, path: str):
        """Save fitted labeler to file."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted labeler")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'n_regimes': self.n_regimes,
                'random_state': self.random_state,
                'kmeans': self.kmeans,
                'scaler': self.scaler,
                'regime_stats': self.regime_stats,
            }, f)

    @classmethod
    def load(cls, path: str) -> 'RegimeLabeler':
        """Load fitted labeler from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        labeler = cls(
            n_regimes=data['n_regimes'],
            random_state=data['random_state'],
        )
        labeler.kmeans = data['kmeans']
        labeler.scaler = data['scaler']
        labeler.regime_stats = data['regime_stats']
        labeler.is_fitted = True
        return labeler


def compute_regime_labels(
    tda_features: np.ndarray,
    n_regimes: int = 4,
    random_state: int = 42,
) -> Tuple[np.ndarray, RegimeLabeler]:
    """
    Convenience function to compute regime labels.

    Args:
        tda_features: (N, 214) TDA features
        n_regimes: Number of regimes
        random_state: Random seed

    Returns:
        Tuple of (labels, fitted RegimeLabeler)
    """
    labeler = RegimeLabeler(n_regimes=n_regimes, random_state=random_state)
    labels = labeler.fit_transform(tda_features)
    return labels, labeler
