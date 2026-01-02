"""
TDA (Topological Data Analysis) Feature Extraction

Based on: "Enhancing financial time series forecasting through topological data analysis"
https://link.springer.com/article/10.1007/s00521-024-10787-x

Extracts topological features from time series using:
- Takens time delay embedding
- Persistence diagrams via Vietoris-Rips complex
- Statistical features from persistence diagrams
"""

import numpy as np
from typing import Tuple, List, Optional
from tqdm import tqdm

try:
    from ripser import ripser
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False
    print("Warning: ripser not installed. TDA features will be zeros.")


def time_delay_embedding(x: np.ndarray, dim: int = 3, tau: int = 1) -> np.ndarray:
    """
    Takens' time delay embedding.

    Converts 1D time series to point cloud in higher dimension.

    Args:
        x: 1D time series
        dim: Embedding dimension
        tau: Time delay

    Returns:
        Point cloud of shape (n_points, dim)
    """
    n = len(x) - (dim - 1) * tau
    if n <= 0:
        raise ValueError(f"Time series too short for embedding: len={len(x)}, dim={dim}, tau={tau}")

    embedded = np.zeros((n, dim))
    for i in range(dim):
        embedded[:, i] = x[i * tau: i * tau + n]

    return embedded


def compute_persistence_diagram(point_cloud: np.ndarray, max_dim: int = 1) -> List[np.ndarray]:
    """
    Compute persistence diagram using Vietoris-Rips complex.

    Args:
        point_cloud: Point cloud from time delay embedding
        max_dim: Maximum homology dimension

    Returns:
        List of persistence diagrams for each dimension
    """
    if not RIPSER_AVAILABLE:
        return [np.array([[0, 0]]), np.array([[0, 0]])]

    result = ripser(point_cloud, maxdim=max_dim)
    return result['dgms']


def persistence_entropy(diagram: np.ndarray) -> float:
    """
    Compute persistent entropy from persistence diagram.

    Measures complexity and irregularity of the time series.

    Args:
        diagram: Persistence diagram (birth, death) pairs

    Returns:
        Entropy value
    """
    # Filter out infinite death times
    finite_mask = np.isfinite(diagram[:, 1])
    diagram = diagram[finite_mask]

    if len(diagram) == 0:
        return 0.0

    # Compute lifetimes
    lifetimes = diagram[:, 1] - diagram[:, 0]
    lifetimes = lifetimes[lifetimes > 0]

    if len(lifetimes) == 0:
        return 0.0

    # Normalize to get probabilities
    total_lifetime = np.sum(lifetimes)
    if total_lifetime == 0:
        return 0.0

    probs = lifetimes / total_lifetime

    # Shannon entropy
    entropy = -np.sum(probs * np.log(probs + 1e-10))

    return float(entropy)


def persistence_amplitude(diagram: np.ndarray) -> float:
    """
    Compute amplitude (spread) of persistence diagram.

    Indicates stability and recurrence of patterns.

    Args:
        diagram: Persistence diagram

    Returns:
        Amplitude value (max lifetime)
    """
    finite_mask = np.isfinite(diagram[:, 1])
    diagram = diagram[finite_mask]

    if len(diagram) == 0:
        return 0.0

    lifetimes = diagram[:, 1] - diagram[:, 0]
    return float(np.max(lifetimes)) if len(lifetimes) > 0 else 0.0


def persistence_num_points(diagram: np.ndarray, threshold: float = 0.0) -> int:
    """
    Count significant topological features.

    Indicates persistent structures (cycles, trends).

    Args:
        diagram: Persistence diagram
        threshold: Minimum lifetime to be considered significant

    Returns:
        Number of significant points
    """
    finite_mask = np.isfinite(diagram[:, 1])
    diagram = diagram[finite_mask]

    if len(diagram) == 0:
        return 0

    lifetimes = diagram[:, 1] - diagram[:, 0]
    return int(np.sum(lifetimes > threshold))


def extract_tda_features(
    window: np.ndarray,
    embedding_dim: int = 3,
    embedding_tau: int = 1,
    homology_dim: int = 1
) -> Tuple[float, float, int]:
    """
    Extract TDA features from a time series window.

    Args:
        window: Time series window (1D array)
        embedding_dim: Takens embedding dimension
        embedding_tau: Takens time delay
        homology_dim: Maximum homology dimension

    Returns:
        Tuple of (entropy, amplitude, num_points)
    """
    # Normalize window
    window = (window - np.mean(window)) / (np.std(window) + 1e-10)

    try:
        # Time delay embedding
        point_cloud = time_delay_embedding(window, dim=embedding_dim, tau=embedding_tau)

        # Compute persistence diagram
        diagrams = compute_persistence_diagram(point_cloud, max_dim=homology_dim)

        # Extract features from H1 (1-dimensional holes/loops)
        h1_diagram = diagrams[1] if len(diagrams) > 1 else diagrams[0]

        entropy = persistence_entropy(h1_diagram)
        amplitude = persistence_amplitude(h1_diagram)
        num_points = persistence_num_points(h1_diagram)

        return entropy, amplitude, num_points

    except Exception as e:
        return 0.0, 0.0, 0


class TDAFeatureExtractor:
    """
    TDA Feature Extractor for time series.

    Extracts entropy, amplitude, and number of points from persistence diagrams
    using multiple embedding configurations.
    """

    def __init__(
        self,
        window_size: int = 72,
        embedding_configs: List[Tuple[int, int]] = None,
        homology_dim: int = 1
    ):
        """
        Args:
            window_size: Size of the sliding window
            embedding_configs: List of (dim, tau) tuples for multiple embeddings
            homology_dim: Maximum homology dimension
        """
        self.window_size = window_size
        self.homology_dim = homology_dim

        if embedding_configs is None:
            # Default: 3 different configurations
            self.embedding_configs = [
                (3, 1),   # Short-term patterns
                (5, 2),   # Medium-term patterns
                (7, 3),   # Long-term patterns
            ]
        else:
            self.embedding_configs = embedding_configs

        self.n_features = len(self.embedding_configs) * 3  # 3 features per config

    def extract_single(self, window: np.ndarray) -> np.ndarray:
        """
        Extract TDA features from a single window.

        Args:
            window: Time series window of shape (window_size,)

        Returns:
            Feature vector of shape (n_features,)
        """
        features = []

        for dim, tau in self.embedding_configs:
            entropy, amplitude, num_points = extract_tda_features(
                window,
                embedding_dim=dim,
                embedding_tau=tau,
                homology_dim=self.homology_dim
            )
            features.extend([entropy, amplitude, num_points])

        return np.array(features, dtype=np.float32)

    def extract_all(
        self,
        series: np.ndarray,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Extract TDA features for entire series using sliding windows.

        Args:
            series: Full time series
            show_progress: Whether to show progress bar

        Returns:
            Array of shape (n_samples, n_features)
        """
        n_samples = len(series)
        features = np.zeros((n_samples, self.n_features), dtype=np.float32)

        iterator = range(self.window_size, n_samples)
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting TDA features")

        for i in iterator:
            window = series[i - self.window_size:i]
            features[i] = self.extract_single(window)

        return features


def extract_tda_features_for_dataset(
    df,
    price_col: str = 'close',
    window_size: int = 72,
    embedding_configs: List[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Extract TDA features for an entire dataset.

    Args:
        df: DataFrame with price data
        price_col: Column name for price data
        window_size: Window size for TDA extraction
        embedding_configs: List of (dim, tau) configurations

    Returns:
        Array of shape (n_samples, n_features)
    """
    extractor = TDAFeatureExtractor(
        window_size=window_size,
        embedding_configs=embedding_configs
    )

    series = df[price_col].values
    features = extractor.extract_all(series)

    return features


if __name__ == "__main__":
    # Test TDA feature extraction
    np.random.seed(42)

    # Create synthetic data
    t = np.linspace(0, 10 * np.pi, 500)
    test_series = np.sin(t) + 0.1 * np.random.randn(500)

    print("Testing TDA feature extraction...")

    extractor = TDAFeatureExtractor(window_size=50)
    features = extractor.extract_all(test_series, show_progress=True)

    print(f"\nExtracted features shape: {features.shape}")
    print(f"Feature dimensions: {extractor.n_features}")
    print(f"\nSample features (last 5 windows):")
    print(features[-5:])

    # Test single extraction
    window = test_series[-50:]
    single_features = extractor.extract_single(window)
    print(f"\nSingle window features: {single_features}")
