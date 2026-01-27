"""
TDA feature extraction from persistence diagrams.

Extracts vectorized features:
- Betti curves
- Persistent entropy
- Total persistence
- Persistence landscape L2 norms
"""

import numpy as np
from typing import List, Optional, Tuple

try:
    from gtda.diagrams import (
        BettiCurve,
        PersistenceEntropy,
        Amplitude,
        PersistenceLandscape,
    )
    GTDA_AVAILABLE = True
except ImportError:
    GTDA_AVAILABLE = False


def compute_betti_curve(
    diagram: np.ndarray,
    homology_dimensions: List[int] = [0, 1],
    n_bins: int = 100,
) -> np.ndarray:
    """
    Compute Betti curve from persistence diagram.

    The Betti curve β_k(t) counts the number of k-dimensional holes
    alive at filtration value t.

    Args:
        diagram: Persistence diagram of shape (n_points, 3)
        homology_dimensions: Homology dimensions to compute
        n_bins: Number of bins for discretization

    Returns:
        Betti curve array of shape (n_bins * n_homology_dims,)
    """
    if GTDA_AVAILABLE:
        return _betti_curve_gtda(diagram, homology_dimensions, n_bins)
    else:
        return _betti_curve_manual(diagram, homology_dimensions, n_bins)


def _betti_curve_gtda(
    diagram: np.ndarray,
    homology_dimensions: List[int],
    n_bins: int,
) -> np.ndarray:
    """Compute Betti curve using giotto-tda."""
    # giotto-tda expects 3D input
    diagram_3d = diagram[np.newaxis, :, :]

    bc = BettiCurve(n_bins=n_bins)
    # Fit to get proper filtration range
    bc.fit(diagram_3d)
    curve = bc.transform(diagram_3d)

    return curve[0].flatten()


def _betti_curve_manual(
    diagram: np.ndarray,
    homology_dimensions: List[int],
    n_bins: int,
) -> np.ndarray:
    """Compute Betti curve manually without giotto-tda."""
    # Get filtration range
    births = diagram[:, 0]
    deaths = diagram[:, 1]

    t_min = births.min() if len(births) > 0 else 0
    t_max = deaths.max() if len(deaths) > 0 else 1

    # Create bins
    t_values = np.linspace(t_min, t_max, n_bins)

    curves = []
    for dim in homology_dimensions:
        mask = diagram[:, 2] == dim
        dim_diagram = diagram[mask, :2]

        curve = np.zeros(n_bins)
        for i, t in enumerate(t_values):
            # Count features alive at time t
            alive = np.sum((dim_diagram[:, 0] <= t) & (dim_diagram[:, 1] > t))
            curve[i] = alive

        curves.append(curve)

    return np.concatenate(curves)


def compute_persistent_entropy(
    diagram: np.ndarray,
    homology_dimensions: List[int] = [0, 1],
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute persistent entropy from persistence diagram.

    Persistent entropy measures the "complexity" of the persistence diagram
    based on the distribution of persistence lifetimes.

    Args:
        diagram: Persistence diagram of shape (n_points, 3)
        homology_dimensions: Homology dimensions to compute
        normalize: Whether to normalize entropy

    Returns:
        Entropy array of shape (n_homology_dims,)
    """
    if GTDA_AVAILABLE:
        return _entropy_gtda(diagram, homology_dimensions, normalize)
    else:
        return _entropy_manual(diagram, homology_dimensions, normalize)


def _entropy_gtda(
    diagram: np.ndarray,
    homology_dimensions: List[int],
    normalize: bool,
) -> np.ndarray:
    """Compute entropy using giotto-tda."""
    diagram_3d = diagram[np.newaxis, :, :]

    pe = PersistenceEntropy(normalize=normalize)
    pe.fit(diagram_3d)
    entropy = pe.transform(diagram_3d)

    return entropy[0]


def _entropy_manual(
    diagram: np.ndarray,
    homology_dimensions: List[int],
    normalize: bool,
) -> np.ndarray:
    """Compute entropy manually without giotto-tda."""
    entropies = []

    for dim in homology_dimensions:
        mask = diagram[:, 2] == dim
        dim_diagram = diagram[mask, :2]

        if len(dim_diagram) == 0:
            entropies.append(0.0)
            continue

        # Compute lifetimes
        lifetimes = dim_diagram[:, 1] - dim_diagram[:, 0]
        lifetimes = lifetimes[lifetimes > 0]

        if len(lifetimes) == 0:
            entropies.append(0.0)
            continue

        # Normalize to get probability distribution
        total = lifetimes.sum()
        probs = lifetimes / total

        # Compute entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        if normalize:
            max_entropy = np.log(len(probs))
            entropy = entropy / max_entropy if max_entropy > 0 else 0

        entropies.append(entropy)

    return np.array(entropies)


def compute_total_persistence(
    diagram: np.ndarray,
    homology_dimensions: List[int] = [0, 1],
    p: float = 2.0,
) -> np.ndarray:
    """
    Compute total persistence (p-norm of lifetimes).

    Total persistence = (Σ |death - birth|^p)^(1/p)

    Args:
        diagram: Persistence diagram of shape (n_points, 3)
        homology_dimensions: Homology dimensions to compute
        p: Power for the norm (default 2 for L2 norm)

    Returns:
        Total persistence array of shape (n_homology_dims,)
    """
    # Use manual implementation for reliability (gtda API varies by version)
    return _amplitude_manual(diagram, homology_dimensions, p)


def _amplitude_gtda(
    diagram: np.ndarray,
    homology_dimensions: List[int],
    p: float,
) -> np.ndarray:
    """Compute amplitude using giotto-tda."""
    diagram_3d = diagram[np.newaxis, :, :]

    # Use wasserstein metric which computes distance to empty diagram
    # This is equivalent to sum of lifetimes raised to power p
    amp = Amplitude(metric="wasserstein", order=p)
    amp.fit(diagram_3d)
    amplitude = amp.transform(diagram_3d)

    return amplitude[0]


def _amplitude_manual(
    diagram: np.ndarray,
    homology_dimensions: List[int],
    p: float,
) -> np.ndarray:
    """Compute amplitude manually without giotto-tda."""
    amplitudes = []

    for dim in homology_dimensions:
        mask = diagram[:, 2] == dim
        dim_diagram = diagram[mask, :2]

        if len(dim_diagram) == 0:
            amplitudes.append(0.0)
            continue

        # Compute lifetimes
        lifetimes = dim_diagram[:, 1] - dim_diagram[:, 0]
        lifetimes = lifetimes[lifetimes > 0]

        if len(lifetimes) == 0:
            amplitudes.append(0.0)
            continue

        # Compute p-norm
        amplitude = np.power(np.sum(np.power(lifetimes, p)), 1/p)
        amplitudes.append(amplitude)

    return np.array(amplitudes)


def compute_landscape_l2_norm(
    diagram: np.ndarray,
    homology_dimensions: List[int] = [0, 1],
    n_layers: int = 5,
    n_bins: int = 100,
) -> np.ndarray:
    """
    Compute L2 norms of persistence landscapes.

    Persistence landscape is a functional summary of persistence diagrams.
    We compute the L2 norm of each landscape layer.

    Args:
        diagram: Persistence diagram of shape (n_points, 3)
        homology_dimensions: Homology dimensions to compute
        n_layers: Number of landscape layers
        n_bins: Number of bins for discretization

    Returns:
        L2 norms array of shape (n_layers * n_homology_dims,)
    """
    # Use manual implementation for reliability (gtda API shape varies by version)
    return _landscape_manual(diagram, homology_dimensions, n_layers, n_bins)


def _landscape_gtda(
    diagram: np.ndarray,
    homology_dimensions: List[int],
    n_layers: int,
    n_bins: int,
) -> np.ndarray:
    """Compute landscape L2 norms using giotto-tda."""
    diagram_3d = diagram[np.newaxis, :, :]

    pl = PersistenceLandscape(n_layers=n_layers, n_bins=n_bins)
    pl.fit(diagram_3d)
    landscapes = pl.transform(diagram_3d)

    # landscapes shape: (1, n_homology_dims, n_layers, n_bins)
    # Compute L2 norm for each layer
    l2_norms = []
    for dim_idx in range(len(homology_dimensions)):
        for layer_idx in range(n_layers):
            landscape = landscapes[0, dim_idx, layer_idx, :]
            l2_norm = np.sqrt(np.sum(landscape ** 2))
            l2_norms.append(l2_norm)

    return np.array(l2_norms)


def _landscape_manual(
    diagram: np.ndarray,
    homology_dimensions: List[int],
    n_layers: int,
    n_bins: int,
) -> np.ndarray:
    """Compute landscape L2 norms manually without giotto-tda."""
    l2_norms = []

    for dim in homology_dimensions:
        mask = diagram[:, 2] == dim
        dim_diagram = diagram[mask, :2]

        if len(dim_diagram) == 0:
            l2_norms.extend([0.0] * n_layers)
            continue

        # Get filtration range
        t_min = dim_diagram[:, 0].min()
        t_max = dim_diagram[:, 1].max()
        t_values = np.linspace(t_min, t_max, n_bins)

        # Compute tent functions for each point
        def tent_function(birth, death, t):
            if t < birth or t > death:
                return 0
            mid = (birth + death) / 2
            if t <= mid:
                return t - birth
            else:
                return death - t

        # For each t, compute all tent function values and sort
        all_values = np.zeros((len(dim_diagram), n_bins))
        for i, (birth, death) in enumerate(dim_diagram):
            for j, t in enumerate(t_values):
                all_values[i, j] = tent_function(birth, death, t)

        # Sort values at each t (descending) to get landscape layers
        sorted_values = -np.sort(-all_values, axis=0)

        # Compute L2 norm for each layer
        for layer_idx in range(n_layers):
            if layer_idx < len(sorted_values):
                landscape = sorted_values[layer_idx, :]
            else:
                landscape = np.zeros(n_bins)

            l2_norm = np.sqrt(np.sum(landscape ** 2))
            l2_norms.append(l2_norm)

    return np.array(l2_norms)


def extract_tda_features(
    time_series: np.ndarray,
    embedding_dim: int = 2,
    time_delay: int = 12,
    betti_bins: int = 100,
    landscape_layers: int = 5,
    homology_dimensions: List[int] = [0, 1],
) -> np.ndarray:
    """
    Extract all TDA features from a time series window.

    This is the main entry point for TDA feature extraction.

    Features extracted:
    - Betti curve: betti_bins * n_homology_dims
    - Persistent entropy: n_homology_dims
    - Total persistence: n_homology_dims
    - Landscape L2 norms: landscape_layers * n_homology_dims

    Args:
        time_series: 1D price time series window
        embedding_dim: Takens embedding dimension
        time_delay: Time delay for embedding
        betti_bins: Number of bins for Betti curve
        landscape_layers: Number of persistence landscape layers
        homology_dimensions: Homology dimensions to compute

    Returns:
        Feature vector of shape (total_dims,)
        where total_dims = betti_bins * n_hom + n_hom + n_hom + landscape_layers * n_hom
    """
    from .persistence import compute_persistence_from_time_series

    # Compute persistence diagram
    diagram = compute_persistence_from_time_series(
        time_series,
        embedding_dim=embedding_dim,
        time_delay=time_delay,
        homology_dimensions=homology_dimensions,
        normalize=True,
    )

    # Extract features
    betti = compute_betti_curve(diagram, homology_dimensions, betti_bins)
    entropy = compute_persistent_entropy(diagram, homology_dimensions)
    persistence = compute_total_persistence(diagram, homology_dimensions)
    landscape = compute_landscape_l2_norm(
        diagram, homology_dimensions, landscape_layers, betti_bins
    )

    # Concatenate all features
    features = np.concatenate([betti, entropy, persistence, landscape])

    # Handle NaN/Inf values
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features.astype(np.float32)


def get_feature_dimensions(
    betti_bins: int = 100,
    landscape_layers: int = 5,
    homology_dimensions: List[int] = [0, 1],
) -> Tuple[int, dict]:
    """
    Get total feature dimension and breakdown.

    Args:
        betti_bins: Number of bins for Betti curve
        landscape_layers: Number of persistence landscape layers
        homology_dimensions: Homology dimensions

    Returns:
        Tuple of (total_dim, breakdown_dict)
    """
    n_hom = len(homology_dimensions)

    breakdown = {
        "betti_curve": betti_bins * n_hom,
        "persistent_entropy": n_hom,
        "total_persistence": n_hom,
        "landscape_l2_norm": landscape_layers * n_hom,
    }

    total_dim = sum(breakdown.values())

    return total_dim, breakdown
