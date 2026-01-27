"""
Takens embedding for time series to point cloud conversion.

Implements delay embedding to transform 1D time series into
point clouds in R^d for topological analysis.
"""

import numpy as np
from typing import Optional


def takens_embedding(
    time_series: np.ndarray,
    embedding_dim: int = 2,
    time_delay: int = 12,
    stride: int = 1,
) -> np.ndarray:
    """
    Apply Takens delay embedding to convert time series to point cloud.

    Given a time series x(t), creates delay vectors:
    [x(t), x(t-τ), x(t-2τ), ..., x(t-(d-1)τ)]

    Args:
        time_series: 1D array of shape (n_points,)
        embedding_dim: Embedding dimension d (default 2, optimal per enhance.pdf)
        time_delay: Time delay τ between coordinates (default 12 = 3 hours at 15-min)
        stride: Step size between consecutive delay vectors

    Returns:
        Point cloud array of shape (n_vectors, embedding_dim)
        where n_vectors = (n_points - (embedding_dim - 1) * time_delay) // stride
    """
    time_series = np.asarray(time_series).flatten()
    n_points = len(time_series)

    # Calculate number of delay vectors we can create
    required_length = (embedding_dim - 1) * time_delay + 1
    if n_points < required_length:
        raise ValueError(
            f"Time series length {n_points} is too short for "
            f"embedding_dim={embedding_dim}, time_delay={time_delay}. "
            f"Need at least {required_length} points."
        )

    # Number of delay vectors
    n_vectors = (n_points - (embedding_dim - 1) * time_delay - 1) // stride + 1

    # Create delay embedding
    point_cloud = np.zeros((n_vectors, embedding_dim), dtype=np.float64)

    for i in range(n_vectors):
        start_idx = i * stride
        for d in range(embedding_dim):
            point_cloud[i, d] = time_series[start_idx + d * time_delay]

    return point_cloud


def normalize_point_cloud(
    point_cloud: np.ndarray,
    method: str = "minmax",
) -> np.ndarray:
    """
    Normalize point cloud for stable persistence computation.

    Args:
        point_cloud: Array of shape (n_points, n_dims)
        method: Normalization method ('minmax', 'zscore', 'none')

    Returns:
        Normalized point cloud
    """
    if method == "none":
        return point_cloud

    point_cloud = point_cloud.copy()

    if method == "minmax":
        # Scale to [0, 1] range
        min_vals = point_cloud.min(axis=0, keepdims=True)
        max_vals = point_cloud.max(axis=0, keepdims=True)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0  # Avoid division by zero
        point_cloud = (point_cloud - min_vals) / range_vals

    elif method == "zscore":
        # Standardize to zero mean, unit variance
        mean = point_cloud.mean(axis=0, keepdims=True)
        std = point_cloud.std(axis=0, keepdims=True)
        std[std == 0] = 1.0  # Avoid division by zero
        point_cloud = (point_cloud - mean) / std

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return point_cloud


def create_point_cloud_from_window(
    window: np.ndarray,
    embedding_dim: int = 2,
    time_delay: int = 12,
    normalize: bool = True,
) -> np.ndarray:
    """
    Create normalized point cloud from a price window.

    Convenience function that combines embedding and normalization.

    Args:
        window: 1D price time series window
        embedding_dim: Takens embedding dimension
        time_delay: Time delay for embedding
        normalize: Whether to apply min-max normalization

    Returns:
        Normalized point cloud array
    """
    # Apply Takens embedding
    point_cloud = takens_embedding(
        window,
        embedding_dim=embedding_dim,
        time_delay=time_delay,
        stride=1,
    )

    # Normalize for stable persistence computation
    if normalize:
        point_cloud = normalize_point_cloud(point_cloud, method="minmax")

    return point_cloud
