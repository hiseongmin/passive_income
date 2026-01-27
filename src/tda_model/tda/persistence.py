"""
Persistent homology computation using Vietoris-Rips complex.

Computes persistence diagrams from point clouds using giotto-tda.
"""

import numpy as np
from typing import List, Tuple, Optional

try:
    from gtda.homology import VietorisRipsPersistence
    GTDA_AVAILABLE = True
except ImportError:
    GTDA_AVAILABLE = False

try:
    import ripser
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False


def compute_persistence_diagram(
    point_cloud: np.ndarray,
    homology_dimensions: List[int] = [0, 1],
    max_edge_length: Optional[float] = None,
    n_jobs: int = 1,
) -> np.ndarray:
    """
    Compute persistence diagram from point cloud using Vietoris-Rips filtration.

    Args:
        point_cloud: Array of shape (n_points, n_dims)
        homology_dimensions: List of homology dimensions to compute (e.g., [0, 1])
        max_edge_length: Maximum edge length for Rips complex. If None, uses diameter.
        n_jobs: Number of parallel jobs for computation

    Returns:
        Persistence diagram array of shape (n_points, 3) where each row is
        [birth, death, homology_dimension]
    """
    if not GTDA_AVAILABLE and not RIPSER_AVAILABLE:
        raise ImportError(
            "Neither giotto-tda nor ripser is available. "
            "Install with: pip install giotto-tda or pip install ripser"
        )

    point_cloud = np.asarray(point_cloud, dtype=np.float64)

    if max_edge_length is None:
        # Use point cloud diameter as max edge length
        from scipy.spatial.distance import pdist
        max_edge_length = pdist(point_cloud).max() if len(point_cloud) > 1 else 1.0

    if GTDA_AVAILABLE:
        return _compute_with_gtda(
            point_cloud, homology_dimensions, max_edge_length, n_jobs
        )
    else:
        return _compute_with_ripser(
            point_cloud, homology_dimensions, max_edge_length
        )


def _compute_with_gtda(
    point_cloud: np.ndarray,
    homology_dimensions: List[int],
    max_edge_length: float,
    n_jobs: int,
) -> np.ndarray:
    """Compute persistence using giotto-tda."""
    # giotto-tda expects 3D input: (n_samples, n_points, n_dims)
    point_cloud_3d = point_cloud[np.newaxis, :, :]

    vr = VietorisRipsPersistence(
        homology_dimensions=homology_dimensions,
        max_edge_length=max_edge_length,
        n_jobs=n_jobs,
    )

    # Returns shape (1, n_features, 3) where 3 = [birth, death, dim]
    diagrams = vr.fit_transform(point_cloud_3d)

    return diagrams[0]  # Return (n_features, 3)


def _compute_with_ripser(
    point_cloud: np.ndarray,
    homology_dimensions: List[int],
    max_edge_length: float,
) -> np.ndarray:
    """Compute persistence using ripser as fallback."""
    max_dim = max(homology_dimensions)

    result = ripser.ripser(
        point_cloud,
        maxdim=max_dim,
        thresh=max_edge_length,
    )

    # Combine diagrams from all dimensions
    all_points = []
    for dim in homology_dimensions:
        dgm = result['dgms'][dim]
        # Add dimension column
        dim_col = np.full((len(dgm), 1), dim)
        dgm_with_dim = np.hstack([dgm, dim_col])
        all_points.append(dgm_with_dim)

    if all_points:
        return np.vstack(all_points)
    else:
        return np.empty((0, 3))


def filter_infinite_deaths(
    diagram: np.ndarray,
    replace_value: Optional[float] = None,
) -> np.ndarray:
    """
    Handle infinite death times in persistence diagrams.

    Args:
        diagram: Persistence diagram array of shape (n_points, 3)
        replace_value: Value to replace inf with. If None, uses max finite death.

    Returns:
        Filtered diagram with finite death times
    """
    diagram = diagram.copy()

    # Find infinite deaths
    inf_mask = np.isinf(diagram[:, 1])

    if inf_mask.any():
        if replace_value is None:
            # Use maximum finite death time
            finite_deaths = diagram[~inf_mask, 1]
            replace_value = finite_deaths.max() if len(finite_deaths) > 0 else 1.0

        diagram[inf_mask, 1] = replace_value

    return diagram


def get_persistence_pairs(
    diagram: np.ndarray,
    homology_dim: int,
) -> np.ndarray:
    """
    Extract (birth, death) pairs for a specific homology dimension.

    Args:
        diagram: Full persistence diagram with dimension column
        homology_dim: Homology dimension to extract

    Returns:
        Array of shape (n_pairs, 2) with [birth, death] pairs
    """
    mask = diagram[:, 2] == homology_dim
    return diagram[mask, :2]


def compute_persistence_from_time_series(
    time_series: np.ndarray,
    embedding_dim: int = 2,
    time_delay: int = 12,
    homology_dimensions: List[int] = [0, 1],
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute persistence diagram directly from time series.

    Convenience function that combines embedding and persistence computation.

    Args:
        time_series: 1D price time series
        embedding_dim: Takens embedding dimension
        time_delay: Time delay for embedding
        homology_dimensions: Homology dimensions to compute
        normalize: Whether to normalize point cloud

    Returns:
        Persistence diagram array
    """
    from .point_cloud import create_point_cloud_from_window

    # Create point cloud from time series
    point_cloud = create_point_cloud_from_window(
        time_series,
        embedding_dim=embedding_dim,
        time_delay=time_delay,
        normalize=normalize,
    )

    # Compute persistence
    diagram = compute_persistence_diagram(
        point_cloud,
        homology_dimensions=homology_dimensions,
    )

    # Handle infinite deaths
    diagram = filter_infinite_deaths(diagram)

    return diagram
