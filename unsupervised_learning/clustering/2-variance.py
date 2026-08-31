#!/usr/bin/env python3
"""Calculate the total intra-cluster variance."""

import numpy as np


def variance(X, C):
    """Calculate the total intra-cluster variance.

    Args:
        X: numpy.ndarray of shape (n, d), dataset.
        C: numpy.ndarray of shape (k, d), centroids.

    Returns:
        The total variance, or None on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None

    if not isinstance(C, np.ndarray) or C.ndim != 2:
        return None

    if X.shape[0] == 0 or X.shape[1] == 0:
        return None

    if C.shape[0] == 0 or C.shape[1] != X.shape[1]:
        return None

    # Calculate squared distances between every point and centroid
    distances = np.sum(
        (X[:, np.newaxis, :] - C[np.newaxis, :, :]) ** 2,
        axis=2
    )

    # For each point, use the distance to its closest centroid
    min_distances = np.min(distances, axis=1)

    # Total intra-cluster variance
    return np.sum(min_distances)
