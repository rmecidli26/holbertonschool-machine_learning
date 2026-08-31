#!/usr/bin/env python3
"""Initialize K-means."""

import numpy as np


def initialize(X, k):
    """Initialize cluster centroids for K-means.

    Args:
        X: numpy.ndarray of shape (n, d), dataset.
        k: positive integer, number of clusters.

    Returns:
        numpy.ndarray of shape (k, d) containing initialized centroids,
        or None on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None

    if not isinstance(k, int) or k <= 0:
        return None

    if X.shape[0] == 0:
        return None

    min_values = np.min(X, axis=0)
    max_values = np.max(X, axis=0)

    return np.random.uniform(
        min_values,
        max_values,
        (k, X.shape[1])
    )
