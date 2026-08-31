#!/usr/bin/env python3
"""K-means clustering."""

import numpy as np


def kmeans(X, k, iterations=1000):
    """Perform K-means clustering on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d), dataset.
        k: positive integer, number of clusters.
        iterations: positive integer, maximum number of iterations.

    Returns:
        C: numpy.ndarray of shape (k, d), centroids.
        clss: numpy.ndarray of shape (n,), cluster assignments.
        Or (None, None) on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None

    if X.shape[0] == 0 or X.shape[1] == 0:
        return None, None

    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    # Initialize centroids
    C = np.random.uniform(
        np.min(X, axis=0),
        np.max(X, axis=0),
        (k, d)
    )

    for _ in range(iterations):
        # Calculate squared distances from every point to every centroid
        distances = np.sum(
            (X[:, np.newaxis, :] - C[np.newaxis, :, :]) ** 2,
            axis=2
        )

        # Assign each point to its closest centroid
        clss = np.argmin(distances, axis=1)

        # Calculate new centroids
        C_new = np.zeros((k, d))

        for j in range(k):
            points = X[clss == j]

            if points.shape[0] == 0:
                # Reinitialize empty centroid
                C_new[j] = np.random.uniform(
                    np.min(X, axis=0),
                    np.max(X, axis=0)
                )
            else:
                C_new[j] = np.mean(points, axis=0)

        # Stop if centroids did not change
        if np.array_equal(C, C_new):
            break

        C = C_new

    # Recalculate final classes using final centroids
    distances = np.sum(
        (X[:, np.newaxis, :] - C[np.newaxis, :, :]) ** 2,
        axis=2
    )
    clss = np.argmin(distances, axis=1)

    return C, clss
