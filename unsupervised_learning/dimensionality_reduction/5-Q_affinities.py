#!/usr/bin/env python3
"""Calculate Q affinities for t-SNE."""

import numpy as np


def Q_affinities(Y):
    """Calculate the Q affinities.

    Args:
        Y: numpy.ndarray of shape (n, ndim), low-dimensional data.

    Returns:
        Q: numpy.ndarray of shape (n, n), Q affinities.
        num: numpy.ndarray of shape (n, n), numerator of Q.
    """
    # Squared pairwise Euclidean distances
    Y_squared = np.sum(Y ** 2, axis=1, keepdims=True)
    D = Y_squared + Y_squared.T - 2 * np.matmul(Y, Y.T)

    # Student t-distribution numerator:
    # 1 / (1 + ||y_i - y_j||^2)
    num = 1 / (1 + D)

    # Diagonal must be zero
    np.fill_diagonal(num, 0)

    # Normalize
    Q = num / np.sum(num)

    return Q, num
