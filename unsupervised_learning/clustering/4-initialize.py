#!/usr/bin/env python3
"""Initialize GMM Module"""


import numpy as np

kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Initializes variables for a Gaussian Mixture Model.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        k: positive integer containing the number of clusters

    Returns:
        pi, m, S, or None, None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None

    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None, None

    n, d = X.shape

    m, _ = kmeans(X, k)
    if m is None:
        return None, None, None

    pi = np.full((k,), 1 / k)
    S = np.tile(np.identity(d), (k, 1, 1))

    return pi, m, S
