#!/usr/bin/env python3
"""Optimum K Module"""


import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Tests for the optimum number of clusters by variance.

    Args:
        X: numpy.ndarray of shape (n, d)
        kmin: minimum number of clusters
        kmax: maximum number of clusters
        iterations: maximum number of K-means iterations

    Returns:
        results, d_vars or None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    n = X.shape[0]

    if not isinstance(kmin, int) or kmin <= 0 or kmin >= n:
        return None, None

    if kmax is None:
        kmax = n

    if not isinstance(kmax, int) or kmax <= kmin or kmax > n:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    results = []
    variances = []

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None or clss is None:
            return None, None

        var = variance(X, C)
        if var is None:
            return None, None

        results.append((C, clss))
        variances.append(var)

    d_vars = [variances[0] - var for var in variances]

    return results, d_vars
