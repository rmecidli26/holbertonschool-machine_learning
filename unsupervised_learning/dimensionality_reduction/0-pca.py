#!/usr/bin/env python3
"""Performs Principal Component Analysis (PCA) on a dataset."""


import numpy as np


def pca(X, var=0.95):
    """Perform PCA on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d)
        var: fraction of the variance to maintain

    Returns:
        W: numpy.ndarray of shape (d, nd) containing the
        weights that preserve the requested variance.
    """
    U, S, Vh = np.linalg.svd(X)

    total = np.sum(S)
    ratio = 0
    i = 0

    while ratio < var:
        ratio += S[i] / total
        i += 1

    return Vh[:i].T
