#!/usr/bin/env python3
"""PCA v2."""

import numpy as np


def pca(X, ndim):
    """Perform PCA on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d)
        ndim: new dimensionality

    Returns:
        T: numpy.ndarray of shape (n, ndim)
    """
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    W = Vt[:ndim].T

    T = np.matmul(X_centered, W)

    return T
