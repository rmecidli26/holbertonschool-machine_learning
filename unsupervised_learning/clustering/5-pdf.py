#!/usr/bin/env python3
"""Probability Density Function (PDF) Module"""


import numpy as np


def pdf(X, m, S):
    """Calculates the probability density function of a Gaussian distribution.

    Args:
        X: numpy.ndarray of shape (n, d) containing data points
        m: numpy.ndarray of shape (d,) containing mean of the distribution
        S: numpy.ndarray of shape (d, d) containing covariance matrix

    Returns:
        P: numpy.ndarray of shape (n,) containing PDF values for each point
        or None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None

    n, d = X.shape

    if m.shape[0] != d or S.shape[0] != d or S.shape[1] != d:
        return None

    det_S = np.linalg.det(S)
    if det_S <= 0:
        return None

    inv_S = np.linalg.inv(S)
    diff = X - m

    # Mahalanobis distance calculation without loops
    mahalanobis = np.sum(np.dot(diff, inv_S) * diff, axis=1)

    norm_const = 1.0 / np.sqrt(((2 * np.pi) ** d) * det_S)
    P = norm_const * np.exp(-0.5 * mahalanobis)

    P = np.maximum(P, 1e-300)

    return P
