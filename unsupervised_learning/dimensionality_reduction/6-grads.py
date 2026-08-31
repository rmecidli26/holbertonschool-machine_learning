#!/usr/bin/env python3
"""Calculate the gradients for t-SNE."""

import numpy as np

Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """Calculate the gradients of Y.

    Args:
        Y: numpy.ndarray of shape (n, ndim)
        P: numpy.ndarray of shape (n, n), P affinities

    Returns:
        dY: numpy.ndarray of shape (n, ndim), gradients of Y
        Q: numpy.ndarray of shape (n, n), Q affinities
    """
    Q, num = Q_affinities(Y)

    diff = Y[:, np.newaxis, :] - Y[np.newaxis, :, :]

    dY = np.sum(
        (P - Q)[:, :, np.newaxis]
        * num[:, :, np.newaxis]
        * diff,
        axis=1
    )

    return dY, Q
