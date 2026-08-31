#!/usr/bin/env python3
"""Calculate the cost of t-SNE."""

import numpy as np


def cost(P, Q):
    """Calculate the cost of the t-SNE transformation.

    Args:
        P: numpy.ndarray of shape (n, n), P affinities.
        Q: numpy.ndarray of shape (n, n), Q affinities.

    Returns:
        C: cost of the t-SNE transformation.
    """
    # Prevent log(0)
    P_safe = np.maximum(P, 1e-12)
    Q_safe = np.maximum(Q, 1e-12)

    # KL divergence
    C = np.sum(P_safe * np.log(P_safe / Q_safe))

    return C
