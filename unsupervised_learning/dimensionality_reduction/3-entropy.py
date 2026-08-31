#!/usr/bin/env python3
"""Calculate Shannon entropy and P affinities for t-SNE."""

import numpy as np


def HP(Di, beta):
    """Calculate Shannon entropy and P affinities.

    Args:
        Di: numpy.ndarray of shape (n - 1,), pairwise squared distances
            from one point to all other points.
        beta: numpy.ndarray of shape (1,), beta value.

    Returns:
        Hi: Shannon entropy.
        Pi: numpy.ndarray of shape (n - 1,), P affinities.
    """
    # Gaussian similarities
    Pi = np.exp(-Di * beta)

    # Sum of probabilities
    sum_Pi = np.sum(Pi)

    # Prevent division by zero
    if sum_Pi == 0:
        return 0, np.zeros(Di.shape)

    # Normalize probabilities
    Pi = Pi / sum_Pi

    # Shannon entropy with base 2
    Hi = -np.sum(Pi * np.log2(Pi))

    return Hi, Pi
