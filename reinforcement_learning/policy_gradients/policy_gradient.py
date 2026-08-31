#!/usr/bin/env python3
"""Module to compute the policy with a weight matrix."""

import numpy as np


def policy(matrix, weight):
    """Computes the policy (action probabilities) using Softmax.

    Args:
        matrix: numpy.ndarray of shape (1, n) representing the state.
        weight: numpy.ndarray of shape (n, m) representing

    Returns:
        numpy.ndarray of shape (1, m) containing the action probabilities.
    """
    z = np.matmul(matrix, weight)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
