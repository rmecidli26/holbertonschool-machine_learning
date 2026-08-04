#!/usr/bin/env python3
"""Contains the l2_reg_cost function for L2 regularization."""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Calculates 

    Args:
        L: Number of layers in the neural network.
        m: Number of data points used.

    Returns:
        The total cost accounting for L2 regularization.
    """
    f_norm_sum = 0

    for i in range(1, L + 1):
        f_norm_sum += np.linalg.norm(weights[f'W{i}']) ** 2

    l2_cost = cost + (lambtha / (2 * m)) * f_norm_sum
    return l2_cost
