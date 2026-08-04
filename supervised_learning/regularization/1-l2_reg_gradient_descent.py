#!/usr/bin/env python3
"""Updates weights and biases"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Updates network parameters

    Args:
        lambtha: L2 regularization parameter.
        L: Number of layers in the neural network.
    """
    m = Y.shape[1]
    dz = cache[f'A{L}'] - Y

    for i in range(L, 0, -1):
        a_prev = cache[f'A{i-1}']
        w_current = weights[f'W{i}']

        # Compute gradients with L2 regularization
        dw = (1 / m) * np.matmul(dz, a_prev.T) + (lambtha / m) * w_current
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

        if i > 1:
            dz = np.matmul(w_current.T, dz) * (1 - (a_prev ** 2))

        # Update parameters in place
        weights[f'W{i}'] -= alpha * dw
        weights[f'b{i}'] -= alpha * db
