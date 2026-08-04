#!/usr/bin/env python3
"""Updates weights and biases"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Updates network parameters

    Args:
        Y: One-hot array of shape (classes, m).
        alpha: Learning rate.
        lambtha: L2 regularization parameter.
        L: Number of layers in the neural network.
    """
    m = Y.shape[1]
    dz = cache[f'A{L}'] - Y

    for l in range(L, 0, -1):
        a_prev = cache[f'A{l-1}']
        w_current = weights[f'W{l}']

        # Compute gradients including L2 regularization penalty for weights
        dw = (1 / m) * np.matmul(dz, a_prev.T) + (lambtha / m) * w_current
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

        if l > 1:
            # Backpropagate through tanh activation: g'(z) = 1 - A^2
            dz = np.matmul(w_current.T, dz) * (1 - (a_prev ** 2))

        # Update weights and biases in place
        weights[f'W{l}'] -= alpha * dw
        weights[f'b{l}'] -= alpha * db
