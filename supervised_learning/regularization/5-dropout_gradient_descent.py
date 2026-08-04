#!/usr/bin/env python3
"""Updates network parameters"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Updates weights and biases in place"""
    m = Y.shape[1]
    dz = cache[f'A{L}'] - Y

    for i in range(L, 0, -1):
        a_prev = cache[f'A{i-1}']
        w_current = weights[f'W{i}']

        # Compute gradients
        dw = (1 / m) * np.matmul(dz, a_prev.T)
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

        if i > 1:
            # Backpropagate through tanh and apply dropout mask + scaling
            da = np.matmul(w_current.T, dz)
            da = (da * cache[f'D{i-1}']) / keep_prob
            dz = da * (1 - (a_prev ** 2))

        # Update weights and biases in place
        weights[f'W{i}'] -= alpha * dw
        weights[f'b{i}'] -= alpha * db
