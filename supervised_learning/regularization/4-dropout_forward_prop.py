#!/usr/bin/env python3
"""Conducts forward propagation using Dropout"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Forward propagation with inverted dropout"""
    cache = {'A0': X}

    for i in range(1, L + 1):
        W = weights[f'W{i}']
        b = weights[f'b{i}']
        A_prev = cache[f'A{i-1}']
        Z = np.matmul(W, A_prev) + b

        if i == L:
            # Softmax activation for the last layer
            exp_Z = np.exp(Z)
            cache[f'A{i}'] = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        else:
            # Tanh activation for hidden layers
            A = np.tanh(Z)
            # Create dropout mask (1 if < keep_prob, else 0)
            D = np.random.rand(*A.shape) < keep_prob
            # Apply mask and scale (Inverted Dropout)
            A = (A * D) / keep_prob
            cache[f'D{i}'] = D.astype(int)
            cache[f'A{i}'] = A

    return cache
