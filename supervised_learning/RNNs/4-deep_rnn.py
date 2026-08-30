#!/usr/bin/env python3
"""
Deep RNN module
"""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Performs forward propagation for a deep RNN."""
    t, m, i = X.shape
    l, _, h = h_0.shape

    # H shape: (t + 1, l, m, h)
    H = np.zeros((t + 1, l, m, h))
    H[0] = h_0

    outputs = []

    for step in range(t):
        x_current = X[step]
        for layer in range(l):
            cell = rnn_cells[layer]
            h_prev = H[step, layer]
            # For the first layer,e of the layer below
            if layer == 0:
                h_next, y = cell.forward(h_prev, x_current)
            else:
                h_next, y = cell.forward(h_prev, H[step + 1, layer - 1])
            H[step + 1, layer] = h_next
            if layer == l - 1:
                outputs.append(y)

    Y = np.array(outputs)
    return H, Y
