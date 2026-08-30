#!/usr/bin/env python3
"""Module to perform forward propagation for a simple RNN."""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """Performs forward propagation for a simple RNN over t time steps."""
    t, m, i = X.shape
    _, h = h_0.shape
    o = rnn_cell.Wy.shape[1]

    # Initialize hidden states array with shape (t + 1, m, h)
    H = np.zeros((t + 1, m, h))
    H[0] = h_0

    # Initialize outputs array with shape (t, m, o)
    Y = np.zeros((t, m, o))

    # Iterate through all time steps
    h_prev = h_0
    for step in range(t):
        x_t = X[step]
        h_prev, y = rnn_cell.forward(h_prev, x_t)
        H[step + 1] = h_prev
        Y[step] = y

    return H, Y
