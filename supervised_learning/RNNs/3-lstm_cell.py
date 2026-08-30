#!/usr/bin/env python3
"""
LSTMCell module
"""

import numpy as np


class LSTMCell:
    """Represents an LSTM unit."""

    def __init__(self, i, h, o):
        """Class constructor for LSTMCell."""
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """Performs forward propagation for one time step."""
        concat_hx = np.concatenate((h_prev, x_t), axis=1)

        f = 1 / (1 + np.exp(-(np.matmul(concat_hx, self.Wf) + self.bf)))
        u = 1 / (1 + np.exp(-(np.matmul(concat_hx, self.Wu) + self.bu)))
        c_bar = np.tanh(np.matmul(concat_hx, self.Wc) + self.bc)

        c_next = f * c_prev + u * c_bar

        o = 1 / (1 + np.exp(-(np.matmul(concat_hx, self.Wo) + self.bo)))
        h_next = o * np.tanh(c_next)

        y_raw = np.matmul(h_next, self.Wy) + self.by
        y_exp = np.exp(y_raw - np.max(y_raw, axis=1, keepdims=True))
        y = y_exp / np.sum(y_exp, axis=1, keepdims=True)

        return h_next, c_next, y
