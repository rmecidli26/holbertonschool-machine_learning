#!/usr/bin/env python3
"""Defines the GRUCell class."""

import numpy as np


class GRUCell:
    """Represents a gated recurrent unit cell."""

    def __init__(self, i, h, o):
        """Initialize the GRU cell weights and biases.

        Parameters:
            i (int): Dimensionality of the input data
            h (int): Dimensionality of the hidden state
            o (int): Dimensionality of the outputs
        """
        # Weights initialized using random normal distribution
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        # Biases initialized as zeros
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Performs forward propagation for one time step.

        Parameters:
            h_prev (numpy.ndarray): Previous hidden state of shape (m, h)
            x_t (numpy.ndarray): Input data for the cell of shape (m, i)

        Returns:
            h_next (numpy.ndarray): Next hidden state
            y (numpy.ndarray): Output of the cell
        """
        # Concatenate h_prev and x_t (Hidden state must come first)
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Update gate: z_t = sigmoid(concat @ Wz + bz)
        z_t = 1 / (1 + np.exp(-(concat @ self.Wz + self.bz)))

        # Reset gate: r_t = sigmoid(concat @ Wr + br)
        r_t = 1 / (1 + np.exp(-(concat @ self.Wr + self.br)))

        # Candidate hidden state: h_tilde = tanh([r_t * h_prev, x_t] @ Wh + bh)
        concat_reset = np.concatenate((r_t * h_prev, x_t), axis=1)
        h_tilde = np.tanh(concat_reset @ self.Wh + self.bh)

        # Next hidden state: h_next = (1 - z_t) * h_prev + z_t * h_tilde
        h_next = (1 - z_t) * h_prev + z_t * h_tilde

        # Output with softmax activation: y = softmax(h_next @ Wy + by)
        y_linear = h_next @ self.Wy + self.by
        y = np.exp(y_linear) / np.sum(np.exp(y_linear), axis=1, keepdims=True)

        return h_next, y
