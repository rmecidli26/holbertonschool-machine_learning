#!/usr/bin/env python3
"""Module containing the RNNCell class."""
import numpy as np


class RNNCell:
    """Represents a cell of a simple RNN."""

    def __init__(self, i, h, o):
        """
        Initialize the RNN cell.

        Parameters:
            i (int): Dimensionality of the input data
            h (int): Dimensionality of the hidden state
            o (int): Dimensionality of the outputs
        """
        # Weights initialized using standard normal distribution
        # Concatenated weights for hidden state (h) and input data (i)
        self.Wh = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)

        # Biases initialized to zeros
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Perform forward propagation for one time step.

        Parameters:
            h_prev (numpy.ndarray): Shape (m, h), previous hidden state
            x_t (numpy.ndarray): Shape (m, i), input data for the cell

        Returns:
            h_next (numpy.ndarray): Next hidden state
            y (numpy.ndarray): Output of the cell
        """
        # Concatenate previous hidden state and
        # Shape becomes (m, h + i)
        concat_input = np.concatenate((h_prev, x_t), axis=1)

        # Calculate next hidden state using tanh activ
        h_next = np.tanh(np.matmul(concat_input, self.Wh) + self.bh)

        # Calculate unnormalized output scores (logits)
        logits = np.matmul(h_next, self.Wy) + self.by

        # Softmax activation for final output probabilities
        # exp_logits / sum(exp_logits) along rows (axis 1)
        exp_logits = np.exp(logits)
        y = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return h_next, y
