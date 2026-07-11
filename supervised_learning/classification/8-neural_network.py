#!/usr/bin/env python3
"""Defines a neural network with one hidden layer"""
import numpy as np


class NeuralNetwork:
    """Represents a neural network with one hidden layer."""

    def __init__(self, nx, nodes):
        """Initializes the neural network."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Hidden layer attributes (Layer 1)
        self.W1 = np.random.normal(size=(nodes, nx))
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0

        # Output neuron attributes (Layer 2)
        self.W2 = np.random.normal(size=(1, nodes))
        self.b2 = 0
        self.A2 = 0
