#!/usr/bin/env python3
"""
Defines a single neuron performing binary classification
"""
import numpy as np


class Neuron:
    """
    Represents a single neuron performing
    """
    def __init__(self, nx):
        """
        Initializes the neuron.

        Parameters:
        nx (int): The number of input features to the neuron.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Weights vector initial
        # Shape is (1, nx) to match thected shape (1, 784)
        self.W = np.random.normal(size=(1, nx))

        # Bias initialized to 0
        self.b = 0

        # Activated outpu ed to 0
        self.A = 0
