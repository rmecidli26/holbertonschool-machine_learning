#!/usr/bin/env python3
"""
Defines a single neuron performing binary
"""
import numpy as np


class Neuron:
    """
    Represents a single neuron performing binary classification.
    """
    def __init__(self, nx):
        """
        Initializes the neuron.

        Parameters:
        nx (int): The number of input features to the neuron.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for the private weights vector __W"""
        return self.__W

    @property
    def b(self):
        """Getter for the private bias __b"""
        return self.__b

    @property
    def A(self):
        """Getter for the private activated output __A"""
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron.

        Parameters:
        X (numpy.ndarray): The input data with shape (nx, m).
                           nx is the number of input features.
                           m is the number of examples.

        Returns:
        The updated private attribute __A.
        """
        # Linear equation: Z = W * X + b
        # W has shape (1, nx) and X has shape
        Z = np.dot(self.__W, X) + self.__b

        # Sigmoid activation function: A = 1 / (1 + e^(-Z))
        self.__A = 1 / (1 + np.exp(-Z))

        return self.__A
