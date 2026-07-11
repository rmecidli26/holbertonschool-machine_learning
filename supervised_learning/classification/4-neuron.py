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

        Returns:
        The updated private attribute __A.
        """
        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.

        Parameters:
        Y (numpy.ndarray): Correct labels for
        A (numpy.ndarray): Activated output

        Returns:
        The cost as a float.
        """
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions.

        Parameters:
        X (numpy.ndarray): Input data, shape (nx, m).
        Y (numpy.ndarray): Correct labels for

        Returns:
        prediction (numpy.ndarray): Predicted binary
        cost (float): The network's cost.
        """
        # Step 1: Compute activated outputs via forward propagation
        A = self.forward_prop(X)
        
        # Step 2: Calculate the network cost
        net_cost = self.cost(Y, A)
        
        # Step 3: Threshold activations to find binary
        # .astype(int) converts the boolean matrix entries to binary ints
        prediction = np.where(A >= 0.5, 1, 0)
        
        return prediction, net_cost