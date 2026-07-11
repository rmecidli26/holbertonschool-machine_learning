#!/usr/bin/env python3
"""
Defines a single neuron performing binary classification with gradient descent
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
        Y (numpy.ndarray): Correct labels for the input data, shape (1, m).
        A (numpy.ndarray): Activated output of the neuron, shape (1, m).

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
        Y (numpy.ndarray): Correct labels for the input data, shape (1, m).

        Returns:
        prediction (numpy.ndarray), cost (float)
        """
        A = self.forward_prop(X)
        net_cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, net_cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron.

        Parameters:
        X (numpy.ndarray): Input data with shape (nx, m).
        Y (numpy.ndarray): Correct labels for the input data, shape (1, m).
        A (numpy.ndarray): Activated output of the neuron, shape (1, m).
        alpha (float): Learning rate.
        """
        m = Y.shape[1]

        # Calculate the error/residual difference
        dZ = A - Y

        # Calculate gradients
        # dW shape needs to match __W shape (1, nx).
        # X is (nx, m), dZ is (1, m) -> np.dot(dZ, X.T) yields (1, nx)
        dW = (1 / m) * np.dot(dZ, X.T)
        db = (1 / m) * np.sum(dZ)

        # Update weights and bias using gradient descent
        self.__W = self.__W - (alpha * dW)
        self.__b = self.__b - (alpha * db)
