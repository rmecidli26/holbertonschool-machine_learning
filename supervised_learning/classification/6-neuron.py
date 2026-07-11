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
        Y (numpy.ndarray): Correct labels
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
        Y (numpy.ndarray): Correct labels

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
        Y (numpy.ndarray): Correct labels
        A (numpy.ndarray): Activated output
        alpha (float): Learning rate.
        """
        m = Y.shape[1]
        dZ = A - Y
        dW = (1 / m) * np.dot(dZ, X.T)
        db = (1 / m) * np.sum(dZ)

        self.__W = self.__W - (alpha * dW)
        self.__b = self.__b - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neuron by running forward propagation and gradient descent
        for a fixed number of iterations.

        Parameters:
        X (numpy.ndarray): Input data with shape (nx, m).
        Y (numpy.ndarray): Correct labels for the input data, shape (1, m).
        iterations (int): The number of iterations to train over.
        alpha (float): The learning rate.

        Returns:
        The evaluation of the training data after training completes.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, (int, float)):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for _ in range(iterations):
            # Step 1: Compute current activations
            A = self.forward_prop(X)
            # Step 2: Compute adjustments based on mistakes
            self.gradient_descent(X, Y, A, alpha)

        # Evaluate and return final predictions and cost after training
        return self.evaluate(X, Y)
