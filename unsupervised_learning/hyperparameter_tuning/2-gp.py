#!/usr/bin/env python3
"""Gaussian Process update implementation using NumPy.
"""

import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian process."""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Initialize the Gaussian Process."""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """Calculate the covariance kernel matrix between two matrices
        using the Radial Basis Function (RBF).
        """
        sqdist = np.sum(X1 ** 2, axis=1, keepdims=True) - \
            2 * np.dot(X1, X2.T) + np.sum(X2 ** 2, axis=1)
        return (self.sigma_f ** 2) * np.exp(-0.5 / (self.l ** 2) * sqdist)

    def predict(self, X_s):
        """Predict the mean and standard deviation of points
        in a Gaussian process.
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        mu = np.dot(K_s.T, np.dot(K_inv, self.Y)).reshape(-1)
        sigma = np.diagonal(K_ss - np.dot(K_s.T, np.dot(K_inv, K_s)))

        return mu, sigma

    def update(self, X_new, Y_new):
        """Update a Gaussian Process with new sample points."""
        self.X = np.vstack((self.X, X_new))
        self.Y = np.vstack((self.Y, Y_new))
        self.K = self.kernel(self.X, self.X)
