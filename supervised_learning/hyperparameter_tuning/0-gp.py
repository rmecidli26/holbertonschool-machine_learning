#!/usr/bin/env python3
"""Gaussian Process module."""
import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian process."""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Initialize the Gaussian Process.

        Args:
            X_init (np.ndarray): Shape (t, 1) representing inputs sampled.
            Y_init (np.ndarray): Shape (t, 1) representing outputs for X_init.
            l (float|int): Length parameter for the kernel.
            sigma_f (float|int): Standard deviation given to output.
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """Calculates the covariance kernel matrix using RBF.

        Args:
            X1 (np.ndarray): Shape (m, 1).
            X2 (np.ndarray): Shape (n, 1).

        Returns:
            np.ndarray: Covariance kernel matrix of shape (m, n).
        """
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return (self.sigma_f ** 2) * np.exp(-0.5 / (self.l ** 2) * sqdist)
