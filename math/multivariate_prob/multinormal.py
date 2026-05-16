#!/usr/bin/env python3
"""Module that defines the MultiNormal class"""
import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution"""

    def __init__(self, data):
        """Initializes mean and covariance from a 2D dataset of shape (d, n)"""
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        # Orta qiyməti hesablayırıq və formasını (d, 1) edirik
        self.mean = np.mean(data, axis=1, keepdims=True)

        # Mərkəzləşdirilmiş datanı tapırıq: (d, n) - (d, 1)
        data_centered = data - self.mean

        # Kovarians matrisini əllə hesablayırıq: (d, n) x (n, d) -> (d, d)
        self.cov = np.dot(data_centered, data_centered.T) / (n - 1)
