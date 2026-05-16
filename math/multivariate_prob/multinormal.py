#!/usr/bin/env python3
"""Module that defines the MultiNormal class with PDF function"""
import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution"""

    def __init__(self, data):
        """Initializes mean 2D dataset of shape (d, n)"""
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)
        data_centered = data - self.mean
        self.cov = np.dot(data_centered, data_centered.T) / (n - 1)

    def pdf(self, x):
        """Calculates the PDF at a data point x of shape (d, 1)"""
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]

        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)

        # Düsturun eksponent (üstlü) hissəsinin hesablanması
        x_centered = x - self.mean
        exponent = -0.5 * np.dot(np.dot(x_centered.T, inv), x_centered)

        # Məxrəcdəki normallaşdırma sabitinin hesablanması
        normalization = 1.0 / np.sqrt(((2 * np.pi) ** d) * det)

        # Nəticə skalyar dəyər olmalıdır, massiv daxilindən çıxarılır [0, 0]
        pdf_value = normalization * np.exp(exponent)

        return pdf_value[0, 0]
