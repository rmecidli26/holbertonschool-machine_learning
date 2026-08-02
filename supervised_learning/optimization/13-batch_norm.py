#!/usr/bin/env python3
"""Module to perform batch normalization using numpy"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network
    using batch normalization

    Parameters:
        Z: numpy.ndarray of shape (m, n) to be normalized
        gamma: numpy.ndarray of shape (1, n) containing the scales
        beta: numpy.ndarray of shape (1, n) containing the offsets
        epsilon: small number used to avoid division by zero

    Returns:
        The normalized Z matrix
    """
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)
    out = gamma * Z_norm + beta
    return out
