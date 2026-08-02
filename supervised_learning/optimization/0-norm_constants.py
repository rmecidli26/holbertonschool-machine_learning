#!/usr/bin/env python3
"""Module to calculate normalization constants of a matrix"""
import numpy as np


def normalization_constants(X):
    """
    Calculates the normalization (standardization) constants of a matrix

    Parameters:
        X: numpy.ndarray of shape (m, nx) to normalize

    Returns:
        mean: numpy.ndarray containing the mean of each feature
        std: numpy.ndarray containing the standard deviation of each feature
    """
    return np.mean(X, axis=0), np.std(X, axis=0)
