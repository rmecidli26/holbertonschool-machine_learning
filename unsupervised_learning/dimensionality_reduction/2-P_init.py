#!/usr/bin/env python3
"""Initialize variables for t-SNE."""

import numpy as np


def P_init(X, perplexity):
    """Initialize variables for calculating P affinities.

    Args:
        X: numpy.ndarray of shape (n, d)
        perplexity: perplexity of the Gaussian distributions

    Returns:
        D: squared pairwise distance matrix of shape (n, n)
        P: initialized P matrix of shape (n, n)
        betas: beta values of shape (n, 1)
        H: Shannon entropy of perplexity with base 2
    """
    n = X.shape[0]

    # Squared pairwise Euclidean distances
    X_squared = np.sum(X ** 2, axis=1, keepdims=True)
    D = X_squared + X_squared.T - 2 * np.matmul(X, X.T)

    # Avoid tiny negative values caused by floating-point errors
    D = np.maximum(D, 0)

    # Initialize P and betas
    P = np.zeros((n, n))
    betas = np.ones((n, 1))

    # Shannon entropy corresponding to the perplexity
    H = np.log2(perplexity)

    return D, P, betas, H
