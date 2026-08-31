#!/usr/bin/env python3
"""t-SNE."""

import numpy as np

pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0,
         iterations=1000, lr=500):
    """Perform t-SNE dimensionality reduction.

    Args:
        X: numpy.ndarray of shape (n, d), the dataset.
        ndims: final dimensionality.
        idims: intermediate dimensionality after PCA.
        perplexity: t-SNE perplexity.
        iterations: number of iterations.
        lr: learning rate.

    Returns:
        Y: numpy.ndarray of shape (n, ndims), transformed data.
    """
    # Reduce dimensionality before calculating P affinities
    X = pca(X, idims)

    # Calculate high-dimensional P affinities
    P = P_affinities(X)

    # Initialize Y randomly
    n = X.shape[0]
    Y = np.random.randn(n, ndims)

    # Initialize velocity
    dY = np.zeros_like(Y)

    for i in range(iterations):
        # Early exaggeration during the first 100 iterations
        if i < 100:
            P_current = P * 4
        else:
            P_current = P

        # Calculate gradient and Q affinities
        grad, Q = grads(Y, P_current)

        # Momentum: 0.5 for first 20 iterations, 0.8 afterwards
        momentum = 0.5 if i < 20 else 0.8

        # Gradient descent with momentum
        dY = momentum * dY - lr * grad

        # Update Y
        Y = Y + dY

        # Re-center Y
        Y = Y - np.mean(Y, axis=0)

        # Print cost every 100 iterations
        if (i + 1) % 100 == 0:
            C = cost(P_current, Q)
            print("Cost at iteration {}: {}".format(i + 1, C))

    return Y
