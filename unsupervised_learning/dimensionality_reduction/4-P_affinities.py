#!/usr/bin/env python3
"""Calculate symmetric P affinities for t-SNE."""

import numpy as np

P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """Calculate the symmetric P affinities of a dataset.

    Args:
        X: numpy.ndarray of shape (n, d)
        tol: maximum allowed difference in Shannon entropy
        perplexity: desired perplexity

    Returns:
        P: numpy.ndarray of shape (n, n), symmetric P affinities.
    """
    D, P, betas, H = P_init(X, perplexity)
    n = X.shape[0]

    for i in range(n):
        Di = np.concatenate((D[i, :i], D[i, i + 1:]))

        beta = betas[i, 0]
        beta_low = None
        beta_high = None

        while True:
            Hi, Pi = HP(Di, np.array([beta]))
            diff = Hi - H

            if abs(diff) <= tol:
                break

            if diff > 0:
                # Entropy is too high -> increase beta
                beta_low = beta

                if beta_high is None:
                    beta *= 2
                else:
                    beta = (beta_low + beta_high) / 2
            else:
                # Entropy is too low -> decrease beta
                beta_high = beta

                if beta_low is None:
                    beta /= 2
                else:
                    beta = (beta_low + beta_high) / 2

        betas[i, 0] = beta

        P[i, :i] = Pi[:i]
        P[i, i + 1:] = Pi[i:]

    # Symmetrize P and normalize
    P = (P + P.T) / (2 * n)

    return P
