#!/usr/bin/env python3
"""Calculates a GMM using sklearn."""

import sklearn.mixture


def gmm(X, k):
    """Calculates a GMM from a dataset.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d).
        k (int): Number of clusters.

    Returns:
        pi, m, S, clss, bic
    """
    model = sklearn.mixture.GaussianMixture(n_components=k)

    model.fit(X)

    pi = model.weights_
    m = model.means_
    S = model.covariances_
    clss = model.predict(X)
    bic = model.bic(X)

    return pi, m, S, clss, bic
