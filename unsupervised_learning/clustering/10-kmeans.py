#!/usr/bin/env python3
"""Performs K-means clustering using sklearn."""

import sklearn.cluster


def kmeans(X, k):
    """Performs K-means on a dataset.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d).
        k (int): Number of clusters.

    Returns:
        C, clss:
            C: Centroids of shape (k, d).
            clss: Cluster labels of shape (n,).
    """
    model = sklearn.cluster.KMeans(n_clusters=k)

    clss = model.fit_predict(X)
    C = model.cluster_centers_

    return C, clss
