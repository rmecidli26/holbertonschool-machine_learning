#!/usr/bin/env python3
"""Performs agglomerative clustering."""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Performs agglomerative clustering on a dataset.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d).
        dist (float): Maximum cophenetic distance.

    Returns:
        numpy.ndarray: Cluster indices for each data point.
    """
    linkage = scipy.cluster.hierarchy.linkage(X, method='ward')

    scipy.cluster.hierarchy.dendrogram(
        linkage,
        color_threshold=dist
    )

    clss = scipy.cluster.hierarchy.fcluster(
        linkage,
        t=dist,
        criterion='distance'
    )

    return clss - 1
