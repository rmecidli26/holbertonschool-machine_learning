#!/usr/bin/env python3
"""Module to calculate a correlation matrix from a covariance matrix"""
import numpy as np


def correlation(C):
    """Calculates the correlation matrix given a covariance matrix C"""
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    # Diaqonal elementləri (dispersiyaları) götürüb kökünü alırıq (standart meyl)
    std_devs = np.sqrt(np.diag(C))

    # Xarici hasil vasitəsilə hər bir i,j kombinasiyası üçün standart meyllərin hasilini tapırıq
    std_matrix = np.outer(std_devs, std_devs)

    # Korrelyasiya matrisini hesablayırıq
    correlation_matrix = C / std_matrix

    return correlation_matrix
