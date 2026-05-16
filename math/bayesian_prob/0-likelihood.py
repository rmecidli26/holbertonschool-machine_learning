#!/usr/bin/env python3
"""Module to calculate likelihood of binomial distribution"""


def likelihood(x, n, P):
    """Calculates the likelihood of obtaining data given probabilities"""
    if not isinstance(n, (int, float)) or n <= 0 or n != int(n):
        raise ValueError("n must be a positive integer")
    n = int(n)

    if not isinstance(x, (int, float)) or x < 0 or x != int(x):
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )
    x = int(x)

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not hasattr(P, "ndim") or P.ndim != 1 or not hasattr(P, "shape"):
        raise TypeError("P must be a 1D numpy.ndarray")

    fact_n = 1
    for i in range(1, n + 1):
        fact_n *= i

    fact_x = 1
    for i in range(1, x + 1):
        fact_x *= i

    fact_nx = 1
    for i in range(1, n - x + 1):
        fact_nx *= i

    combination = fact_n / (fact_x * fact_nx)

    result = P.copy()
    for i in range(len(P)):
        p = P[i]
        if p < 0 or p > 1:
            raise ValueError("All values in P must be in the range [0, 1]")
        result[i] = combination * (p**x) * ((1 - p) ** (n - x))

    return result
