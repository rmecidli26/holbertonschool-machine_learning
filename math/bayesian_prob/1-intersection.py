#!/usr/bin/env python3
"""Module to calculate the intersection without importing numpy"""


def intersection(x, n, P, Pr):
    """Calculates the intersection of obtaining data with prior beliefs"""
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

    if not hasattr(Pr, "shape") or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

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

    pr_sum = 0
    result = P.copy()

    for i in range(len(P)):
        p = P[i]
        pr = Pr[i]

        if p < 0 or p > 1:
            raise ValueError("All values in P must be in the range [0, 1]")

        if pr < 0 or pr > 1:
            raise ValueError("All values in Pr must be in the range [0, 1]")

        pr_sum += pr
        likelihood_val = combination * (p**x) * ((1 - p) ** (n - x))
        result[i] = likelihood_val * pr

    if abs(pr_sum - 1.0) > 1e-5:
        raise ValueError("Pr must sum to 1")

    return result
