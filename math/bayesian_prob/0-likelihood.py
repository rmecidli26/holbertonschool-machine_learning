#!/usr/bin/env python3
import math


def likelihood(x, n, P):
    if not isinstance(n, (int, float)) or isinstance(n, bool) or n != int(n) or n <= 0:
        raise ValueError("n must be a positive integer")
    n = int(n)

    if not isinstance(x, (int, float)) or isinstance(x, bool) or x != int(x) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")
    x = int(x)

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not hasattr(P, "ndim") or P.ndim != 1 or not hasattr(P, "shape"):
        raise TypeError("P must be a 1D numpy.ndarray")

    fact_n = math.factorial(n)
    fact_x = math.factorial(x)
    fact_nx = math.factorial(n - x)
    combination = fact_n / (fact_x * fact_nx)

    result_list = []
    for p in P:
        if p < 0 or p > 1:
            raise ValueError("All values in P must be in the range [0, 1]")
        
        likelihood_val = combination * (p ** x) * ((1 - p) ** (n - x))
        result_list.append(likelihood_val)

    return P.__class__(result_list)
