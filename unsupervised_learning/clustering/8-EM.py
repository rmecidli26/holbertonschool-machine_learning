#!/usr/bin/env python3
"""Performs the expectation maximization for a GMM"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """Performs the expectation maximization for a GMM"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None, None, None
    if type(k) is not int or k <= 0:
        return None, None, None, None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None, None
    if type(tol) is not float or tol < 0:
        return None, None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    g, lkhd_prev = expectation(X, pi, m, S)
    if g is None or lkhd_prev is None:
        return None, None, None, None, None

    for i in range(iterations):
        if verbose and i % 10 == 0:
            print('Log Likelihood after {} iterations: {}'.format(
                i, round(lkhd_prev, 5)))

        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return None, None, None, None, None

        g, lkhd = expectation(X, pi, m, S)
        if g is None or lkhd is None:
            return None, None, None, None, None

        if np.abs(lkhd - lkhd_prev) <= tol:
            break

        lkhd_prev = lkhd

    i += 1
    if verbose:
        print('Log Likelihood after {} iterations: {}'.format(
            i, round(lkhd, 5)))

    return pi, m, S, g, lkhd
