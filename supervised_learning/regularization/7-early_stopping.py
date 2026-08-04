#!/usr/bin/env python3
"""Determines if gradient descent should stop early"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Checks early stopping condition"""
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1

    stop = count >= patience
    return stop, count
