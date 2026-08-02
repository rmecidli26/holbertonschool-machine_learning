#!/usr/bin/env python3
"""Module to calculate the exponentially"""


def moving_average(data, beta):
    """
    Calculates the weighted moving average

    Parameters:
        data: list of data to calculate the moving average of
        beta: weight used for the moving average

    Returns:
        list containing the moving averages of data
    """
    moving_averages = []
    v = 0

    for i, theta in enumerate(data, 1):
        v = beta * v + (1 - beta) * theta
        v_corrected = v / (1 - beta ** i)
        moving_averages.append(v_corrected)

    return moving_averages
