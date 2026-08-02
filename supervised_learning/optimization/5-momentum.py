#!/usr/bin/env python3
"""Module to update variables using """


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using

    Parameters:
        alpha: learning rate
        beta1: momentum weight
        var: numpy.ndarray containing the variable to be updated
        grad: numpy.ndarray containing the gradient of var
        v: previous first moment of var

    Returns:
        var, v: the updated variable and
    """
    v = beta1 * v + (1 - beta1) * grad
    var = var - alpha * v
    return var, v
