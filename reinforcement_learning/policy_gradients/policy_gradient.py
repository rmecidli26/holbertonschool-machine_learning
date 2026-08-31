#!/usr/bin/env python3
"""Module to compute the policy gradient."""

import numpy as np


def policy(matrix, weight):
    """Computes the policy (action probabilities) using Softmax.

    Args:
        matrix: numpy.ndarray of shape (1, n) representing the state.
        weight: numpy.ndarray of shape (n, m) representing the weight matrix.

    Returns:
        numpy.ndarray of shape (1, m) containing the action probabilities.
    """
    z = np.matmul(matrix, weight)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def policy_gradient(state, weight):
    """Computes the Monte-Carlo policy gradient based on a state and weight matrix.

    Args:
        state: numpy.ndarray representing the current state observation.
        weight: numpy.ndarray of shape (n, m) representing the weight matrix.

    Returns:
        action: Sampled action based on the policy probabilities.
        grad: The gradient of the log-policy with respect to the weight matrix.
    """
    probs = policy(state, weight)
    action = np.random.choice(probs.shape[1], p=probs[0])

    dsoftmax = probs.copy()
    dsoftmax[0, action] -= 1
    grad = np.matmul(state.T, -dsoftmax)

    return action, grad
