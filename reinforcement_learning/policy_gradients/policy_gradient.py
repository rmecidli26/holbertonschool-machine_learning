#!/usr/bin/env python3
"""Module to compute the policy gradient."""

import numpy as np


def policy(matrix, weight):
    """Computes the policy (action probabilities) using Softmax.

    Args:
        matrix: numpy.ndarray representing the state.
        weight: numpy.ndarray of shape (n, m) representing

    Returns:
        numpy.ndarray of shape (1, m) containing the action probabilities.
    """
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    z = np.matmul(matrix, weight)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def policy_gradient(state, weight):
    """Computes the Monte-Carlo policy gradient for a state

    Args:
        state: numpy.ndarray representing current state observation.
        weight: numpy.ndarray of shape (n, m) representing

    Returns:
        action: Sampled action based on policy probabilities.
        grad: Gradient of log-policy with respect to weight matrix.
    """
    if state.ndim == 1:
        state = state.reshape(1, -1)

    probs = policy(state, weight)
    action = np.random.choice(probs.shape[1], p=probs[0])

    dsoftmax = -probs
    dsoftmax[0, action] += 1
    grad = np.matmul(state.T, dsoftmax)

    return action, grad
