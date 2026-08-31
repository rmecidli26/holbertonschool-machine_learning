#!/usr/bin/env python3
"""Module to select an action using the epsilon-greedy strategy."""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Uses epsilon-greedy strategy to determine the next action.

    Args:
        Q: A numpy.ndarray containing the Q-table.
        state: The current state.
        epsilon: The epsilon value to determine exploration vs exploitation.

    Returns:
        The next action index.
    """
    p = np.random.uniform(0, 1)

    if p < epsilon:
        action = np.random.randint(0, Q.shape[1])
    else:
        action = np.argmax(Q[state])

    return action
