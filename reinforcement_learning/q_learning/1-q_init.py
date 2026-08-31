#!/usr/bin/env python3
"""Module to initialize the Q-table for reinforcement learning."""

import numpy as np


def q_init(env):
    """Initializes a Q-table with zeros for a given Gymnasium environment.

    Args:
        env: The FrozenLakeEnv instance.

    Returns:
        A numpy.ndarray of zeros with shape (action_space.n
    """
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.n

    return np.zeros((state_space_size, action_space_size))
