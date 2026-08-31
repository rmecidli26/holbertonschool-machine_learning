#!/usr/bin/env python3
"""Module to play an episode using a trained Q-table."""

import numpy as np


def play(env, Q, max_steps=100):
    """Has the trained agent play an episode in FrozenLake.

    Args:
        env: The FrozenLakeEnv instance.
        Q: A numpy.ndarray containing the Q-table.
        max_steps: Maximum number of steps in the episode.

    Returns:
        total_rewards: The total rewards for the episode.
        rendered_outputs: List of rendered outputs representing board states.
    """
    state = env.s
    rendered_outputs = [env.render()]
    total_rewards = 0

    for step in range(max_steps):
        action = np.argmax(Q[state])
        state, reward, terminated, truncated, _ = env.step(action)
        rendered_outputs.append(env.render())
        total_rewards += reward

        if terminated or truncated:
            break

    return total_rewards, rendered_outputs
