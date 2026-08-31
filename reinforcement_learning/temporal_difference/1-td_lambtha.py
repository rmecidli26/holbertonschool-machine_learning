#!/usr/bin/env python3
"""Module to perform the TD(lambda) algorithm."""

import numpy as np


def td_lambtha(
    env, V, policy, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99
):
    """Performs the TD(lambda) algorithm for policy evaluation.

    Args:
        env: Environment instance.
        V: Numpy array of shape (s,) containing the value estimate.
        policy: Function that takes in a state and returns next action.
        lambtha: Eligibility trace factor.
        episodes: Total number of episodes to train over.
        max_steps: Maximum number of steps per episode.
        alpha: Learning rate.
        gamma: Discount rate.

    Returns:
        V: The updated value estimate.
    """
    n_states = V.shape[0]

    for episode in range(episodes):
        state, _ = env.reset()
        Et = np.zeros(n_states)

        for step in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Adjust reward when falling in a hole
            if terminated and reward == 0:
                reward = -1

            # Update eligibility trace (accumulating trace)
            Et[state] += 1

            # Temporal Difference error
            td_error = reward + gamma * V[next_state] - V[state]

            # Update value function and decay eligibility traces
            V += alpha * td_error * Et
            Et *= gamma * lambtha

            state = next_state

            if terminated or truncated:
                break

    return V
