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

            if terminated and reward == 0:
                reward = -1

            # Eligibility trace-i yeniləyirik
            Et *= gamma * lambtha
            Et[state] += 1

            # Terminal vəziyyətdə növbəti state dəyəri 0 olur
            if terminated or truncated:
                td_target = reward
            else:
                td_target = reward + gamma * V[next_state]

            td_error = td_target - V[state]

            # V matrisini və trace-ləri yeniləyirik
            V += alpha * td_error * Et

            state = next_state

            if terminated or truncated:
                break

    return V
