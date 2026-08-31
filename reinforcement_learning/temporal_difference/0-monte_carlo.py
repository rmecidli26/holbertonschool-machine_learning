#!/usr/bin/env python3
"""Module to perform Monte Carlo algorithm for policy evaluation."""

import numpy as np


def monte_carlo(
    env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99
):
    """Performs the Monte Carlo algorithm to evaluate a given policy.

    Args:
        env: Environment instance.
        V: Numpy array of shape (s,) containing the value estimate.
        policy: Function that takes in a state and returns next action.
        episodes: Total number of episodes to train over.
        max_steps: Maximum number of steps per episode.
        alpha: Learning rate.
        gamma: Discount rate.

    Returns:
        V: The updated value estimate.
    """
    for episode in range(episodes):
        state, _ = env.reset()
        episode_data = []

        for step in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_data.append((state, action, reward))
            state = next_state

            if terminated or truncated:
                break

        G = 0
        # Traverse backwards through the episode (Every-Visit MC)
        for state, action, reward in reversed(episode_data):
            G = gamma * G + reward
            V[state] = V[state] + alpha * (G - V[state])

    return V
