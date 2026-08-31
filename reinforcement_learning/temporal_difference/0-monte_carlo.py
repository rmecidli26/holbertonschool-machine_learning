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

        # Calculate returns
        G = 0
        returns = []
        for s, a, r in reversed(episode_data):
            G = gamma * G + r
            returns.append((s, G))

        returns.reverse()

        # First-Visit MC update
        visited = set()
        for s, G in returns:
            if s not in visited:
                V[s] = V[s] + alpha * (G - V[s])
                visited.add(s)

    return V
