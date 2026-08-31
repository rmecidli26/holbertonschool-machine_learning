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
        states = []
        rewards = []

        for step in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Quyuya düşəndə reward 0-dırsa, -1 kimi nəzərə alınır
            if terminated and reward == 0:
                reward = -1

            states.append(state)
            rewards.append(reward)
            state = next_state

            if terminated or truncated:
                break

        G = 0
        visited = set()
        # Arxadan qabağa hesablama (First-Visit MC)
        for t in range(len(states) - 1, -1, -1):
            s = states[t]
            G = gamma * G + rewards[t]

            if s not in states[:t]:
                V[s] = V[s] + alpha * (G - V[s])

    return V
