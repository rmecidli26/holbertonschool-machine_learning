#!/usr/bin/env python3

import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the TD(lambda) algorithm.

    Args:
        env: environment instance
        V: numpy.ndarray of shape (s,) containing value estimates
        policy: function that takes a state and returns an action
        lambtha: eligibility trace factor
        episodes: number of episodes
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate

    Returns:
        V: updated value estimate
    """
    for _ in range(episodes):
        state, _ = env.reset()
        eligibility = np.zeros_like(V)

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # TD error
            if terminated or truncated:
                delta = reward - V[state]
            else:
                delta = reward + gamma * V[next_state] - V[state]

            # Accumulating eligibility trace
            eligibility[state] += 1

            # Update all states
            V += alpha * delta * eligibility

            # Decay eligibility traces
            eligibility *= gamma * lambtha

            if terminated or truncated:
                break

            state = next_state

    return V
