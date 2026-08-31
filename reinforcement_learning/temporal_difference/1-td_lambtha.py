#!/usr/bin/env python3

import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the TD(lambda) algorithm.

    Returns:
        V: updated value estimate
    """
    for _ in range(episodes):
        state, _ = env.reset()

        # Eligibility traces
        E = np.zeros_like(V)

        for _ in range(max_steps):
            action = policy(state)

            next_state, reward, terminated, truncated, _ = env.step(action)

            # TD error
            if terminated or truncated:
                delta = reward - V[state]
            else:
                delta = reward + gamma * V[next_state] - V[state]

            # Accumulate eligibility for current state
            E[state] += 1

            # Update value function
            V += alpha * delta * E

            # Decay eligibility traces
            E *= gamma * lambtha

            if terminated or truncated:
                break

            state = next_state

    return V
