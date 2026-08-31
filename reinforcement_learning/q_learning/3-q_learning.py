#!/usr/bin/env python3
"""Module to train a Q-learning agent in a Gymnasium environment."""

import numpy as np

epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(
    env,
    Q,
    episodes=5000,
    max_steps=100,
    alpha=0.1,
    gamma=0.99,
    epsilon=1,
    min_epsilon=0.1,
    epsilon_decay=0.05,
):
    """Performs Q-learning to train an agent in FrozenLake.

    Args:
        env: The FrozenLakeEnv instance.
        Q: A numpy.ndarray containing the initial Q-table.
        episodes: Total number of episodes to train over.
        max_steps: Maximum number of steps per episode.
        alpha: Learning rate.
        gamma: Discount rate.
        epsilon: Initial threshold for epsilon-greedy.
        min_epsilon: Minimum value that epsilon should decay to.
        epsilon_decay: Decay rate for updating epsilon between episodes.

    Returns:
        Q: The updated Q-table.
        total_rewards: List containing the rewards per episode.
    """
    total_rewards = []
    initial_epsilon = epsilon

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Fall in hole adjustment
            if terminated and reward == 0:
                reward = -1

            # Q-learning update rule
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + gamma * Q[next_state, best_next_action]
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            episode_reward += reward
            state = next_state

            if terminated or truncated:
                break

        total_rewards.append(episode_reward)

        # Decay epsilon using exponential decay rule
        epsilon = min_epsilon + (initial_epsilon - min_epsilon) * np.exp(
            -epsilon_decay * episode
        )

    return Q, total_rewards
