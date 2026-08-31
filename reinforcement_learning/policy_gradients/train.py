#!/usr/bin/env python3
"""Module to implement Monte Carlo policy gradient training for CartPole."""

import numpy as np

policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """Implements full policy gradient training over specified episodes.

    Args:
        env: Initial Gymnasium environment.
        nb_episodes: Number of episodes used for training.
        alpha: Learning rate.
        gamma: Discount factor.

    Returns:
        scores: List containing total rewards (scores) earned per episode.
    """
    weight = np.random.rand(env.observation_space.shape[0], env.action_space.n)
    scores = []

    for episode in range(nb_episodes):
        state, _ = env.reset()
        gradients = []
        rewards = []
        score = 0.0

        while True:
            action, grad = policy_gradient(state, weight)
            next_state, reward, terminated, truncated, _ = env.step(action)

            gradients.append(grad)
            rewards.append(reward)
            score += reward
            state = next_state

            if terminated or truncated:
                break

        scores.append(score)

        # Compute discounted returns and update weights using policy gradient
        for i in range(len(rewards)):
            G = sum([r * (gamma ** j) for j, r in enumerate(rewards[i:])])
            weight += alpha * gradients[i] * G

        print("Episode: {} Score: {}".format(episode, score))

    return scores
