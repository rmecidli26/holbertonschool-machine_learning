#!/usr/bin/env python3
"""Module to load the FrozenLake environment from Gymnasium."""

import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Loads the FrozenLakeEnv environment from gymnasium.

    Args:
        desc: List of lists containing a custom map description.
        map_name: String containing pre-made map name ('4x4' or '8x8').
        is_slippery: Boolean to determine if ice is slippery.

    Returns:
        The gymnasium FrozenLake environment instance.
    """
    if desc is None and map_name is None:
        env = gym.make(
            "FrozenLake-v1",
            desc=None,
            map_name="8x8",
            is_slippery=is_slippery,
            render_mode="ansi",
        )
    else:
        env = gym.make(
            "FrozenLake-v1",
            desc=desc,
            map_name=map_name,
            is_slippery=is_slippery,
            render_mode="ansi",
        )

    return env
