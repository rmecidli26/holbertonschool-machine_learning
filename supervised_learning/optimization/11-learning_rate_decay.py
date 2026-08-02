#!/usr/bin/env python3
"""Module to update learning rate using inverse time decay in numpy"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay in numpy

    Parameters:
        alpha: original learning rate
        decay_rate: weight used to determine the rate
        global_step: number of passes of gradient
        decay_step: number of passes that should occur before decay

    Returns:
        The updated value for alpha
    """
    return alpha / (1 + decay_rate * (global_step // decay_step))
