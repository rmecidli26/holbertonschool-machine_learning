#!/usr/bin/env python3
"""Module to create a learning rate decay operation in TensorFlow"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    Creates a learning rate decay operation in TensorFlow using
    inverse time decay

    Parameters:
        alpha: initial learning rate
        decay_rate: weight used to determine the rate at which alpha decays
        decay_step: number of passes that should occur before alpha decays

    Returns:
        The learning rate decay operation (InverseTimeDecay instance)
    """
    return tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
