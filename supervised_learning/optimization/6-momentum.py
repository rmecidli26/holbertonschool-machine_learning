#!/usr/bin/env python3
"""Module to set up Gradient Descent with Momentum optimizer in TensorFlow"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Sets up the gradient descent with momentum optimization algorithm in TF

    Parameters:
        alpha: learning rate
        beta1: momentum weight

    Returns:
        optimizer: tf.keras.optimizers.SGD instance with momentum
    """
    return tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
