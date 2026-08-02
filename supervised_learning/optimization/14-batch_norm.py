#!/usr/bin/env python3
"""Module to create a batch normalization layer in TensorFlow"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer"""

    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    x = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=init,
        use_bias=False
    )(prev)

    x = tf.keras.layers.BatchNormalization(
        axis=-1,
        momentum=0.99,
        epsilon=1e-7,
        center=True,
        scale=True,
        beta_initializer=tf.zeros_initializer(),
        gamma_initializer=tf.ones_initializer()
    )(x)

    if activation is not None:
        x = activation(x)

    return x
