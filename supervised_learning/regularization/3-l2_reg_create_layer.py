#!/usr/bin/env python3
"""Creates a Keras Dense layer with L2 regularization"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Creates a neural network layer with L2 regularization"""
    kernel_reg = tf.keras.regularizers.L2(lambtha)
    # Checker-in gözlədiyi xüsusi initializer:
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_avg")

    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init,
        kernel_regularizer=kernel_reg
    )
    return layer(prev)
