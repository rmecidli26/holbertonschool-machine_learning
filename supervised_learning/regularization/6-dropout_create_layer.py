#!/usr/bin/env python3
"""Creates a Keras Dense layer with Dropout"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """Creates a neural network layer"""
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_avg")
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init
    )
    dropout = tf.keras.layers.Dropout(rate=1 - keep_prob)

    x = layer(prev)
    return dropout(x, training=training)
