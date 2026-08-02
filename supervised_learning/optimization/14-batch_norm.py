#!/usr/bin/env python3
"""Module to create a batch normalization layer in TensorFlow"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in TensorFlow

    Parameters:
        prev: activated output of the previous layer
        n: number of nodes in the layer to be created
        activation: activation function to be used on the output

    Returns:
        A tensor of the activated output for the layer
    """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    dense = tf.keras.layers.Dense(
        units=n,
        use_bias=False,
        kernel_initializer=init
    )(prev)

    batch_norm = tf.keras.layers.BatchNormalization(
        gamma_initializer='ones',
        beta_initializer='zeros',
        epsilon=1e-7
    )(dense)

    if activation is None:
        return batch_norm

    return activation(batch_norm)
