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
    
    layer = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=init,
        use_bias=False
    )
    base_output = layer(prev)
    
    norm_layer = tf.keras.layers.BatchNormalization(
        epsilon=1e-7
    )
    norm_output = norm_layer(base_output)
    
    if activation is None:
        return norm_output
        
    return activation(norm_output)
