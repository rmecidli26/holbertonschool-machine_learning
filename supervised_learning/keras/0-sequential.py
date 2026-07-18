#!/usr/bin/env python3
""" Keras Sequential Model """
import tensorflow as tf


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ Builds a neural network with the Keras library """
    model = tf.keras.Sequential()
    regularizer = tf.keras.regularizers.l2(lambtha)

    for i in range(len(layers)):
        if i == 0:
            # First layer
            model.add(tf.keras.layers.Dense(
                units=layers[i],
                activation=activations[i],
                kernel_regularizer=regularizer,
                input_dim=nx
            ))
        else:
            model.add(tf.keras.layers.Dense(
                units=layers[i],
                activation=activations[i],
                kernel_regularizer=regularizer
            ))

        # Last layer check for dropout
        if i < len(layers) - 1:
            model.add(tf.keras.layers.Dropout(rate=1 - keep_prob))

    return model
