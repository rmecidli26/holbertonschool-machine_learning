#!/usr/bin/env python3
"""Builds a projection block using Keras"""

from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """Builds a projection block as described
    Returns:
        the activated output of the projection block
    """
    F11, F3, F12 = filters
    initializer = K.initializers.he_normal(seed=0)

    # Main Path
    # First component of main path
    X = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        strides=(s, s),
        padding='same',
        kernel_initializer=initializer
    )(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Second component of main path
    X = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer=initializer
    )(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Third component of main path
    X = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    # Shortcut Path
    X_shortcut = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(s, s),
        padding='same',
        kernel_initializer=initializer
    )(A_prev)
    X_shortcut = K.layers.BatchNormalization(axis=3)(X_shortcut)

    # Final step: Add shortcut to main path and pass through ReLU
    X = K.layers.Add()([X, X_shortcut])
    X = K.layers.Activation('relu')(X)

    return X
