#!/usr/bin/env python3
"""Builds the ResNet-50 architecture using Keras"""

from tensorflow import keras as K

identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Returns:
        the compiled Keras model
    """
    initializer = K.initializers.he_normal(seed=0)
    X_input = K.Input(shape=(224, 224, 3))

    # Stage 1: Conv1 & MaxPool
    X = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding='same',
        kernel_initializer=initializer
    )(X_input)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(X)

    # Stage 2: conv2_x (1 projection block s=1, 2 identity blocks)
    X = projection_block(X, filters=[64, 64, 256], s=1)
    X = identity_block(X, filters=[64, 64, 256])
    X = identity_block(X, filters=[64, 64, 256])

    # Stage 3: conv3_x (1 projection block s=2, 3 identity blocks)
    X = projection_block(X, filters=[128, 128, 512], s=2)
    X = identity_block(X, filters=[128, 128, 512])
    X = identity_block(X, filters=[128, 128, 512])
    X = identity_block(X, filters=[128, 128, 512])

    # Stage 4: conv4_x (1 projection block s=2, 5 identity blocks)
    X = projection_block(X, filters=[256, 256, 1024], s=2)
    X = identity_block(X, filters=[256, 256, 1024])
    X = identity_block(X, filters=[256, 256, 1024])
    X = identity_block(X, filters=[256, 256, 1024])
    X = identity_block(X, filters=[256, 256, 1024])
    X = identity_block(X, filters=[256, 256, 1024])

    # Stage 5: conv5_x (1 projection block s=2, 2 identity blocks)
    X = projection_block(X, filters=[512, 512, 2048], s=2)
    X = identity_block(X, filters=[512, 512, 2048])
    X = identity_block(X, filters=[512, 512, 2048])

    # Average Pooling & Dense Output Layer
    X = K.layers.AveragePooling2D(
        pool_size=(7, 7),
        strides=(1, 1)
    )(X)

    X = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=initializer
    )(X)

    model = K.Model(inputs=X_input, outputs=X)

    return model
