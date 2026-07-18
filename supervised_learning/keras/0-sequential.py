#!/usr/bin/env python3
""" Keras Sequential Model """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ Builds a neural network with the Keras library """
    model = K.Sequential()
    regularizer = K.regularizers.l2(lambtha)

    for i in range(len(layers)):
        if i == 0:
            # First layer
            model.add(K.layers.Dense(
                units=layers[i],
                activation=activations[i],
                kernel_regularizer=regularizer,
                input_dim=nx
            ))
        else:
            model.add(K.layers.Dense(
                units=layers[i],
                activation=activations[i],
                kernel_regularizer=regularizer
            ))

        # Last layer check for dropout
        if i < len(layers) - 1:
            model.add(K.layers.Dropout(rate=1 - keep_prob))

    return model
