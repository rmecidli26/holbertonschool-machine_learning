#!/usr/bin/env python3
""" Train Model with Validation """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, validation_data=None, verbose=True, shuffle=False):
    """ Trains a model using mini-batch gradient descent and validates it """
    # validation_data parametrini network.fit metoduna ötürürük.
    # Əgər None olarsa, Keras bunu normal şəkildə keçirir (validasiya etmir).
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data
    )

    return history
