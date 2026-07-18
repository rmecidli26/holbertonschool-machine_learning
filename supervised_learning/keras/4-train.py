#!/usr/bin/env python3
""" Train Model """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, 
epochs, verbose=True, shuffle=False):
    """ Trains a model using mini-batch gradient descent """
    # network.fit metodu modeli verilən parametrlərlə təlim edir 
    # və geriyə təlim tarixçəsini (History obyektini) qaytarır.
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle
    )
    
    return history
