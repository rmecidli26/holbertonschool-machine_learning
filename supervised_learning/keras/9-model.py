#!/usr/bin/env python3
""" Module for saving and loading Keras models """
import tensorflow.keras as K


def save_model(network, filename):
    """ Saves an entire model to a specific file path

    Args:
        network: The Keras model instance to save
        filename: The path of the file where the model should be saved
    Returns:
        None
    """
    network.save(filename)


def load_model(filename):
    """ Loads an entire model from a specific file path

    Args:
        filename: The path of the file the model should be loaded from
    Returns:
        The loaded Keras model instance
    """
    return K.models.load_model(filename)
