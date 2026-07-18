#!/usr/bin/env python3
""" One Hot Matrix """
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """ Converts a label vector into a one-hot matrix """
    # K.utils.to_categorical funksiy
    # Əgər classes=None olar
    return K.utils.to_categorical(labels, num_classes=classes)
