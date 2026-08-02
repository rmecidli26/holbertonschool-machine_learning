#!/usr/bin/env python3
"""
TensorFlow ilə şəklin üfüqi çevrilməsi modulu.
"""
import tensorflow as tf


def flip_image(image):
    """
    3D tensor şəklində verilmiş şəkli üfüqi olaraq çevirir.

    Parameters:
        image: çevriləcək şəkli özündə saxlayan 3D tf.Tensor

    Returns:
        Üfüqi çevrilmiş tf.Tensor
    """
    return tf.image.flip_left_right(image)
