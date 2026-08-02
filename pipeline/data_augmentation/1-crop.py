#!/usr/bin/env python3
"""
TensorFlow ilə şəklin təsadüfi kəsilməsi (random crop) modulu.
"""
import tensorflow as tf


def crop_image(image, size):
    """
    3D tensor şəklində verilmiş şəkli təsadüfi ölçülərlə kəsir.

    Parameters:
        image: kəsiləcək şəkli özündə saxlayan 3D tf.Tensor
        size: kəsiləcək sahənin ölçüsünü göstərən k

    Returns:
        Təsadüfi kəsilmiş tf.Tensor
    """
    return tf.image.random_crop(image, size=size)
