#!/usr/bin/env python3
"""Module to crop an image randomly"""
import tensorflow as tf


def crop_image(image, size):
    """Crops an image randomly to a given size"""
    return tf.image.random_crop(image, size=size)
