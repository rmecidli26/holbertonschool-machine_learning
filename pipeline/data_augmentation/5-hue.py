#!/usr/bin/env python3
"""Module to change the hue of an image"""
import tensorflow as tf


def change_hue(image, delta):
    """Changes the hue of an image"""
    return tf.image.adjust_hue(image, delta)
