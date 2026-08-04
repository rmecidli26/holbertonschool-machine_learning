#!/usr/bin/env python3
"""Performs a same convolution on grayscale images"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """Performs a same convolution on grayscale images using 2 loops
    Returns:
        numpy.ndarray of convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate padding needed to keep output dimensions same as input
    ph = kh // 2
    pw = kw // 2

    # Pad images with zeros along height and width
    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )

    convolved = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            image_slice = padded[:, i:i + kh, j:j + kw]
            convolved[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2))

    return convolved
