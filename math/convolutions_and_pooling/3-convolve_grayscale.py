#!/usr/bin/env python3
"""Performs a strided convolution"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Performs a convolution on grayscale images with
    Returns:
        numpy.ndarray of convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2
        if ((h - 1) * sh + kh - h) % 2 != 0:
            ph += 1

        pw = ((w - 1) * sw + kw - w) // 2
        if ((w - 1) * sw + kw - w) % 2 != 0:
            pw += 1
    else:
        ph, pw = padding

    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )

    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1

    convolved = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            slice_h = slice(i * sh, i * sh + kh)
            slice_w = slice(j * sw, j * sw + kw)
            image_slice = padded[:, slice_h, slice_w]
            convolved[:, i, j] = np.sum(
                image_slice * kernel,
                axis=(1, 2)
            )

    return convolved
