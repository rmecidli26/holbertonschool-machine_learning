#!/usr/bin/env python3
"""Performs pooling on images"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Performs pooling on images using 2 loops
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    out_h = (h - kh) // sh + 1
    out_w = (w - kw) // sw + 1

    pooled = np.zeros((m, out_h, out_w, c))

    for i in range(out_h):
        for j in range(out_w):
            slice_h = slice(i * sh, i * sh + kh)
            slice_w = slice(j * sw, j * sw + kw)
            image_slice = images[:, slice_h, slice_w, :]

            if mode == 'max':
                pooled[:, i, j, :] = np.max(image_slice, axis=(1, 2))
            elif mode == 'avg':
                pooled[:, i, j, :] = np.mean(image_slice, axis=(1, 2))

    return pooled
