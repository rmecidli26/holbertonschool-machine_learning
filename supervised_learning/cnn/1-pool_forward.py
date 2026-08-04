#!/usr/bin/env python3
"""Performs forward propagation over a pooling layer"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs forward propagation
    Returns:
        the output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    out_h = int((h_prev - kh) / sh) + 1
    out_w = int((w_prev - kw) / sw) + 1

    A = np.zeros((m, out_h, out_w, c_prev))

    for i in range(out_h):
        for j in range(out_w):
            slice_h = slice(i * sh, i * sh + kh)
            slice_w = slice(j * sw, j * sw + kw)
            image_slice = A_prev[:, slice_h, slice_w, :]

            if mode == 'max':
                A[:, i, j, :] = np.max(
                    image_slice,
                    axis=(1, 2)
                )
            elif mode == 'avg':
                A[:, i, j, :] = np.mean(
                    image_slice,
                    axis=(1, 2)
                )

    return A
