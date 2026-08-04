#!/usr/bin/env python3
"""Performs forward propagation over a convolutional layer"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Performs forward propagation over a convolutional layer
    Returns:
        the output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))

    padded = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    out_h = int((h_prev + 2 * ph - kh) / sh) + 1
    out_w = int((w_prev + 2 * pw - kw) / sw) + 1

    Z = np.zeros((m, out_h, out_w, c_new))

    for k in range(c_new):
        kernel = W[:, :, :, k]
        bias = b[:, :, :, k]
        for i in range(out_h):
            for j in range(out_w):
                slice_h = slice(i * sh, i * sh + kh)
                slice_w = slice(j * sw, j * sw + kw)
                image_slice = padded[:, slice_h, slice_w, :]
                Z[:, i, j, k] = np.sum(
                    image_slice * kernel,
                    axis=(1, 2, 3)
                ) + bias

    return activation(Z)
