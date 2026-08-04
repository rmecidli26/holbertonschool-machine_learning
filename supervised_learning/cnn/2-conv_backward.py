#!/usr/bin/env python3
"""Performs back propagation"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Performs back propagation over a convolutional layer of a neural network
    Returns:
        dA_prev, dW, db
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    _, h_new, w_new, _ = dZ.shape
    sh, sw = stride

    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))

    padded_A_prev = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    padded_dA_prev = np.zeros_like(padded_A_prev)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(h_new):
        for j in range(w_new):
            slice_h = slice(i * sh, i * sh + kh)
            slice_w = slice(j * sw, j * sw + kw)

            for k in range(c_new):
                dz_slice = dZ[:, i:i + 1, j:j + 1, k:k + 1]
                padded_dA_prev[:, slice_h, slice_w, :] += (
                    dz_slice * W[:, :, :, k]
                )

                a_slice = padded_A_prev[:, slice_h, slice_w, :]
                dW[:, :, :, k] += np.sum(
                    a_slice * dz_slice,
                    axis=0
                )

    if padding == 'same':
        if ph > 0 and pw > 0:
            dA_prev = padded_dA_prev[:, ph:-ph, pw:-pw, :]
        elif ph > 0:
            dA_prev = padded_dA_prev[:, ph:-ph, :, :]
        elif pw > 0:
            dA_prev = padded_dA_prev[:, :, pw:-pw, :]
        else:
            dA_prev = padded_dA_prev
    else:
        dA_prev = padded_dA_prev

    return dA_prev, dW, db
