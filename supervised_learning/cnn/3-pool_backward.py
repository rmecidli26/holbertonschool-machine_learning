#!/usr/bin/env python3
"""Performs back propagation over a pooling layer"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs back propagation
    Returns:
        dA_prev: partial derivatives
    """
    m, h_new, w_new, c_new = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)

    for i in range(h_new):
        for j in range(w_new):
            slice_h = slice(i * sh, i * sh + kh)
            slice_w = slice(j * sw, j * sw + kw)

            if mode == 'max':
                for n in range(m):
                    for k in range(c_new):
                        a_prev_slice = A_prev[n, slice_h, slice_w, k]
                        mask = (a_prev_slice == np.max(a_prev_slice))
                        dA_prev[n, slice_h, slice_w, k] += (
                            mask * dA[n, i, j, k]
                        )
            elif mode == 'avg':
                avg_dA = dA[:, i:i + 1, j:j + 1, :] / (kh * kw)
                dA_prev[:, slice_h, slice_w, :] += avg_dA

    return dA_prev
