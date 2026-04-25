#!/usr/bin/env python3
"""
Bu modul NumPy massivlərini müəyyən ox üzrə birləşdirən funksiyanı ehtiva edir.
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    İki NumPy massivini göstərilən axis üzrə birləşdirir.

    Arqumentlər:
        mat1: Birinci numpy.ndarray.
        mat2: İkinci numpy.ndarray.
        axis: Birləşmə oxu (susmaya görə 0).

    Qaytarır:
        Birləşdirilmiş yeni numpy.ndarray.
    """
    return np.concatenate((mat1, mat2), axis=axis)
