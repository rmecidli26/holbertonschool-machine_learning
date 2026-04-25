#!/usr/bin/env python3
"""
Bu modul NumPy massivləri ilə matris vurulmasını yerinə yetirən
funksiyanı ehtiva edir.
"""
import numpy as np


def np_matmul(mat1, mat2):
    """
    İki NumPy massivini (matrisi) bir-birinə vurur.

    Arqumentlər:
        mat1: Birinci numpy.ndarray.
        mat2: İkinci numpy.ndarray.

    rt
        Hasil olan yeni numpy.ndarray.
    """
    return np.matmul(mat1, mat2)
