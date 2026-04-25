#!/usr/bin/env python3
"""
Bu modul NumPy massivləri üzərində element-wise (qarşılıqlı elementlər)
riyazi əməliyyatları yerinə yetirən funksiyanı ehtiva edir.
"""


def np_elementwise(mat1, mat2):
    """
    İki massivin cəmini, fərqini, hasilini və qismətini qaytarır.

    Arqumentlər:
        mat1: Birinci numpy.ndarray və ya skalyar.
        mat2: İkinci numpy.ndarray və ya skalyar.

    Qaytarır:
        (sum, difference, product, quotient) formatında tuple.
    """
    sum_res = mat1 + mat2
    diff_res = mat1 - mat2
    prod_res = mat1 * mat2
    quot_res = mat1 / mat2
    return (sum_res, diff_res, prod_res, quot_res)
