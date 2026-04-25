#!/usr/bin/env python3
"""
Bu modul matrislərin ölçüsünü hesablamaq üçün funksiyanı ehtiva edir.
Daxil olan siyahının (list) hər bir dərinliyini ölçür.
"""


def matrix_shape(matrix):
    """
    Verilmiş matrisin ölçülərini (shape) bir siyahı kimi qaytarır.
    Arqumentlər: matrix (list)
    Qaytarır: list of integers
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        if not matrix:
            break
        matrix = matrix[0]
    return shape
