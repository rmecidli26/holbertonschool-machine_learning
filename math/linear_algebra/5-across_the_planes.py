#!/usr/bin/env python3
"""
Bu modul iki 2D matrisin toplanması üçün funksiyanı ehtiva edir.
"""


def add_matrices2D(mat1, mat2):
    """
    İki 2D matrisi element-wise (qarşılıqlı elementlərini) toplayır.
    Əgər matrislərin ölçüləri (shape) eyni deyilsə, None qaytarır.
    """
    if len(mat1) != len(mat2):
        return None
    if len(mat1[0]) != len(mat2[0]):
        return None
    new_matrix = [
        [mat1[i][j] + mat2[i][j] for j in range(len(mat1[0]))]
        for i in range(len(mat1))
    ]
    return new_matrix
