#!/usr/bin/env python3
"""
Bu modul iki matrisi müəyyən ox üzrə birləşdirmək üçün funksiyanı ehtiva edir.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    İki 2D matrisi axis parametrindən asılı olaraq birləşdirir.
    Arqumentlər:
        mat1: Birinci 2D matris.
        mat2: İkinci 2D matris.
        axis: Birləşmə oxu (0 - şaquli, 1 - üfüqi).
    Qaytarır:
        Birləşdirilmiş yeni matris, ölçülər uyğun deyilsə None.
    """
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return [row[:] for row in mat1] + [row[:] for row in mat2]
    if axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [mat1[i] + mat2[i] for i in range(len(mat1))]
    return None
