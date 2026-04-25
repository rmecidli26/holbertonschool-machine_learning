#!/usr/bin/env python3
"""
Bu modul iki 2D matrisin vurulması üçün funksiyanı ehtiva edir.
"""


def mat_mul(mat1, mat2):
    """
    İki 2D matrisi bir-birinə vurur.
    Əgər matrislər vurula bilmirsə, None qaytarır.
    """
    # Vurma şərti: mat1-in sütun sayı == mat2-nin sətir sayı
    if len(mat1[0]) != len(mat2):
        return None
    result = [
        [
            sum(mat1[i][k] * mat2[k][j] for k in range(len(mat2)))
            for j in range(len(mat2[0]))
        ]
        for i in range(len(mat1))
    ]
    return result
