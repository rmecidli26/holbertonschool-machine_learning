#!/usr/bin/env python3
"""
Matrisin tərsini (inverse) hesablayan modul.
"""


def determinant(matrix):
    """
    Kvadrat matrisin determinantını hesablayır.
    """
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for j in range(n):
        sub_matrix = [row[:j] + row[j+1:] for row in matrix[1:]]
        det += ((-1) ** j) * matrix[0][j] * determinant(sub_matrix)
    return det


def cofactor(matrix):
    """
    Matrisin kofaktorlar matrisini hesablayır.
    """
    n = len(matrix)
    if n == 1:
        return [[1]]

    cofactor_matrix = []
    for i in range(n):
        row_cofactors = []
        for j in range(n):
            sub_matrix = [row[:j] + row[j+1:]
                          for row in (matrix[:i] + matrix[i+1:])]
            sign = (-1) ** (i + j)
            row_cofactors.append(sign * determinant(sub_matrix))
        cofactor_matrix.append(row_cofactors)
    return cofactor_matrix


def inverse(matrix):
    """
    Siyahılardan ibarət siyahı formatında olan matrisin tərsini hesablayır.
    """
    if not isinstance(matrix, list) or not all(isinstance(row, list)
                                               for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 0 or not all(len(row) == len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Öncə determinantı hesablayırıq
    det = determinant(matrix)

    # Əgər determinant 0-dırsa, matris sinqulyardır və tərsi yoxdur
    if det == 0:
        return None

    n = len(matrix)
    # 1x1 matris üçün xüsusi hal
    if n == 1:
        return [[1 / matrix[0][0]]]

    # Kofaktorlar matrisini alırıq
    cof_matrix = cofactor(matrix)

    # Adjuqat (kofaktorun transponirəsi) və determinantın bölünməsi
    inv_matrix = []
    for j in range(n):
        new_row = []
        for i in range(n):
            # Transponirə edərək hər elementi determinanta bölürük
            new_row.append(cof_matrix[i][j] / det)
        inv_matrix.append(new_row)

    return inv_matrix
