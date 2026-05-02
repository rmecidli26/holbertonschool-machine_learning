#!/usr/bin/env python3
"""
Matrisin kofaktorlar matrisini hesablayan modul.
"""


def determinant(matrix):
    """
    Kvadrat matrisin determinantını hesablayır.
    """
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for j in range(len(matrix)):
        sub_matrix = [row[:j] + row[j+1:] for row in matrix[1:]]
        det += ((-1) ** j) * matrix[0][j] * determinant(sub_matrix)
    return det


def cofactor(matrix):
    """
    Siyahılardan ibarət siyahı formatında olan matrisin kofaktorunu hesablayır.
    """
    if not isinstance(matrix, list) or not all(isinstance(row, list)
                                               for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 0 or not all(len(row) == len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)
    if n == 1:
        return [[1]]

    cofactor_matrix = []
    for i in range(n):
        row_cofactors = []
        for j in range(n):
            # Minor üçün sub-matrisi yaradırıq
            sub_matrix = [row[:j] + row[j+1:]
                          for row in (matrix[:i] + matrix[i+1:])]
            # Kofaktor işarəsini təyin edirik: (-1)^(i+j)
            sign = (-1) ** (i + j)
            row_cofactors.append(sign * determinant(sub_matrix))
        cofactor_matrix.append(row_cofactors)

    return cofactor_matrix
