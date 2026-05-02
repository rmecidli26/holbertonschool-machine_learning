#!/usr/bin/env python3
"""
Matrisin determinantını hesablayan modul.
"""


def determinant(matrix):
    """
    Siyahılari hesablayır.
    """
    # Matrix-in siyahı olub-olmadığını yoxlayırıq
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    #tapşırığa uyğun 1)
    if matrix == [[]]:
        return 1

    # Daxili elementlərin siyahı olub-olmadığını yoxlayırıq
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Boş siyahı [] halını yoxlayırıq
    if len(matrix) == 0:
        return 1

    n = len(matrix)

    # Kvadrat matris yoxlanışı
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # 1x1 matris üçün
    if n == 1:
        return matrix[0][0]

    # 2x2 matris üçün (hesablamanı sürətləndirmək üçün)
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # NxN matris üçün rekursiv determinant hesablama (Laplas açılışı)
    det = 0
    for j in range(n):
        # Kofaktor hesablamaq üçün sub-matris (minor) yaradırıq
        sub_matrix = [row[:j] + row[j+1:] for row in matrix[1:]]
        # İşarəni və rekursiv nəticəni əlavə edirik
        det += ((-1) ** j) * matrix[0][j] * determinant(sub_matrix)

    return det
