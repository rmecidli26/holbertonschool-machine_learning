#!/usr/bin/python3
"""
Bu modul matrisin transpozunu hesablamaq üçün funksiyadir.
"""


def matrix_transpose(matrix):
    """
    2D matrisin transpozunu qaytarır.
    
    Arqumentlər:
        matrix: Siyahılardan ibarət 2D siyahı (list of lists).
        
    Qaytarır:
        Yeni transpoz edilmiş matris.
    """
    # Matrisin sütun sayı qədər dövr qurur və hər sütunu yeni sətir edir
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
