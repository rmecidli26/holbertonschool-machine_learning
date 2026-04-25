#!/usr/bin/env python3
"""
Bu modul bir NumPy massivinin ölçülərini (shape) 
hesablamaq üçün funksiyanı ehtiva edir.
"""


def np_shape(matrix):
    """
    Bir numpy.ndarray-in ölçülərini (shape) 
    bir tuple kimi qaytarır.
    
    Arqumentlər:
        matrix: Ölçüsü hesablanacaq numpy.ndarray.
        
    Qaytarır:
        Tam ədədlərdən ibarət tuple.
    """
    return matrix.shape
