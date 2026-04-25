#!/usr/bin/env python3
def matrix_shape(matrix):
    '''Bura bos olmalidir'''
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
