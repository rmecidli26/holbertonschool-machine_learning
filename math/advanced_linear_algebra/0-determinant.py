#!/usr/bin/env python3
import numpy as np

def determinant(matrix):
    try:
        arr = np.array(matrix)
        
        
        if arr.size == 0:
            return "This matrix is empty!"
        
        
        if arr.ndim < 2 or arr.shape[0] != arr.shape[1]:
            return "This matrix is not square"
            
        return np.linalg.det(arr)
    except Exception as e:
        return f"System error: {e}"
