#!/usr/bin/env python3
import numpy as np

def determinant(matrix):
    try:
        # Siyahını NumPy massivinə çeviririk
        arr = np.array(matrix)

        # Matrisin boş olub-olmadığını yoxlayırıq
        if arr.size == 0:
            return "This matrix is empty!"

        # Matrisin 2 ölçülü və kvadrat olduğunu yoxlayırıq
        if arr.ndim < 2 or arr.shape[0] != arr.shape[1]:
            return "This matrix is not square"

        # Determinantı hesablayıb qaytarırıq
        return np.linalg.det(arr)
    except Exception as e:
        # Hər hansı sistem xətası baş verərsə tuturuq
        return f"System error: {e}"
