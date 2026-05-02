#!/usr/bin/env python3
"""
Matrisin təyin olunmasını (definiteness) hesablayan modul.
"""
import numpy as np


def definiteness(matrix):
    """
    Kvadrat matrisin təyin olunma növünü müəyyən edir.
    """
    # Matrisin numpy.ndarray olub-olmadığını yoxlayırıq
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # Kvadrat matris və boş olub-olmadığını yoxlayırıq
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1] or \
       matrix.size == 0:
        return None

    # Simmetrik matris yoxlanışı (Definiteness adətən simmetrik matrislər üçündür)
    if not np.allclose(matrix, matrix.T):
        return None

    try:
        # Matrisin məxsusi qiymətlərini hesablayırıq
        eigenvalues = np.linalg.eigvals(matrix)

        pos = np.all(eigenvalues > 0)
        pos_semi = np.all(eigenvalues >= 0)
        neg = np.all(eigenvalues < 0)
        neg_semi = np.all(eigenvalues <= 0)

        if pos:
            return "Positive definite"
        elif neg:
            return "Negative definite"
        elif pos_semi:
            return "Positive semi-definite"
        elif neg_semi:
            return "Negative semi-definite"
        else:
            # Həm müsbət, həm də mənfi məxsusi qiymət varsa
            return "Indefinite"

    except Exception:
        # Hesablama zamanı xəta baş verərsə
        return None
