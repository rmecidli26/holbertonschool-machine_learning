#!/usr/bin/env python3
"""
Sensitivity calculation module.
"""
import numpy as np


def sensitivity(confusion):
    """
    Canybybub
    """
    # Matrisin  Positives) verir.
    tp = np.diag(confusion)

    # Satırların toplamı, o sınıfa a).
    actual_positives = np.sum(confusion, axis=1)

    # Sensitivit
    return tp / actual_positives
