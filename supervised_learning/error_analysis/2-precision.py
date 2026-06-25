#!/usr/bin/env python3
"""
Precision calculation module.
"""
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each
    """
    # Matrisin köşegen eleman) verir.
    tp = np.diag(confusion)

    # Süturir (TP + FP).
    predicted_positives = np.sum(confusion, axis=0)

    # Precision = TP / (TP + FP)
    return tp / predicted_positives
