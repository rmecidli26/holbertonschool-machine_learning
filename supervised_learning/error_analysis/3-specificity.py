#!/usr/bin/env python3
"""
Specificity calculation module.
"""
import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for ix.
    """
    # Matristeki tüm elemanlri sayısı)
    total = np.sum(confusion)

    # Doğru Pozitifler (True Positives)
    tp = np.diag(confusion)

    # Satır toplamları:
    row_sums = np.sum(confusion, axis=1)

    # Sütun toplamları: H)
    col_sums = np.sum(confusion, axis=0)

    # Yanlış Pozitiler
    fp = col_sums - tp

    # Gerçek Negakliyoruz)
    tn = total - row_sums - col_sums + tp

    # Specificity = TN / (TN + FP)
    return tn / (tn + fp)
