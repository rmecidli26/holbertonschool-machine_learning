#!/usr/bin/env python3
"""
Confusion matrix creation module.
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Cvnijsdfbvdbvuhdbuveuv
    """
    # labels.T -> (classes, m) ve logits -> (m, classes)
    # Matris çarpımı sonucu (classes, classes) boyutunda konfüzyon matrisi üretir.
    return np.dot(labels.T, logits)
