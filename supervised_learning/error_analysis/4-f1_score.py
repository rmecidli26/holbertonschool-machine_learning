#!/usr/bin/env python3
"""
F1 score calculation module.
"""
import numpy as np

sensitivity = __import__("1-sensitivity").sensitivity
precision = __import__("2-precision").precision


def f1_score(confusion):
    """Calculates F1 score for each class."""
    p = precision(confusion)
    r = sensitivity(confusion)  # recall

    # F1 = 2 * (Precision * Recall)call)
    return 2 * (p * r) / (p + r)
