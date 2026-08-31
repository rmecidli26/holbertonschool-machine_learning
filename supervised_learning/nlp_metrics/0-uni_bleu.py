#!/usr/bin/env python3
"""Module to calculate the unigram BLEU score for a sentence."""

import numpy as np


def uni_bleu(references, sentence):
    """Calculates the unigram BLEU score for a sentence.

    Args:
        references: list of reference translations, each reference is a list
                   of words.
        sentence: list containing the model proposed sentence.

    Returns:
        The unigram BLEU score.
    """
    c = len(sentence)

    # Find closest reference length r
    ref_lens = [len(ref) for ref in references]
    closest_ref_len = min(ref_lens, key=lambda r: (abs(r - c), r))

    # Calculate Brevity Penalty
    if c > closest_ref_len:
        bp = 1.0
    else:
        bp = np.exp(1 - closest_ref_len / c)

    # Calculate Modified Unigram Precision
    words = set(sentence)
    clipped_count = 0

    for word in words:
        count_in_sentence = sentence.count(word)
        max_count_in_refs = max([ref.count(word) for ref in references])
        clipped_count += min(count_in_sentence, max_count_in_refs)

    precision = clipped_count / c

    return float(bp * precision)
