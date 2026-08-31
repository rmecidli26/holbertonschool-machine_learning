#!/usr/bin/env python3
"""Module to calculate the n-gram BLEU score for a sentence."""

import numpy as np


def ngram_bleu(references, sentence, n):
    """Calculates the n-gram BLEU score for a sentence.

    Args:
        references: list of reference translations, each reference is a list
                   of words.
        sentence: list containing the model proposed sentence.
        n: size of the maximum n-gram to use for evaluation.

    Returns:
        The n-gram BLEU score.
    """
    c = len(sentence)

    # Find closest reference length r
    # If tie, pick the shorter reference length
    ref_lens = [len(ref) for ref in references]
    closest_ref_len = min(ref_lens, key=lambda r: (abs(r - c), r))

    # Calculate Brevity Penalty
    if c > closest_ref_len:
        bp = 1.0
    else:
        bp = np.exp(1 - closest_ref_len / c)

    precisions = []

    for k in range(1, n + 1):
        if c < k:
            precisions.append(0)
            continue

        # Generate k-grams for sentence
        sent_ngrams = {}
        for i in range(c - k + 1):
            ngram = tuple(sentence[i:i + k])
            sent_ngrams[ngram] = sent_ngrams.get(ngram, 0) + 1

        # Generate max k-gram counts across references
        max_ref_ngrams = {}
        for ref in references:
            ref_len = len(ref)
            if ref_len < k:
                continue
            current_ref_ngrams = {}
            for i in range(ref_len - k + 1):
                ngram = tuple(ref[i:i + k])
                current_ref_ngrams[ngram] = (
                    current_ref_ngrams.get(ngram, 0) + 1
                )
            for ngram, count in current_ref_ngrams.items():
                max_ref_ngrams[ngram] = max(
                    max_ref_ngrams.get(ngram, 0), count
                )

        # Calculate clipped count for k-grams
        clipped_count = 0
        total_ngrams = sum(sent_ngrams.values())

        for ngram, count in sent_ngrams.items():
            max_count = max_ref_ngrams.get(ngram, 0)
            clipped_count += min(count, max_count)

        if total_ngrams > 0:
            precisions.append(clipped_count / total_ngrams)
        else:
            precisions.append(0)

    if any(p == 0 for p in precisions):
        return 0.0

    # Geometric mean of precisions
    log_precision_sum = sum(np.log(p) for p in precisions)
    geo_mean = np.exp(log_precision_sum / n)

    return float(bp * geo_mean)
