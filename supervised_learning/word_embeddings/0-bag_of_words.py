#!/usr/bin/env python3
"""Module to create a Bag of Words embedding matrix."""

import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """Creates a bag of words embedding matrix.

    Args:
        sentences: list of sentences to analyze.
        vocab: list of the vocabulary words to use for the analysis.
               If None, all words within sentences should be used.

    Returns:
        embeddings: numpy.ndarray of shape (s, f) containing the embeddings.
        features: list of the features used for embeddings.
    """
    tokenized_sentences = []
    for sentence in sentences:
        # Preprocessing: convert to lower case and remove punctuation/apostrophes
        words = re.findall(r"\b\w+\b", sentence.lower())
        tokenized_sentences.append(words)

    if vocab is None:
        features = set()
        for words in tokenized_sentences:
            features.update(words)
        features = sorted(list(features))
    else:
        features = list(vocab)

    feature_map = {word: i for i, word in enumerate(features)}
    s = len(sentences)
    f = len(features)

    embeddings = np.zeros((s, f), dtype=int)

    for i, words in enumerate(tokenized_sentences):
        for word in words:
            if word in feature_map:
                embeddings[i, feature_map[word]] += 1

    return embeddings, features
