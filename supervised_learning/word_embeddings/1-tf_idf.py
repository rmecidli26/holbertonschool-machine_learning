#!/usr/bin/env python3
"""Module to calculate TF-IDF embeddings for a list of sentences."""

import numpy as np


def tf_idf(sentences, vocab=None):
    """Creates a TF-IDF embedding matrix.

    Args:
        sentences: list of sentences to analyze.
        vocab: list of the vocabulary words to use for the analysis.
               If None, all words within sentences should be used.

    Returns:
        embeddings: numpy.ndarray of shape (s, f) containing TF-IDF scores.
        features: numpy.ndarray of the features used for embeddings.
    """
    tokenized_sentences = []
    for sentence in sentences:
        clean_sentence = sentence.lower().replace("'s", "")
        words = []
        for word in clean_sentence.split():
            clean_word = "".join(c for c in word if c.isalnum())
            if clean_word:
                words.append(clean_word)
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

    tf = np.zeros((s, f), dtype=float)
    df = np.zeros(f, dtype=float)

    for i, words in enumerate(tokenized_sentences):
        if not words:
            continue
        word_counts = {}
        for word in words:
            if word in feature_map:
                word_counts[word] = word_counts.get(word, 0) + 1

        total_words = len(words)
        for word, count in word_counts.items():
            col_idx = feature_map[word]
            tf[i, col_idx] = count / total_words

        for word in word_counts:
            col_idx = feature_map[word]
            df[col_idx] += 1

    idf = np.log10(s / (df + 1e-16))

    tf_idf_matrix = tf * idf

    norms = np.linalg.norm(tf_idf_matrix, axis=1, keepdims=True)
    embeddings = np.divide(
        tf_idf_matrix,
        norms,
        out=np.zeros_like(tf_idf_matrix),
        where=norms != 0,
    )

    return embeddings, np.array(features)
