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
        sentence_words = set()
        for word in words:
            if word in feature_map:
                col_idx = feature_map[word]
                tf[i, col_idx] += 1
                sentence_words.add(col_idx)

        for col_idx in sentence_words:
            df[col_idx] += 1

    # Standard natural log without smoothing: ln(s / df)
    idf = np.zeros(f, dtype=float)
    nonzero_df = df > 0
    idf[nonzero_df] = np.log(s / df[nonzero_df])

    tf_idf_matrix = tf * idf

    # L2 Normalization per row
    norms = np.linalg.norm(tf_idf_matrix, axis=1, keepdims=True)
    embeddings = np.divide(
        tf_idf_matrix,
        norms,
        out=np.zeros_like(tf_idf_matrix),
        where=norms != 0,
    )

    return embeddings, np.array(features)
