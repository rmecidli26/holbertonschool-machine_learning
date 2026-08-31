#!/usr/bin/env python3
"""Module to convert a Gensim Word2Vec model to a Keras Embedding layer."""

import keras


def gensim_to_keras(model):
    """Converts a Gensim Word2Vec model to a Keras Embedding layer.

    Args:
        model: trained Gensim Word2Vec model.

    Returns:
        The trainable Keras Embedding layer.
    """
    weights = model.wv.vectors
    vector_size = model.wv.vector_size

    layer = keras.layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=vector_size,
        weights=[weights],
        trainable=True,
    )

    return layer
