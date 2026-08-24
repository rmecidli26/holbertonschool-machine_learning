#!/usr/bin/env python3
"""Sparse Autoencoder implementation using TensorFlow Keras.
"""

import tensorflow.keras as K


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """Creates a sparse autoencoder with L1"""
    # --- Encoder ---
    input_enc = K.Input(shape=(input_dims,))
    x = input_enc
    for units in hidden_layers:
        x = K.layers.Dense(units, activation='relu')(x)

    # Latent qatına L1 regularizasiyası (aktivliyə
    latent = K.layers.Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=K.regularizers.l1(lambtha)
    )(x)
    encoder = K.Model(input_enc, latent, name='encoder')

    # --- Decoder ---
    input_dec = K.Input(shape=(latent_dims,))
    x = input_dec
    reversed_layers = hidden_layers[::-1]
    for units in reversed_layers:
        x = K.layers.Dense(units, activation='relu')(x)

    # Sonuncu qat sigmoid aktivasiya funksiyasından istifadə edir
    output_dec = K.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = K.Model(input_dec, output_dec, name='decoder')

    # --- Full Autoencoder ---
    auto_input = K.Input(shape=(input_dims,))
    latent_repr = encoder(auto_input)
    reconstructed = decoder(latent_repr)
    auto = K.Model(auto_input, reconstructed, name='autoencoder')

    # Modeli adam optimizatoru və binary_crossentropy
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
