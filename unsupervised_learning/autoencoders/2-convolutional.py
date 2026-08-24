#!/usr/bin/env python3
"""Convolutional Autoencoder implementation using TensorFlow Keras.
"""

import tensorflow.keras as K


def autoencoder(input_dims, filters, latent_dims):
    """Creates a convolutional autoencoder."""
    input_enc = K.Input(shape=input_dims)
    x = input_enc
    for f in filters:
        x = K.layers.Conv2D(f, (3, 3), activation='relu', padding='same')(x)
        x = K.layers.MaxPooling2D((2, 2), padding='same')(x)
    encoder = K.Model(input_enc, x, name='encoder')

    input_dec = K.Input(shape=latent_dims)
    x = input_dec
    reversed_filters = filters[::-1]

    for i in range(len(reversed_filters) - 1):
        x = K.layers.Conv2D(reversed_filters[i], (3, 3), activation='relu',
                            padding='same')(x)
        x = K.layers.UpSampling2D((2, 2))(x)

    x = K.layers.Conv2D(reversed_filters[-1], (3, 3), activation='relu',
                        padding='valid')(x)
    x = K.layers.UpSampling2D((2, 2))(x)

    n_channels = input_dims[-1]
    output_dec = K.layers.Conv2D(n_channels, (3, 3), activation='sigmoid',
                                 padding='same')(x)
    decoder = K.Model(input_dec, output_dec, name='decoder')

    auto_input = K.Input(shape=input_dims)
    latent_repr = encoder(auto_input)
    reconstructed = decoder(latent_repr)
    auto = K.Model(auto_input, reconstructed, name='autoencoder')

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
