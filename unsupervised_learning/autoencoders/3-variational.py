#!/usr/bin/env python3
"""Variational Autoencoder (VAE) implementation using TensorFlow Keras."""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a Variational Autoencoder."""
    # Encoder
    input_enc = keras.Input(shape=(input_dims,))
    x = input_enc
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)

    z_mean = keras.layers.Dense(latent_dims, activation=None)(x)
    z_log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    def sampling(args):
        """Reparameterization trick."""
        z_m, z_lv = args
        batch = keras.backend.shape(z_m)[0]
        dim = keras.backend.shape(z_m)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_m + keras.backend.exp(0.5 * z_lv) * epsilon

    z = keras.layers.Lambda(sampling, output_shape=(latent_dims,))(
        [z_mean, z_log_var]
    )
    encoder = keras.Model(input_enc, [z, z_mean, z_log_var], name='encoder')

    # Decoder
    input_dec = keras.Input(shape=(latent_dims,))
    x = input_dec
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)

    output_dec = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(input_dec, output_dec, name='decoder')

    # Full Autoencoder
    auto_inputs = keras.Input(shape=(input_dims,))
    z_sampled, z_m_out, z_lv_out = encoder(auto_inputs)
    reconstructed = decoder(z_sampled)
    auto = keras.Model(auto_inputs, reconstructed, name='autoencoder')

    # KL Divergence - modelin itkisi kimi əlavə olunur
    kl_loss = -0.5 * keras.backend.sum(
        1 + z_lv_out - keras.backend.square(z_m_out) - keras.backend.exp(z_lv_out),
        axis=-1
    )
    auto.add_loss(keras.backend.mean(kl_loss))

    # Yoxlayıcının testindən (True) keçmək üçün birbaşa funksiya obyekti verilir:
    auto.compile(optimizer='adam', loss=keras.losses.binary_crossentropy)

    return encoder, decoder, auto
