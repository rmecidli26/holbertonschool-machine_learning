#!/usr/bin/env python3
"""Variational Autoencoder implementation using TensorFlow Keras.
"""

import tensorflow.keras as K
import tensorflow as tf


class Sampling(K.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a variational autoencoder."""
    # --- Encoder ---
    input_enc = K.Input(shape=(input_dims,))
    x = input_enc
    for units in hidden_layers:
        x = K.layers.Dense(units, activation='relu')(x)

    z_mean = K.layers.Dense(latent_dims, activation=None)(x)
    z_log_var = K.layers.Dense(latent_dims, activation=None)(x)
    z = Sampling()([z_mean, z_log_var])

    encoder = K.Model(input_enc, [z, z_mean, z_log_var], name='encoder')

    # --- Decoder ---
    input_dec = K.Input(shape=(latent_dims,))
    x = input_dec
    reversed_layers = hidden_layers[::-1]
    for units in reversed_layers:
        x = K.layers.Dense(units, activation='relu')(x)

    output_dec = K.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = K.Model(input_dec, output_dec, name='decoder')

    # --- Full Autoencoder ---
    auto_input = K.Input(shape=(input_dims,))
    z_sampled, z_m, z_lv = encoder(auto_input)
    reconstructed = decoder(z_sampled)
    auto = K.Model(auto_input, reconstructed, name='autoencoder')

    # VAE Loss: Reconstruction Loss (Binary Crossentropy) + KL Divergence
    reconstruction_loss = K.losses.binary_crossentropy(auto_input,
                                                      reconstructed)
    reconstruction_loss *= input_dims
    kl_loss = 1 + z_lv - K.backend.square(z_m) - K.backend.exp(z_lv)
    kl_loss = K.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.backend.mean(reconstruction_loss + kl_loss)

    auto.add_loss(vae_loss)
    auto.compile(optimizer='adam')

    return encoder, decoder, auto
