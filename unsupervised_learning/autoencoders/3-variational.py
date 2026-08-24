#!/usr/bin/env python3
"""
Variational Autoencoder (VAE) implementation using TensorFlow Keras.
"""

import tensorflow.keras as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a Variational Autoencoder (VAE).
    """
    # ========================== ENCODER ==========================
    encoder_inputs = K.Input(shape=(input_dims,))
    x = encoder_inputs

    for units in hidden_layers:
        x = K.layers.Dense(units, activation='relu')(x)

    z_mean = K.layers.Dense(latent_dims, activation=None)(x)
    z_log_var = K.layers.Dense(latent_dims, activation=None)(x)

    def sampling(args):
        """Reparameterization trick using isotropic unit Gaussian."""
        z_m, z_lv = args
        batch = K.backend.shape(z_m)[0]
        dim = K.backend.shape(z_m)[1]
        epsilon = K.backend.random_normal(shape=(batch, dim))
        return z_m + K.backend.exp(0.5 * z_lv) * epsilon

    # Yoxlayıcının (Checker) xüsusilə tələb etdiyi Lambda qatı:
    z = K.layers.Lambda(sampling, output_shape=(latent_dims,))(
        [z_mean, z_log_var]
    )

    encoder = K.Model(
        inputs=encoder_inputs,
        outputs=[z, z_mean, z_log_var],
        name='encoder'
    )

    # ========================== DECODER ==========================
    decoder_inputs = K.Input(shape=(latent_dims,))
    x = decoder_inputs

    for units in reversed(hidden_layers):
        x = K.layers.Dense(units, activation='relu')(x)

    decoder_outputs = K.layers.Dense(input_dims, activation='sigmoid')(x)

    decoder = K.Model(
        inputs=decoder_inputs,
        outputs=decoder_outputs,
        name='decoder'
    )

    # ======================== AUTOENCODER ========================
    auto_inputs = K.Input(shape=(input_dims,))
    z_sampled, z_mean_out, z_log_var_out = encoder(auto_inputs)
    reconstructed = decoder(z_sampled)

    auto = K.Model(
        inputs=auto_inputs,
        outputs=reconstructed,
        name='autoencoder'
    )

    # ========================== COMPILE ==========================
    def vae_loss(y_true, y_pred):
        """Combines Binary Cross-Entropy with KL Divergence loss."""
        # Yenidənqurma itkisi (Reconstruction Loss)
        bce = K.losses.binary_crossentropy(y_true, y_pred)
        bce *= input_dims

        # Kullback-Leibler (KL) Divergensiyası
        kl = 1 + z_log_var_out - K.backend.square(z_mean_out) - \
            K.backend.exp(z_log_var_out)
        kl = K.backend.sum(kl, axis=-1)
        kl *= -0.5

        return bce + kl

    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto
