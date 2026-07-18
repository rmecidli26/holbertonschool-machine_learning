#!/usr/bin/env python3
""" Keras Functional Model """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ Builds a neural network with the Keras library using Functional API """
    # Giriş layını (Input tensor) təyin edirik
    inputs = K.Input(shape=(nx,))

    # L2 Requlyarizasiyası
    regularizer = K.regularizers.l2(lambtha)

    # İlk olaraq giriş tensorunu x dəyişəninə mənsub edirik
    x = inputs

    # Layları zəncirvari (Functional) şəkildə bir-birinə bağlayırıq
    for i in range(len(layers)):
        x = K.layers.Dense(
            units=layers[i],
            activation=activations[i],
            kernel_regularizer=regularizer
        )(x)

        # Sonuncu laydan başqa hər laydan sonra Dropout əlavə edirik
        if i < len(layers) - 1:
            x = K.layers.Dropout(rate=1 - keep_prob)(x)

    # Giriş və çıxışı birləşdirərək yekun Modeli yaradırıq
    model = K.Model(inputs=inputs, outputs=x)

    return model
