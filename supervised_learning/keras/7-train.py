#!/usr/bin/env python3
""" Keras Training with Learning Rate Decay """
import tensorflow.keras as K


def train_model(
    network,
    data,
    labels,
    batch_size,
    epochs,
    validation_data=None,
    early_stopping=False,
    patience=0,
    learning_rate_decay=False,
    alpha=0.1,
    decay_rate=1,
    verbose=True,
    shuffle=False
):
    """ Trains a model using mini-batch gradient descent with

    early stopping and learning rate decay
    """
    callbacks = []

    # 1. Erkən dayandırma (Early Stopping) yoxlanışı
    if validation_data and early_stopping:
        callbacks.append(K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        ))

    # 2. Öyrənmə sürətinin azaldılması (Learning Rate Decay) yoxlanışı
    if validation_data and learning_rate_decay:
        def lr_schedule(epoch):
            """ Calculates the learning rate based on inverse time decay """
            return alpha / (1 + decay_rate * epoch)

        callbacks.append(K.callbacks.LearningRateScheduler(
            schedule=lr_schedule,
            verbose=1
        ))

    # Modelin təlim edilməsi
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=callbacks
    )
    return history
