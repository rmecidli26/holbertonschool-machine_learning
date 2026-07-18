#!/usr/bin/env python3
""" Keras Training with Model Checkpoint """
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
    save_best=False,
    filepath=None,
    verbose=True,
    shuffle=False
):
    """ Trains a model using mini-batch gradient descent with early stopping,

    learning rate decay, and saving the best model iteration
    """
    callbacks = []

    # 1. Erkən dayandırma (Early Stopping)
    if validation_data and early_stopping:
        callbacks.append(K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        ))

    # 2. Öyrənmə sürətinin azaldılması (Learning Rate Decay)
    if validation_data and learning_rate_decay:
        def lr_schedule(epoch):
            """ Calculates the learning rate based on inverse time decay """
            return alpha / (1 + decay_rate * epoch)

        callbacks.append(K.callbacks.LearningRateScheduler(
            schedule=lr_schedule,
            verbose=1
        ))

    # 3. Ən yaxşı modeli yadda saxlama (Model Checkpoint)
    if validation_data and save_best and filepath:
        callbacks.append(K.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor='val_loss',
            save_best_only=True,
            mode='min'
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
