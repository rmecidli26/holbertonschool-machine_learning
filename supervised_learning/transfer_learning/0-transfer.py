#!/usr/bin/env python3
"""Transfer learning model"""

from tensorflow import keras as K


def preprocess_data(X, Y):
    """Pre-processes the CIFAR-10 data for the model
    Returns:
        X_p: preprocessed X
        Y_p: preprocessed Y
    """
    # Preprocess inputs using DenseNet's specific preprocessing logic
    X_p = K.applications.densenet.preprocess_input(X)
    # One-hot encode the target labels
    Y_p = K.utils.to_categorical(Y, num_classes=10)

    return X_p, Y_p


def train_cifar10():
    """Trains a CNN on CIFAR-
    """
    # Load dataset
    (X_train, Y_train), (X_valid, Y_valid) = K.datasets.cifar10.load_data()

    # Preprocess data
    X_train_p, Y_train_p = preprocess_data(X_train, Y_train)
    X_valid_p, Y_valid_p = preprocess_data(X_valid, Y_valid)

    # Define input layer
    inputs = K.Input(shape=(32, 32, 3))

    # Lambda layer to resize images to (224, 224)
    resized_inputs = K.layers.Lambda(
        lambda image: K.backend.resize_images(
            image, height_factor=7, width_factor=7,
            data_format='channels_last'
        )
    )(inputs)

    # Base model from Keras Applications with pre-trained ImageNet weights
    base_model = K.applications.DenseNet121(
        include_top=False,
        weights='imagenet',
        input_tensor=resized_inputs
    )

    # Freeze base model layers initially
    base_model.trainable = False

    # Build custom classification top
    X = base_model.output
    X = K.layers.GlobalAveragePooling2D()(X)
    X = K.layers.BatchNormalization()(X)
    X = K.layers.Dense(256, activation='relu')(X)
    X = K.layers.Dropout(0.3)(X)
    outputs = K.layers.Dense(10, activation='softmax')(X)

    model = K.Model(inputs=inputs, outputs=outputs)

    # Compile model for initial training phase
    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks for learning rate decay and best weight saving
    callbacks = [
        K.callbacks.ModelCheckpoint(
            filepath='cifar10.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        K.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.2,
            patience=2,
            verbose=1
        )
    ]

    # Train top layers (transfer learning phase)
    model.fit(
        X_train_p, Y_train_p,
        batch_size=64,
        epochs=5,
        validation_data=(X_valid_p, Y_valid_p),
        callbacks=callbacks
    )

    # Fine-tuning phase: Unfreeze base model layers
    base_model.trainable = True

    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Fine-tune the full architecture
    model.fit(
        X_train_p, Y_train_p,
        batch_size=64,
        epochs=5,
        validation_data=(X_valid_p, Y_valid_p),
        callbacks=callbacks
    )


if __name__ == '__main__':
    train_cifar10()
