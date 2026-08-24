#!/usr/bin/env python3
"""
Optimizing a Machine Learning model using GPyOpt and Bayesian Optimization.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import GPyOpt


def build_and_train_model(hyperparameters):
    """
    Builds, trains, and evaluates a neural network based on the given
    hyperparameters.
    """
    # Extract hyperparameters from GPyOpt 2D array
    learning_rate = float(hyperparameters[0, 0])
    units = int(hyperparameters[0, 1])
    dropout_rate = float(hyperparameters[0, 2])
    l2_weight = float(hyperparameters[0, 3])
    batch_size = int(hyperparameters[0, 4])

    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((-1, 784)) / 255.0
    x_test = x_test.reshape((-1, 784)) / 255.0

    # Build the model
    model = Sequential([
        Dense(units, activation='relu',
              kernel_regularizer=l2(l2_weight), input_shape=(784,)),
        Dropout(dropout_rate),
        Dense(10, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Setup callbacks
    checkpoint_name = (f"checkpoint_lr_{learning_rate:.4f}_units_{units}_"
                       f"drop_{dropout_rate:.2f}_l2_{l2_weight:.4f}_"
                       f"bs_{batch_size}.h5")

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=3,
                                   restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(filepath=checkpoint_name,
                                       monitor='val_accuracy',
                                       save_best_only=True)

    # Train the model
    history = model.fit(x_train, y_train,
                        epochs=10,
                        batch_size=batch_size,
                        validation_data=(x_test, y_test),
                        callbacks=[early_stopping, model_checkpoint],
                        verbose=0)

    # We want to maximize validation accuracy. Since GPyOpt minimizes the
    # objective function by default, we return negative accuracy.
    best_val_accuracy = max(history.history['val_accuracy'])
    return -best_val_accuracy


if __name__ == '__main__':
    # Define the hyperparameter space (5 hyperparameters)
    bounds = [
        {'name': 'learning_rate', 'type': 'continuous',
         'domain': (1e-4, 1e-1)},
        {'name': 'units', 'type': 'discrete',
         'domain': (32, 64, 128, 256, 512)},
        {'name': 'dropout_rate', 'type': 'continuous',
         'domain': (0.0, 0.5)},
        {'name': 'l2_weight', 'type': 'continuous',
         'domain': (1e-5, 1e-2)},
        {'name': 'batch_size', 'type': 'discrete',
         'domain': (32, 64, 128)}
    ]

    # Initialize Bayesian Optimization
    bayesian_opt = GPyOpt.methods.BayesianOptimization(
        f=build_and_train_model,
        domain=bounds,
        model_type='GP',
        acquisition_type='EI',
        exact_feval=False,
        maximize=False
    )

    # Run for a maximum of 30 iterations
    bayesian_opt.run_optimization(max_iter=30)

    # Save the convergence plot
    bayesian_opt.plot_convergence(filename='convergence.png')

    # Save the report
    with open('bayes_opt.txt', 'w') as report_file:
        report_file.write("Bayesian Optimization Report\n")
        report_file.write("============================\n")
        report_file.write(f"Best objective value (Negative Accuracy): "
                          f"{bayesian_opt.fx_opt}\n")
        report_file.write(f"Best Validation Accuracy: "
                          f"{-bayesian_opt.fx_opt}\n\n")
        report_file.write("Best Hyperparameters:\n")
        report_file.write(f"- Learning Rate: {bayesian_opt.x_opt[0]:.6f}\n")
        report_file.write(f"- Units: {int(bayesian_opt.x_opt[1])}\n")
        report_file.write(f"- Dropout Rate: {bayesian_opt.x_opt[2]:.6f}\n")
        report_file.write(f"- L2 Weight: {bayesian_opt.x_opt[3]:.6f}\n")
        report_file.write(f"- Batch Size: {int(bayesian_opt.x_opt[4])}\n")

    print("Optimization finished. Results saved to 'bayes_opt.txt'.")
