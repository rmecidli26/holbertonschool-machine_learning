#!/usr/bin/env python3
"""
Bitcoin price forecasting using RNN / Keras and tf.data.Dataset
"""

import tensorflow as tf
import numpy as np
import pandas as pd


def create_sequences(data, window_size=1440, horizon=60):
    """Creates rolling windows for time series forecasting."""
    X, y = [], []
    for i in range(len(data) - window_size - horizon + 1):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size + horizon - 1, 4])  # Predicting close price
    return np.array(X), np.array(y)


def build_model(input_shape):
    """Builds and compiles the RNN forecasting model."""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, activation='tanh', input_shape=input_shape, return_sequences=False),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model