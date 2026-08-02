#!/usr/bin/env python3
"""Module to create mini-batches from X and Y"""
import tensorflow as tf


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices synchronously

    Parameters:
        X: tf.Tensor or numpy.ndarray of shape (m, nx)
        Y: tf.Tensor or numpy.ndarray of shape (m, ny)

    Returns:
        X_shuffled, Y_shuffled as numpy arrays
    """
    m = X.shape[0]
    permutation = tf.random.shuffle(tf.range(m))
    return tf.gather(X, permutation), tf.gather(Y, permutation)


def mini_batches(X, Y, batch_size=64, seed=0):
    """
    Creates mini-batches from X and Y for mini-batch gradient descent

    Parameters:
        X: numpy.ndarray of shape (m, nx) representing input data
        Y: numpy.ndarray of shape (m, ny) representing labels
        batch_size: number of data points in a batch
        seed: seed for random number generator

    Returns:
        list of mini-batches containing (X_batch, Y_batch)
    """
    tf.random.set_seed(seed)
    X_shuffled, Y_shuffled = shuffle_data(X, Y)

    m = X.shape[0]
    mini_batches_list = []

    num_complete_batches = m // batch_size
    for k in range(num_complete_batches):
        X_batch = X_shuffled[k * batch_size:(k + 1) * batch_size]
        Y_batch = Y_shuffled[k * batch_size:(k + 1) * batch_size]
        mini_batches_list.append((X_batch, Y_batch))

    if m % batch_size != 0:
        X_batch = X_shuffled[num_complete_batches * batch_size:]
        Y_batch = Y_shuffled[num_complete_batches * batch_size:]
        mini_batches_list.append((X_batch, Y_batch))

    return mini_batches_list
