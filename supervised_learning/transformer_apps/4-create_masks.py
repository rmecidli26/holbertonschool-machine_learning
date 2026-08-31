#!/usr/bin/env python3
"""Creates all masks for training/validation"""
import tensorflow as tf


def create_padding_mask(seq):
    """
    Creates a padding mask for a sequence
    seq is a tf.Tensor of shape (batch_size, seq_len)
    Returns: a tf.Tensor of shape (batch_size, 1, 1, seq_len)
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    """
    Creates a look ahead mask
    size is the size of the mask
    Returns: a tf.Tensor of shape (size, size)
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def create_masks(inputs, target):
    """
    Creates all masks for training/validation
    inputs is a tf.Tensor of shape (batch_size, seq_len_in) that
        contains the input sentence
    target is a tf.Tensor of shape (batch_size, seq_len_out) that
        contains the target sentence
    Returns: encoder_mask, combined_mask, decoder_mask
        encoder_mask is the tf.Tensor padding mask of shape
            (batch_size, 1, 1, seq_len_in) to be applied in the encoder
        combined_mask is the tf.Tensor of shape
            (batch_size, 1, seq_len_out, seq_len_out) used in the 1st
            attention block in the decoder to pad and mask future
            tokens in the input received by the decoder. It takes the
            maximum between a look ahead mask and the decoder target
            padding mask.
        decoder_mask is the tf.Tensor padding mask of shape
            (batch_size, 1, 1, seq_len_in) used in the 2nd attention
            block in the decoder.
    """
    encoder_mask = create_padding_mask(inputs)
    decoder_mask = create_padding_mask(inputs)

    seq_len_out = tf.shape(target)[1]
    look_ahead_mask = create_look_ahead_mask(seq_len_out)
    decoder_target_padding_mask = create_padding_mask(target)
    combined_mask = tf.maximum(
        decoder_target_padding_mask, look_ahead_mask
    )

    return encoder_mask, combined_mask, decoder_mask
