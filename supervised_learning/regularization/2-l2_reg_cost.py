#!/usr/bin/env python3
"""Calculates L2 regularization cost for a Keras model"""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """Calculates total cost with L2 regularization"""
    return cost + model.losses
