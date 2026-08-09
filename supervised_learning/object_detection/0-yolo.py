#!/usr/bin/env python3
"""
Contains the Yolo class for object detection using YOLO v3.
"""
import tensorflow as tf


class Yolo:
    """
    Uses the YOLO v3 algorithm to perform object detection.
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Class constructor for Yolo.

        Parameters:
        - model_path: str, path to Darknet Keras model
        - classes_path: str, path to class names list
        - class_t: float, box score threshold for initial filtering step
        - nms_t: float, IOU threshold for non-max suppression
        """
        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
