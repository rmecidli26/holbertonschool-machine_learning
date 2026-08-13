#!/usr/bin/env python3
"""Yolo class module"""

import tensorflow.keras as K
import numpy as np


class Yolo:
    """Yolo class for object detection using YOLO v3 model"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Class constructor

        parameters:
            model_path: path to where Keras model is stored
            classes_path: path to list of class names
            class_t: box score threshold for initial filtering
            nms_t: IOU threshold for non-max suppression
            anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2)
                     containing all anchor boxes
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Processes predictions from Darknet model for a single image.

        parameters:
            outputs: list of numpy.ndarrays containing predictions from the
                     Darknet model for a single image.
                     Shape: (grid_height, grid_width, anchor_boxes,
                             4 + 1 + classes)
            image_size: numpy.ndarray containing image's original size
                        [image_height, image_width]

        returns:
            tuple of (boxes, box_confidences, box_class_probs):
                boxes: list of numpy.ndarrays of shape
                       (grid_height, grid_width, anchor_boxes, 4)
                       containing boundary boxes relative to original image
                       (x1, y1, x2, y2)
                box_confidences: list of numpy.ndarrays of shape
                                 (grid_height, grid_width, anchor_boxes, 1)
                                 containing box confidences
                box_class_probs: list of numpy.ndarrays of shape
                                 (grid_height, grid_width, anchor_boxes,
                                  classes)
                                 containing class probabilities
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size
        input_width = self.model.input.shape[1]
        input_height = self.model.input.shape[2]

        def _sigmoid(x):
            """Sigmoid activation function"""
            return 1 / (1 + np.exp(-x))

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            # Extract prediction components
            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            confidences = _sigmoid(output[..., 4:5])
            class_probs = _sigmoid(output[..., 5:])

            box_confidences.append(confidences)
            box_class_probs.append(class_probs)

            # Create grid cell indices (c_x, c_y)
            c_x = np.tile(
                np.arange(0, grid_width), (grid_height, 1)
            )[:, :, np.newaxis]
            c_y = np.tile(
                np.arange(0, grid_height)[:, np.newaxis], (1, grid_width)
            )[:, :, np.newaxis]

            # Calculate center position relative to normalized image
            b_x = (_sigmoid(t_x) + c_x) / grid_width
            b_y = (_sigmoid(t_y) + c_y) / grid_height

            # Retrieve anchor dimensions for current scale
            p_w = self.anchors[i, :, 0]
            p_h = self.anchors[i, :, 1]

            # Calculate box width and height relative to normalized image
            b_w = (p_w * np.exp(t_w)) / input_width
            b_h = (p_h * np.exp(t_h)) / input_height

            # Convert to absolute bounding box coordinates (x1, y1, x2, y2)
            x1 = (b_x - b_w / 2) * image_width
            y1 = (b_y - b_h / 2) * image_height
            x2 = (b_x + b_w / 2) * image_width
            y2 = (b_y + b_h / 2) * image_height

            box = np.zeros((grid_height, grid_width, anchor_boxes, 4))
            box[..., 0] = x1
            box[..., 1] = y1
            box[..., 2] = x2
            box[..., 3] = y2

            boxes.append(box)

        return boxes, box_confidences, box_class_probs
