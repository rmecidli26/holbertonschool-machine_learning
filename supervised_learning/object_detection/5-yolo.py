#!/usr/bin/env python3
"""
Contains the Yolo class
"""
import cv2
import numpy as np
import tensorflow.keras as K


class Yolo:
    """
    Yolo class to perform object detection
    """

    def __init__(self, model_path, classes_path, class_threshold, nms_threshold, anchors):
        """
        Class constructor
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_threshold = class_threshold
        self.nms_threshold = nms_threshold
        self.anchors = anchors

    def process_images(self, image_paths):
        """
        Preprocesses images for YOLO model
        """
        pimages = []
        image_shapes = []

        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        for path in image_paths:
            image = cv2.imread(path)
            image_shapes.append(image.shape[:2])

            resized = cv2.resize(
                image,
                (input_w, input_h),
                interpolation=cv2.INTER_CUBIC
            )

            pimage = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) / 255.0
            pimages.append(pimage)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes
