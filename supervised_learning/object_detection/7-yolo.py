#!/usr/bin/env python3
"""Yolo class module"""

import cv2
import glob
import os
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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters boundary boxes based on box scores and class threshold.

        parameters:
            boxes: list of numpy.ndarrays of shape
                   (grid_height, grid_width, anchor_boxes, 4)
            box_confidences: list of numpy.ndarrays of shape
                             (grid_height, grid_width, anchor_boxes, 1)
            box_class_probs: list of numpy.ndarrays of shape
                             (grid_height, grid_width, anchor_boxes, classes)

        returns:
            tuple of (filtered_boxes, box_classes, box_scores):
                filtered_boxes: numpy.ndarray of shape (?, 4)
                                containing filtered bounding boxes
                box_classes: numpy.ndarray of shape (?,)
                             containing class number for each box
                box_scores: numpy.ndarray of shape (?,)
                            containing box scores for each box
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for box, box_conf, box_prob in zip(
            boxes, box_confidences, box_class_probs
        ):
            # Calculate overall box scores (confidence * class probability)
            scores = box_conf * box_prob

            # Identify predicted class and max score per anchor box
            classes = np.argmax(scores, axis=-1)
            class_scores = np.max(scores, axis=-1)

            # Filter out boxes below the class threshold score
            mask = class_scores >= self.class_t

            filtered_boxes.append(box[mask])
            box_classes.append(classes[mask])
            box_scores.append(class_scores[mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Applies non-max suppression to filtered bounding boxes.

        parameters:
            filtered_boxes: numpy.ndarray of shape (?, 4)
                            containing filtered bounding boxes
            box_classes: numpy.ndarray of shape (?,)
                         containing class number for each box
            box_scores: numpy.ndarray of shape (?,)
                        containing box scores for each box

        returns:
            tuple of (box_predictions, predicted_box_classes,
                      predicted_box_scores):
                box_predictions: numpy.ndarray of shape (?, 4)
                                 containing predicted bounding boxes
                                 ordered by class and box score
                predicted_box_classes: numpy.ndarray of shape (?,)
                                       containing class number for
                                       box_predictions ordered by
                                       class and box score
                predicted_box_scores: numpy.ndarray of shape (?,)
                                      containing box scores for
                                      box_predictions ordered by
                                      class and box score
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        unique_classes = np.unique(box_classes)

        for cls in unique_classes:
            idx = np.where(box_classes == cls)[0]
            cls_boxes = filtered_boxes[idx]
            cls_scores = box_scores[idx]

            x1 = cls_boxes[:, 0]
            y1 = cls_boxes[:, 1]
            x2 = cls_boxes[:, 2]
            y2 = cls_boxes[:, 3]

            areas = (x2 - x1) * (y2 - y1)
            order = cls_scores.argsort()[::-1]

            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)

                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])

                w = np.maximum(0.0, xx2 - xx1)
                h = np.maximum(0.0, yy2 - yy1)

                inter = w * h
                iou = inter / (areas[i] + areas[order[1:]] - inter)

                inds = np.where(iou <= self.nms_t)[0]
                order = order[inds + 1]

            box_predictions.append(cls_boxes[keep])
            predicted_box_classes.append(np.full(len(keep), cls))
            predicted_box_scores.append(cls_scores[keep])

        if box_predictions:
            box_predictions = np.concatenate(box_predictions, axis=0)
            predicted_box_classes = np.concatenate(
                predicted_box_classes, axis=0
            )
            predicted_box_scores = np.concatenate(
                predicted_box_scores, axis=0
            )
        else:
            box_predictions = np.array([])
            predicted_box_classes = np.array([])
            predicted_box_scores = np.array([])

        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """
        Loads all images from a folder.

        parameters:
            folder_path: string representing the path to the folder
                         holding all images to load

        returns:
            tuple of (images, image_paths):
                images: list of images as numpy.ndarrays
                image_paths: list of paths to the individual images
        """
        image_paths = glob.glob(folder_path + '/*', recursive=False)
        if image_paths is not None:
            image_paths.sort()  # Sort to ensure consistent array output matching
            
        images = [cv2.imread(p) for p in image_paths]
        return images, image_paths

    def preprocess_images(self, images):
        """
        Preprocesses a list of images for the YOLO model.

        parameters:
            images: list of images as numpy.ndarrays

        returns:
            tuple of (pimages, image_shapes):
                pimages: numpy.ndarray of shape (ni, input_h, input_w, 3)
                         containing all of the preprocessed images
                image_shapes: numpy.ndarray of shape (ni, 2) containing the
                              original height and width of each image
        """
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        pimages = []
        image_shapes = []

        for image in images:
            image_shapes.append([image.shape[0], image.shape[1]])
            resized = cv2.resize(
                image, (input_w, input_h), interpolation=cv2.INTER_CUBIC
            )
            pimages.append(resized / 255.0)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Displays the image with all boundary boxes, class names,
        and box scores.

        parameters:
            image: a numpy.ndarray containing an unprocessed image
            boxes: a numpy.ndarray containing the boundary boxes for the image
            box_classes: a numpy.ndarray containing the class indices
                         for each box
            box_scores: a numpy.ndarray containing the box scores for each box
            file_name: the file path where the original image is stored
        """
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            class_name = self.class_names[box_classes[i]]
            score = round(box_scores[i], 2)
            text = "{} {}".format(class_name, score)

            cv2.putText(
                image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 1, cv2.LINE_AA
            )

        cv2.imshow(file_name, image)
        key = cv2.waitKey(0)

        if key == ord('s'):
            if not os.path.exists('detections'):
                os.makedirs('detections')

            save_path = os.path.join(
                'detections',
                os.path.basename(file_name) if ('/' in file_name or '\\' in file_name) else file_name
            )
            cv2.imwrite(save_path, image)

        cv2.destroyAllWindows()

    def predict(self, folder_path):
        """
        Performs object detection on all images in a folder.

        parameters:
            folder_path: a string representing the path to the folder
                         holding all images to predict

        returns:
            tuple of (predictions, image_paths):
                predictions: a list of tuples for each image of
                             (boxes, box_classes, box_scores)
                image_paths: a list of image paths corresponding to
                             each prediction in predictions
        """
        images, image_paths = self.load_images(folder_path)
        if not images:
            return [], []

        pimages, image_shapes = self.preprocess_images(images)
        model_outputs = self.model.predict(pimages)

        predictions = []

        for i, image_path in enumerate(image_paths):
            image_outputs = [output[i] for output in model_outputs]

            boxes, box_confidences, box_class_probs = self.process_outputs(
                image_outputs, image_shapes[i]
            )

            filtered_boxes, box_classes, box_scores = self.filter_boxes(
                boxes, box_confidences, box_class_probs
            )

            box_predictions, predicted_box_classes, predicted_box_scores = self.non_max_suppression(
                filtered_boxes, box_classes, box_scores
            )

            predictions.append(
                (box_predictions, predicted_box_classes, predicted_box_scores)
            )

            file_name = os.path.basename(image_path)
            self.show_boxes(
                images[i], box_predictions, predicted_box_classes,
                predicted_box_scores, file_name
            )

        return predictions, image_paths
