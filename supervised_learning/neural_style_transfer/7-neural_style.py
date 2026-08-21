#!/usr/bin/env python3
"""Neural Style Transfer Class Module."""
import numpy as np
import tensorflow as tf


class NST:
    """Performs tasks for Neural Style Transfer."""

    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initialize Neural Style Transfer parameters and images.

        Args:
            style_image (np.ndarray): Image used as style reference
            content_image (np.ndarray): Image used as content reference
            alpha (float/int): Weight for content cost
            beta (float/int): Weight for style cost
        """
        if not isinstance(style_image, np.ndarray) or \
           style_image.ndim != 3 or style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(content_image, np.ndarray) or \
           content_image.ndim != 3 or content_image.shape[2] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(alpha, (int, float)) or \
           isinstance(alpha, bool) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or \
           isinstance(beta, bool) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """Rescales image pixel values to [0, 1] with max side 512.

        Args:
            image (np.ndarray): Image array with shape (h, w, 3)

        Returns:
            tf.Tensor: Scaled image tensor of shape (1, h_new, w_new, 3)
        """
        if not isinstance(image, np.ndarray) or \
           image.ndim != 3 or image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w, _ = image.shape
        if h > w:
            h_new = 512
            w_new = int(w * (512 / h))
        else:
            w_new = 512
            h_new = int(h * (512 / w))

        image = tf.expand_dims(image, axis=0)
        resized_image = tf.image.resize(
            image,
            size=[h_new, w_new],
            method=tf.image.ResizeMethod.BICUBIC
        )
        scaled_image = resized_image / 255.0
        return tf.clip_by_value(scaled_image, 0.0, 1.0)

    def load_model(self):
        """Creates the model used to calculate Neural Style Transfer costs."""
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )

        x = vgg.input
        outputs = []
        self.output_names = []

        for layer in vgg.layers[1:]:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                layer = tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    name=layer.name
                )
            layer.trainable = False
            x = layer(x)
            if layer.name in self.style_layers or \
               layer.name == self.content_layer:
                outputs.append(x)
                self.output_names.append(layer.name)

        model = tf.keras.Model(inputs=vgg.input, outputs=outputs)
        model.trainable = False
        self.model = model

    @staticmethod
    def gram_matrix(input_layer):
        """Calculates the Gram matrix of a layer output.

        Args:
            input_layer (tf.Tensor|tf.Variable): Rank-4 tensor of shape
                (1, h, w, c)

        Returns:
            tf.Tensor: Gram matrix of shape (1, c, c)
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or \
           input_layer.shape.ndims != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        channels = tf.shape(input_layer)[-1]
        height = tf.cast(tf.shape(input_layer)[1], tf.float32)
        width = tf.cast(tf.shape(input_layer)[2], tf.float32)

        gram = tf.linalg.einsum('bijc,bijd->bcd', input_layer, input_layer)
        return gram / (height * width)

    def generate_features(self):
        """Extracts features used to calculate neural style cost."""
        style_preprocessed = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255.0
        )
        content_preprocessed = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255.0
        )

        style_outputs = self.model(style_preprocessed)
        content_outputs = self.model(content_preprocessed)

        outputs_dict_style = dict(zip(self.output_names, style_outputs))
        outputs_dict_content = dict(zip(self.output_names, content_outputs))

        self.gram_style_features = [
            self.gram_matrix(outputs_dict_style[layer])
            for layer in self.style_layers
        ]
        self.content_feature = outputs_dict_content[self.content_layer]

    def layer_style_cost(self, style_output, gram_target):
        """Calculates the style cost for a single layer.

        Args:
            style_output (tf.Tensor|tf.Variable): Layer style output tensor of
                generated image with shape (1, h, w, c)
            gram_target (tf.Tensor|tf.Variable): Gram matrix of target style
                output with shape (1, c, c)

        Returns:
            tf.Tensor: Calculated layer style cost scalar
        """
        if not isinstance(style_output, (tf.Tensor, tf.Variable)) or \
           style_output.shape.ndims != 4:
            raise TypeError("style_output must be a tensor of rank 4")

        c = style_output.shape[-1]
        if not isinstance(gram_target, (tf.Tensor, tf.Variable)) or \
           gram_target.shape != (1, c, c):
            raise TypeError(
                f"gram_target must be a tensor of shape [1, {c}, {c}]"
            )

        gram_style = self.gram_matrix(style_output)
        return tf.reduce_mean(tf.square(gram_style - gram_target))

    def style_cost(self, style_outputs):
        """Calculates the total style cost for the generated image.

        Args:
            style_outputs (list): List of style layer outputs (tf.Tensor)
                for the generated image.

        Returns:
            tf.Tensor: Total style cost scalar.
        """
        num_layers = len(self.style_layers)
        if not isinstance(style_outputs, list) or \
           len(style_outputs) != num_layers:
            raise TypeError(
                f"style_outputs must be a list with a length of {num_layers}"
            )

        weight = 1.0 / num_layers
        total_style_cost = 0.0

        for style_output, gram_target in zip(
            style_outputs, self.gram_style_features
        ):
            cost = self.layer_style_cost(style_output, gram_target)
            total_style_cost += weight * cost

        return total_style_cost

    def content_cost(self, content_output):
        """Calculates the content cost for the generated image.

        Args:
            content_output (tf.Tensor|tf.Variable): Content output tensor for
                the generated image.

        Returns:
            tf.Tensor: Content cost scalar.
        """
        s = self.content_feature.shape
        if not isinstance(content_output, (tf.Tensor, tf.Variable)) or \
           content_output.shape != s:
            raise TypeError(
                f"content_output must be a tensor of shape {s}"
            )

        return tf.reduce_mean(tf.square(content_output - self.content_feature))

    def total_cost(self, generated_image):
        """Calculates the total cost for the generated image.

        Args:
            generated_image (tf.Tensor|tf.Variable): Tensor of shape
                (1, nh, nw, 3) representing the generated image.

        Returns:
            tuple: (J, J_content, J_style)
        """
        s = self.content_image.shape
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)) or \
           generated_image.shape != s:
            raise TypeError(
                f"generated_image must be a tensor of shape {s}"
            )

        gen_preprocessed = tf.keras.applications.vgg19.preprocess_input(
            generated_image * 255.0
        )
        gen_outputs = self.model(gen_preprocessed)
        outputs_dict = dict(zip(self.output_names, gen_outputs))

        style_outputs = [
            outputs_dict[layer] for layer in self.style_layers
        ]
        content_output = outputs_dict[self.content_layer]

        J_content = self.content_cost(content_output)
        J_style = self.style_cost(style_outputs)

        J = (self.alpha * J_content) + (self.beta * J_style)

        return J, J_content, J_style
