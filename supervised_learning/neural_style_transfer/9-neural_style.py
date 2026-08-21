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

    def __init__(self, style_image, content_image, alpha=1e4, beta=1, var=10):
        """Initialize Neural Style Transfer parameters and images.

        Args:
            style_image (np.ndarray): Image used as style reference
            content_image (np.ndarray): Image used as content reference
            alpha (float/int): Weight for content cost
            beta (float/int): Weight for style cost
            var (float/int): Weight for variational cost
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

        if not isinstance(var, (int, float)) or \
           isinstance(var, bool) or var < 0:
            raise TypeError("var must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.var = float(var)
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
        outputs_dict = {}

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
            outputs_dict[layer.name] = x

        outputs = [outputs_dict[layer] for layer in self.style_layers]
        outputs.append(outputs_dict[self.content_layer])

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

        self.gram_style_features = [
            self.gram_matrix(style_outputs[i])
            for i in range(len(self.style_layers))
        ]
        self.content_feature = content_outputs[-1]

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
        target_shape = self.content_feature.shape
        if not isinstance(content_output, (tf.Tensor, tf.Variable)) or \
           content_output.shape != target_shape:
            raise TypeError(
                f"content_output must be a tensor of shape {target_shape}"
            )

        return tf.reduce_mean(tf.square(content_output - self.content_feature))

    @staticmethod
    def variational_cost(generated_image):
        """Calculates the variational cost for the generated image.

        Args:
            generated_image (tf.Tensor|tf.Variable): Generated image tensor of
                shape (1, nh, nw, 3).

        Returns:
            tf.Tensor: Variational cost scalar.
        """
        return tf.reduce_sum(tf.image.total_variation(generated_image))

    def total_cost(self, generated_image):
        """Calculates the total cost for the generated image.

        Args:
            generated_image (tf.Tensor|tf.Variable): Tensor of shape
                (1, nh, nw, 3) representing the generated image.

        Returns:
            tuple: (J, J_content, J_style, J_var)
        """
        target_shape = self.content_image.shape
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)) or \
           generated_image.shape != target_shape:
            raise TypeError(
                f"generated_image must be a tensor of shape {target_shape}"
            )

        gen_preprocessed = tf.keras.applications.vgg19.preprocess_input(
            generated_image * 255.0
        )
        gen_outputs = self.model(gen_preprocessed)

        style_outputs = gen_outputs[:-1]
        content_output = gen_outputs[-1]

        J_content = self.content_cost(content_output)
        J_style = self.style_cost(style_outputs)
        J_var = self.variational_cost(generated_image)

        J = (self.alpha * J_content) + (self.beta * J_style) + \
            (self.var * J_var)

        return J, J_content, J_style, J_var

    def compute_grads(self, generated_image):
        """Calculates the gradients for the generated image.

        Args:
            generated_image (tf.Tensor|tf.Variable): Generated image tensor
                of shape (1, nh, nw, 3).

        Returns:
            tuple: (grads, J_total, J_content, J_style, J_var)
        """
        target_shape = self.content_image.shape
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)) or \
           generated_image.shape != target_shape:
            raise TypeError(
                f"generated_image must be a tensor of shape {target_shape}"
            )

        with tf.GradientTape() as tape:
            tape.watch(generated_image)
            J_total, J_content, J_style, J_var = self.total_cost(
                generated_image
            )

        grads = tape.gradient(J_total, generated_image)

        return grads, J_total, J_content, J_style, J_var

    def generate_image(self, iterations=1000, step=None, lr=0.01,
                       beta1=0.9, beta2=0.99):
        """Generates the neural style transferred image.

        Args:
            iterations (int): Number of iterations for gradient descent
            step (int|None): Iteration step for printing progress
            lr (float|int): Learning rate for Adam optimizer
            beta1 (float): Beta1 parameter for Adam optimizer
            beta2 (float): Beta2 parameter for Adam optimizer

        Returns:
            tuple: (generated_image, cost)
        """
        if not isinstance(iterations, int) or isinstance(iterations, bool):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be positive")

        if step is not None:
            if not isinstance(step, int) or isinstance(step, bool):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError(
                    "step must be positive and less than iterations"
                )

        if not isinstance(lr, (int, float)) or isinstance(lr, bool):
            raise TypeError("lr must be a number")
        if lr <= 0:
            raise ValueError("lr must be positive")

        if not isinstance(beta1, float):
            raise TypeError("beta1 must be a float")
        if beta1 < 0.0 or beta1 > 1.0:
            raise ValueError("beta1 must be in the range [0, 1]")

        if not isinstance(beta2, float):
            raise TypeError("beta2 must be a float")
        if beta2 < 0.0 or beta2 > 1.0:
            raise ValueError("beta2 must be in the range [0, 1]")

        generated_image = tf.Variable(self.content_image)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr,
            beta_1=beta1,
            beta_2=beta2
        )

        best_cost = float('inf')
        best_image = None

        for i in range(iterations + 1):
            grads, J_total, J_content, J_style, J_var = self.compute_grads(
                generated_image
            )

            current_cost = float(J_total.numpy())

            if current_cost < best_cost:
                best_cost = current_cost
                best_image = generated_image[0].numpy()

            if step is not None and (
                i == 0 or i % step == 0 or i == iterations
            ):
                print(
                    f"Cost at iteration {i}: {current_cost}, "
                    f"content {float(J_content.numpy())}, "
                    f"style {float(J_style.numpy())}, "
                    f"var {float(J_var.numpy())}"
                )

            if i < iterations:
                optimizer.apply_gradients([(grads, generated_image)])
                clipped = tf.clip_by_value(generated_image, 0.0, 1.0)
                generated_image.assign(clipped)

        return best_image, float(best_cost)
