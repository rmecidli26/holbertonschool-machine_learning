import tensorflow as tf

def create_batch_norm_layer(prev, n, activation):
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    x = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=initializer,
        use_bias=False
    )(prev)

    x = tf.keras.layers.BatchNormalization(
        axis=-1,
        momentum=0.99,
        epsilon=1e-7,
        center=True,
        scale=True
    )(x)

    if activation is not None:
        x = activation(x)

    return x
