import tensorflow as tf  # for deep learning based ops


# create a sample model for the dcgan
class WasserstienGenerative(tf.keras.models.Model):
    """
    Class for implementaion of Generative model for Wasserstien GAN Arch
    """
    def __init__(self, *args, **kwargs):
        super(WasserstienGenerative, self).__init__(*args, **kwargs)

        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(7 * 7 * 256,
                                  use_bias=False,
                                  input_shape=(100, )),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            # reshape function
            tf.keras.layers.Reshape((7, 7, 256)),

            # first conv2d transpose
            tf.keras.layers.Conv2DTranspose(filters=128,
                                            kernel_size=(5, 5),
                                            strides=(1, 1),
                                            padding="same",
                                            use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            # second conv2d transpose
            tf.keras.layers.Conv2DTranspose(filters=64,
                                            kernel_size=(5, 5),
                                            strides=(2, 2),
                                            padding="same",
                                            use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            # third conv2d transpose
            tf.keras.layers.Conv2DTranspose(filters=1,
                                            kernel_size=(5, 5),
                                            strides=(2, 2),
                                            padding="same",
                                            use_bias=False,
                                            activation=tf.nn.tanh)
        ])

    def call(self, inputs):
        return self.model(inputs)


# create a sample model for the dcgan
class WasserstienDiscriminative(tf.keras.models.Model):
    """
    Discriminative Model for Wasserstien GAN Arch
    """
    def __init__(self, *args, **kwargs):
        super(WasserstienDiscriminative, self).__init__(*args, **kwargs)

        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64,
                                   kernel_size=(5, 5),
                                   strides=(2, 2),
                                   padding="same"),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dropout(0.3),

            # first conv2d transpose
            tf.keras.layers.Conv2D(filters=128,
                                   kernel_size=(5, 5),
                                   strides=(1, 1),
                                   padding="same"),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dropout(0.3),

            # third conv2d transpose
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1)
        ])

    def call(self, inputs):
        return self.model(inputs)
