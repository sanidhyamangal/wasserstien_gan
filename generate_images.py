import os  # for os related ops

import matplotlib.pyplot as plt  # for plotting the results
import tensorflow as tf  # for deep learning related ops

from models import WasseterianGenerative


class WasseterianImageGenerator():
    """
    A Function for performing Image generation operations
    """
    def __init__(self, model_checkpoint_dir: str = "./training_checkpoints"):
        # init wasseterian generative examples
        self.generator = WasseterianGenerative()

        # creation of checkpoint dirs
        checkpoint_dir = './training_checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(generator=self.generator)

        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    def generate_save_images(self, image_name: str = "generated.png"):
        seed = tf.random.normal([16, 100])

        predictions = self.generator(seed)

        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap="gray")
            plt.axis('off')

        plt.savefig(image_name)
        plt.show()
