# -*- coding: utf-8 -*-
"""
author: Sanidhya Mangal
"""

import os  # for os related ops
import time  # for time related ops

import matplotlib.pyplot as plt  # for plotting
import PIL  # for image related ops
import tensorflow as tf  # for deep learning related steps
from IPython import display

from data_handler import data_loader_csv_unsupervisied
from models import WasserstienDiscriminative, WasserstienGenerative

# Batch and shuffle the data
#TODO: if you dont have fashion-minst data then you can download it from here:
# !wget https://gitlab.com/sanidhyamangal/datasets/-/raw/master/fashion-mnist_train.csv

# load traning dataset for the GAN
train_dataset = data_loader_csv_unsupervisied("./fashion-mnist_train.csv", 256)


# intansiate generator
generator = WasserstienGenerative()

print(generator.model.summary())
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap="gray")

# work on discriminator model
discriminator = WasserstienDiscriminative()
decision = discriminator(generated_image)
print(decision)


# create a loss for generative
def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)


# create a loss function for discriminator
def discriminator_loss(real_output, fake_output):
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)


# optimizers for the generative models
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# creation of checkpoint dirs
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator)

# all the epochs for the training
EPOCHS = 100
noise_dim = 100
num_examples_to_generate = 16
BATCH_SIZE = 256

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])


@tf.function
def train_step(images):
    """
    TF Graph function to be compiled into a graph
    """
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss,
                                               generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
    """
    Function to perform training ops on given set of epochs
    """
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        # Produce images for the GIF as we go
        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1,
                                                   time.time() - start))

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)


def generate_and_save_images(model, epoch, test_input):
    """
    A helper function for generating and saving images during training ops
    """
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap="gray")
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


# call to train function
train(train_dataset, EPOCHS)

# restoring the checkpoints for the
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


# Display a single image using the epoch number
def display_image(epoch_no):
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))


# display the image
display_image(EPOCHS)
