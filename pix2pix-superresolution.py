#!/usr/bin/env python
# coding: utf-8

# ##### Copyright 2019 The TensorFlow Authors.
# 
# Licensed under the Apache License, Version 2.0 (the "License");

#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import, division, print_function, unicode_literals

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

try:
  # %tensorflow_version only exists in Colab.
  get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
  pass
import tensorflow as tf

import os
import time

from matplotlib import pyplot as plt
from IPython import display


_URL = "http://download.tensorflow.org/example_images/flower_photos.tgz"

path_to_zip = tf.keras.utils.get_file('flower_photos.tgz',
                                      origin=_URL,
                                      extract=True)

PATH = os.path.join(os.path.dirname(path_to_zip), 'flower_photos/')
print(f"PATH : {PATH}")

get_ipython().system('ls -al /home/jamin/.keras/datasets/flower_photos/')
get_ipython().system('ls -al /home/jamin/.keras/datasets/flower_photos/daisy')

if not os.path.isdir("output"):
  os.mkdir("output")

BUFFER_SIZE = 400
BATCH_SIZE = 64

#inp, re = load(PATH+'daisy/14221848160_7f0a37c395.jpg')
## casting to int for matplotlib to show the image
#plt.figure()
## plt.imshow(tf.squeeze(inp)/255.0)
#plt.imshow(inp/255.0)
#plt.figure()
#plt.imshow(re/255.0)

from data_augmentation import load

# As you can see in the images below
# that they are going through random jittering
# Random jittering as described in the paper is to
#
# 1. Resize an image to bigger height and width
# 2. Randomly crop to the target size
# 3. Randomly flip the image horizontally
from model import load_image_train, load_image_test

## Input Pipeline

# train_dataset = tf.data.Dataset.list_files(PATH+'train/*.jpg')
train_dataset = tf.data.Dataset.list_files(PATH+'*/*.jpg')
train_dataset = train_dataset.map(load_image_train,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)


test_dataset = tf.data.Dataset.list_files(PATH+'*/*.jpg')
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)


OUTPUT_CHANNELS = 3

from model import downsample, upsample
from model import Generator, Discriminator, generate_images

inp, re = load(PATH+'daisy/9611923744_013b29e4da_n.jpg')

down_model = downsample(3, 4)
down_result = down_model(tf.expand_dims(inp, 0))
print (down_result.shape)

up_model = upsample(3, 4)
up_result = up_model(down_result)
print (up_result.shape)

generator = Generator()
# tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)


# gen_output = generator(inp[tf.newaxis,...], training=False)
# plt.imshow(gen_output[0,...])


# * **Generator loss**
#   * It is a sigmoid cross entropy loss of the generated images and an **array of ones**.
#   * The [paper](https://arxiv.org/abs/1611.07004) also includes L1 loss which is MAE (mean absolute error) between the generated image and the target image.
#   * This allows the generated image to become structurally similar to the target image.
#   * The formula to calculate the total generator loss = gan_loss + LAMBDA * l1_loss, where LAMBDA = 100. This value was decided by the authors of the [paper](https://arxiv.org/abs/1611.07004).

# The training procedure for the generator is shown below:

LAMBDA = 100


def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss


# 
# 
# ![Generator Update Image](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/images/gen.png?raw=1)
# 

# ## Build the Discriminator
#   * The Discriminator is a PatchGAN.
#   * Each block in the discriminator is (Conv -> BatchNorm -> Leaky ReLU)
#   * The shape of the output after the last layer is (batch_size, 30, 30, 1)
#   * Each 30x30 patch of the output classifies a 70x70 portion of the input image (such an architecture is called a PatchGAN).
#   * Discriminator receives 2 inputs.
#     * Input image and the target image, which it should classify as real.
#     * Input image and the generated image (output of generator), which it should classify as fake.
#     * We concatenate these 2 inputs together in the code (`tf.concat([inp, tar], axis=-1)`)


discriminator = Discriminator()
# tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)

# **Discriminator loss**
#   * The discriminator loss function takes 2 inputs; **real images, generated images**
#   * real_loss is a sigmoid cross entropy loss of the **real images** and an **array of ones(since these are the real images)**
#   * generated_loss is a sigmoid cross entropy loss of the **generated images** and an **array of zeros(since these are the fake images)**
#   * Then the total_loss is the sum of real_loss and the generated_loss
# 

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss


# The training procedure for the discriminator is shown below.
# 
# To learn more about the architecture and the hyperparameters you can refer the [paper](https://arxiv.org/abs/1611.07004).

# 
# ![Discriminator Update Image](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/images/dis.png?raw=1)
# 

# ## Define the Optimizers and Checkpoint-saver
# 
# 
# 

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


# ## Generate Images
#
# Write a function to plot some images during training.
#
# * We pass images from the test dataset to the generator.
# * The generator will then translate the input image into the output.
# * Last step is to plot the predictions and **voila!**

# Note: The `training=True` is intentional here since
# we want the batch statistics while running the model
# on the test dataset. If we use training=False, we will get
# the accumulated statistics learned from the training dataset
# (which we don't want)


# for index, (example_input, example_target) in enumerate(test_dataset.take(1)):
#   generate_images("./output{}.jpg".format(index), generator, example_input, example_target)
#

# ## Training
# 
# * For each example input generate an output.
# * The discriminator receives the input_image and the generated image as the first input. The second input is the input_image and the target_image.
# * Next, we calculate the generator and the discriminator loss.
# * Then, we calculate the gradients of loss with respect to both the generator and the discriminator variables(inputs) and apply those to the optimizer.
# * Then log the losses to TensorBoard.

EPOCHS = 150


import datetime
log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


@tf.function
def train_step(input_image, target, epoch):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
    tf.summary.scalar('disc_loss', disc_loss, step=epoch)


# The actual training loop:
# 
# * Iterates over the number of epochs.
# * On each epoch it clears the display, and runs `generate_images` to show it's progress.
# * On each epoch it iterates over the training dataset, printing a '.' for each example.
# * It saves a checkpoint every 20 epochs.

def fit(train_ds, epochs, test_ds):
  for epoch in range(epochs):
    start = time.time()

    display.clear_output(wait=True)

    for example_input, example_target in test_ds.take(1):
      generate_images("./output/epoch_{}.jpg".format(epoch), generator, example_input, example_target)
    print("Epoch: ", epoch)

    # Train
    for n, (input_image, target) in train_ds.enumerate():
      print('.', end='')
      if (n+1) % 100 == 0:
        print()
      train_step(input_image, target, epoch)
    print()

    # saving (checkpoint) the model every 20 epochs
    if (epoch + 1) % 20 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))
  checkpoint.save(file_prefix = checkpoint_prefix)


#docs_infra: no_execute
#get_ipython().run_line_magic('load_ext', 'tensorboard')
#get_ipython().run_line_magic('tensorboard', '--logdir {log_dir}')


# Now run the training loop:

fit(train_dataset, EPOCHS, test_dataset)

get_ipython().system('ls {}'.format(checkpoint_dir))

# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


## Generate using test dataset

# Run the trained model on a few examples from the test dataset
for idx, (inp, tar) in enumerate(test_dataset.take(5)):
  generate_images("./output/result_{}.jpg".format(idx), generator, inp, tar)

