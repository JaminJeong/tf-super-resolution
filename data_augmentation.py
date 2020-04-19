from __future__ import absolute_import, division, print_function, unicode_literals

try:
  # %tensorflow_version only exists in Colab.
  get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
  pass

import tensorflow as tf

def load(image_file):
  image = tf.io.read_file(image_file)
  image = tf.image.decode_jpeg(image)

  input_image = image
  real_image = image

  input_image = tf.cast(input_image, tf.float32)
  real_image = tf.cast(real_image, tf.float32)

  return input_image, real_image

def down_scale(input_image, IMG_HEIGHT=256, IMG_WIDTH=256):
  input_image = tf.image.resize(input_image, [int(IMG_HEIGHT/4), int(IMG_WIDTH/4)],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  input_image = tf.image.resize(input_image, [IMG_HEIGHT, IMG_HEIGHT],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  return input_image

def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image


def random_crop(input_image, real_image, IMG_HEIGHT=256, IMG_WIDTH=256):
  stacked_image = tf.stack([input_image, real_image], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image[0], cropped_image[1]


# normalizing the images to [-1, 1]

def normalize(input_image, real_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1

  return input_image, real_image


@tf.function()
def random_jitter(input_image, real_image, input_image_size = 286):
  # resizing to 286 x 286 x 3
  input_image, real_image = resize(input_image, real_image, input_image_size, input_image_size)

  # randomly cropping to 256 x 256 x 3
  input_image, real_image = random_crop(input_image, real_image, IMG_HEIGHT=256, IMG_WIDTH=256)
  input_image = down_scale(input_image)

  if tf.random.uniform(()) > 0.5:
    # random mirroring
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)

  return input_image, real_image

# As you can see in the images below
# that they are going through random jittering
# Random jittering as described in the paper is to
#
# 1. Resize an image to bigger height and width
# 2. Randomly crop to the target size
# 3. Randomly flip the image horizontally

if __name__ == "__main__":
  import argparse
  import cv2
  import os
  import numpy as np

  parser = argparse.ArgumentParser(prog="data augmentation test",
                                   description="data augmentation", add_help=True)
  parser.add_argument('-i', '--INPUTFILE', help='input image.', required=True)
  parser.add_argument('-o', '--OUTPUTFILE', help='result images.', required=True)
  args = parser.parse_args()

  assert os.path.isfile(args.INPUTFILE)
  assert (".jpg" in args.INPUTFILE) or (".png" in args.INPUTFILE)

  input_image, real_image = load(args.INPUTFILE)
  input_image, real_image = random_jitter(input_image, real_image)
  result = np.concatenate((input_image, real_image), axis=1)
  result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
  cv2.imwrite(args.OUTPUTFILE, result)