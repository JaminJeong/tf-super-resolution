import argparse
import os
import tensorflow as tf

# import cv2
import numpy as np

from data_augmentation import load, normalize, resize, down_scale
from model import Generator, generate_images

if __name__ == "__main__":
  parser = argparse.ArgumentParser(prog="Generator test",
                                   description="Generator", add_help=True)
  parser.add_argument('-i', '--INPUTFILE', help='input image.', required=True)
  parser.add_argument('-c', '--CKPTFILE', help='input image.', required=True)
  args = parser.parse_args()

  assert os.path.isfile(args.INPUTFILE)
  assert (".jpg" in args.INPUTFILE) or (".png" in args.INPUTFILE)
  if not os.path.isdir("./generate_image"):
    os.mkdir("./generate_image")

  assert os.path.isfile(args.CKPTFILE + ".index")

  filename = os.path.splitext(os.path.basename(args.INPUTFILE))[0]

  checkpoint_dir = './training_checkpoints'
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
  generator = Generator()
  checkpoint = tf.train.Checkpoint(generator=generator)
  checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

  input_image, real_image = load(args.INPUTFILE)
  input_image, real_image = resize(input_image, real_image, 256, 256)
  input_image = down_scale(input_image)
  input_image, real_image = normalize(input_image, real_image)
  input_image = tf.expand_dims(input_image, axis=0)
  real_image = tf.expand_dims(real_image, axis=0)
  generate_images("./generate_image/{}_output.jpg".format(filename), generator, input_image, real_image)
