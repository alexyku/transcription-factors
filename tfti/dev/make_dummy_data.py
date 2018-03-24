# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transcription factor transformer imputation (TFTI)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Dependency imports

from six.moves import xrange  # pylint: disable=redefined-builtin
from tensor2tensor.data_generators import generator_utils
from tfti import DeepseaProblem

import tensorflow as tf

tf.flags.DEFINE_string("data_dir", "~/t2t_data", "")
tf.flags.DEFINE_string("tmp_dir", "/tmp/t2t_datagen", "")
tf.flags.DEFINE_integer("num_train", 100, "")
tf.flags.DEFINE_integer("num_dev", 100, "")

FLAGS = tf.flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)


def main():
  data_dir = os.path.expanduser(FLAGS.data_dir)
  tmp_dir = os.path.expanduser(FLAGS.tmp_dir)
  p = DeepseaProblem()
  p.maybe_download_and_unzip(FLAGS.tmp_dir)
  def capped_generator(tmp_dir, is_training, max_to_gen):
    g = p.generator(tmp_dir, is_training)
    for _, example_dict in zip(xrange(max_to_gen), g):
      yield example_dict
  # Generate a subset of the data.
  generator_utils.generate_dataset_and_shuffle(
    capped_generator(tmp_dir, True, FLAGS.num_train),
    p.training_filepaths(data_dir, 1, shuffled=True),
    capped_generator(tmp_dir, False, FLAGS.num_dev),
    p.dev_filepaths(data_dir, 1, shuffled=True))


if __name__ == '__main__':
  main()
