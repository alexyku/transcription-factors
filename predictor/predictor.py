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

"""Transcription factor transformer imputation (TFTI).

Run from one directory up:

PROBLEM=genomics_binding_deepsea
MODEL=transformer_cnn
HPARAMS_SET=transformer_base_single_gpu
HPARAMS='batch_size=2,learning_rate=0.0001'

USR_DIR=./predictor
DATA_DIR=./tfti/dev
TRAIN_DIR=$HOME/t2t_train/$PROBLEM/$MODEL-$HPARAMS

mkdir -p $DATA_DIR $TRAIN_DIR

# Train
t2t-trainer \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DATA_DIR \
  --problems=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS_SET \
  --output_dir=$TRAIN_DIR \
  --hparams=$HPARAMS
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile

# Dependency imports

import h5py
import numpy as np
import scipy

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.data_generators import dna_encoder
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.layers import common_layers
from tensor2tensor.models import transformer
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry

import tensorflow as tf


@registry.register_problem("genomics_binding_deepsea")
class DeepseaProblem(problem.Problem):
  """N-gram/k-mer embedded deepsea data."""

  @property
  def chunk_size(self):
    return 4  # k-mer/n-gram size

  @property
  def num_binary_predictions(self):
    return 919  # DNase, TFs and histones.
  
  @property
  def input_sequence_length(self):
    return 1000 # Number of residues.

  def dataset_filename(self):
    return "genomics_binding_deepsea"

  def eval_metrics(self):
    # Note: this requires a modified metrics.py file in T2T.
    return [metrics.Metrics.AUCB]
  
  def stringify(self, one_hot_seq):
    """One-hot sequence to an ACTG string."""
    ids = one_hot_seq.dot(np.arange(1, 5))
    # An all-zero column is denoted by "N".
    bases = np.array(list("NACTG"))
    return "".join(bases[ids])
    
  def generator(self, tmp_dir, is_training):
    """Generates example dicts."""
    def train_generator():
      filename = os.path.join(tmp_dir, "deepsea_train/train.mat")
      tmp = h5py.File(filename)
      inputs = tmp["trainxdata"]
      targets = tmp["traindata"]
      for i in xrange(inputs.shape[2]):
        yield (self.stringify(inputs[:, :, i]),
             targets[:, i])
    def valid_generator():
      filename = os.path.join(tmp_dir, "deepsea_train/valid.mat")
      tmp = scipy.io.loadmat(filename)
      inputs = tmp['validxdata']
      targets = tmp['validdata']
      for i in xrange(inputs.shape[0]):
        yield (self.stringify(inputs[i].transpose([1, 0])),
             targets[i])
    generator = train_generator if is_training else valid_generator
    for i, (inputs, targets) in enumerate(generator()):
      if i % 1000 == 0:
        tf.logging.info(f"Generated {i} examples.")
      assert len(inputs) == self.input_sequence_length
      assert len(targets) == self.num_binary_predictions
      yield {"index": [i], "inputs": list(map(ord, inputs)),
             "targets": list(map(int, targets))}
  
  def maybe_download_and_unzip(self, tmp_dir):
    """Downloads deepsea data if it doesn't already exist."""
    url = ("http://deepsea.princeton.edu/media/code/"
           "deepsea_train_bundle.v0.9.tar.gz")
    filename = "deepsea_train_bundle.v0.9.tar.gz"
    generator_utils.maybe_download(tmp_dir, filename, url)
    dirpath = os.path.join(tmp_dir, "deepsea_train")
    if not os.path.exists(dirpath):
      tf.logging.info(f"Extracting archive {filename} to directory: {dirpath}")
      filepath = os.path.join(tmp_dir, filename)
      tarfile.open(filepath, "r:gz").extractall(tmp_dir)
    else:
      tf.logging.info(
          f"Not extracting archive, directory already found: {dirpath}")
    return dirpath
  
  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    """Generates tf-refords for the problem."""
    self.maybe_download_and_unzip(tmp_dir)
    generator_utils.generate_dataset_and_shuffle(
      self.generator(tmp_dir, is_training=True),
      self.training_filepaths(data_dir, 100, shuffled=True),
      self.generator(tmp_dir, is_training=False),
      self.dev_filepaths(data_dir, 1, shuffled=True))
  
  def hparams(self, defaults, model_hparams):
    """Augment the hparams for this problem."""
    p = defaults
    vocab_size = dna_encoder.DNAEncoder(self.chunk_size).vocab_size
    p.input_modality = {"inputs": (registry.Modalities.SYMBOL, vocab_size)}
    p.target_modality = (registry.Modalities.GENERIC, 2)  # binary labels
  
  def example_reading_spec(self):
    """Reader spec for tf-record examples."""
    data_fields = {
      "index": tf.FixedLenFeature([1], tf.int64),
      "inputs": tf.FixedLenFeature([self.input_sequence_length], tf.int64),
      "targets": tf.FixedLenFeature([self.num_binary_predictions], tf.int64),
    }
    data_items_to_decoders = None
    return (data_fields, data_items_to_decoders)
  
  def preprocess_example(self, example, mode, hparams):
    """Preprocess the model inputs."""
    inputs = example["inputs"]
    targets = example["targets"]
    encoder = dna_encoder.DNAEncoder(self.chunk_size)
    def to_ids(inputs):
      ids = encoder.encode("".join(map(chr, inputs)))
      return np.array(ids, dtype=np.int64)
    [inputs] = tf.py_func(to_ids, [inputs], [tf.int64], stateful=False)
    # Reshape to the [p0, p1, channels] modality convention.
    out_size = int(np.ceil(self.input_sequence_length / self.chunk_size))
    example["inputs"] = tf.reshape(inputs, [out_size, 1, 1])
    example["targets"] = tf.reshape(targets,[self.num_binary_predictions, 1, 1])
    return example


@registry.register_model("transformer_cnn")
class TransformerCNN(transformer.Transformer):
  """Sequence-CNN with interwoven self-attention."""
  # Gated Transcription Factor cOnvolutions (GTFO)

  def bottom(self, features):
    # input ids ==> input embeddings
    input_modality = self._problem_hparams.input_modality["inputs"]
    features["inputs"] = input_modality.bottom(features["inputs"])
    return features
    
  def top(self, body_output, unused_features):
    # body_output ==> logits
    logits = tf.reduce_mean(body_output, axis=-1)  # [batch_size, target_length]
    return logits
        
  def loss(self, logits, features):
    # logits ==> loss
    hparams = self._hparams
    labels = tf.squeeze(features["targets"], axis=[2, 3])
    #loss_num = tf.losses.sigmoid_cross_entropy(
    #    labels, logits, label_smoothing=hparams.label_smoothing)
    if not self._hparams.get("pos_weight"):
        self._hparams.pos_weight = 50
    loss_num = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
        tf.cast(labels, dtype=tf.float32), logits, self._hparams.pos_weight))
    return (loss_num, tf.constant(1.0))  # loss_num, loss_denom
  
  def body(self, features):
    hparams = self._hparams

    # by default conv ffn uses a kernel of size 3
    # by default local att uses a kernel size of 128
    global_attention_local_ffn = ("dot_product", "conv_relu_conv")
    local_attention_global_ffn = ("local_unmasked", "dense_relu_dense")

    # generally only use `local_attention_global_ffn` when using n-grams
    # otherwise the numer of parameters in the dense layers will be very large

    # hardcode this settings for now
    # TODO: eventually move this to hparams
    (hparams.self_attention_type,
      hparams.ffn_layer) = global_attention_local_ffn
    
    # actual model body
    inputs = features["inputs"]
    target_space = features["target_space_id"]
    body_output, _ = self.encode(
        inputs, target_space, hparams, features=features)
    
    targets = features["targets"]
    target_length = targets.shape[1]
    
    # dense connection along input_length
    # [batch_size, hidden_dim, input_length]
    body_output = tf.transpose(body_output, [0, 2, 1])

    # TODO: improvement suggestions:
    # - more sparse fully connected layers
    # - l0 regularization to encourage sparsity
    body_output = common_layers.dense(
        body_output, target_length, activation=tf.nn.relu)

    # [batch_size, target_length, hidden_dim]
    body_output = tf.transpose(body_output, [0, 2, 1])
    return body_output
