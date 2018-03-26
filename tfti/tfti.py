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
import tarfile

# Dependency imports

import h5py
import numpy as np
import scipy

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.data_generators import dna_encoder
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import modalities
from tensor2tensor.models import transformer
from tensor2tensor.utils import registry

import tensorflow as tf


@registry.register_symbol_modality("position_sensitive")
class PositionSensitiveSymbolModality(modalities.SymbolModality):
  """Symbol modality for position sensitive sequences.

  Use this modality when an symbol id in one position means something completely
  different in a different position. That is an id X in position I has a 
  different meaning than the same id X at position J.

  To put this concretely, an id of 1 for transcription factor A should not map
  to the same embedding as an id of 1 for transcription factor B.

  To remedy this, we do two fold:
  - Each (position, id) pair gets its own embedding.
  - Each position gets its own logit projection var.

  This is quite expensive, and is only practical if the number of positions
  and/or vocab size are not too big. In the tfti case, the vocab size is
  generally 3 [PAD, NEG, POS] or 4 [PAD, NEG, POS, UNK], which is relatively
  small.
  """

  @property
  def name(self):
    return "position_sensitive_symbol_modality_%d_%d" % (self._vocab_size,
                                                         self._body_input_depth)

  def embedding(self, x, vocab_size, dense_size, **kwargs):
    """Each (position, id) pair gets its own embedding."""
    shape_list = common_layers.shape_list(x)
    # If we had an extra channel dimensions, assume they are 1.
    if len(shape_list) > 2:
      x = tf.reshape(x, shape_list[:2])
    seq_length = common_layers.shape_list(x)[1]
    x += tf.range(seq_length, dtype=x.dtype) * vocab_size
    return common_layers.embedding(
        x, vocab_size * seq_length, dense_size, **kwargs)

  def bottom_simple(self, x, name, reuse):
    with tf.variable_scope(name, reuse=reuse):
      # Squeeze out the channels dimension.
      x = tf.squeeze(x, axis=3)
      x = common_layers.dropout_no_scaling(
          x, 1.0 - self._model_hparams.symbol_dropout)
      ret = self.embedding(x, self._vocab_size, self._body_input_depth)
      ret = tf.expand_dims(ret, 2)
      if self._model_hparams.multiply_embedding_mode == "sqrt_depth":
        ret *= self._body_input_depth**0.5
      # No embedding for padding ids (set to zero).
      ret *= tf.expand_dims(tf.to_float(tf.not_equal(x, 0)), -1)
      return ret

  def top(self, body_output, _):
    """Generate logits.

    Args:
      body_output: A Tensor with shape [batch, p0, p1, body_input_depth]

    Returns:
      logits: A Tensor with shape  [batch, p0, p1, ?, vocab_size].
    """
    if self._model_hparams.symbol_modality_skip_top:
      return tf.expand_dims(body_output, 3)

    if self._model_hparams.shared_embedding_and_softmax_weights:
      scope_name = "shared"
      reuse = True
    else:
      scope_name = "softmax"
      reuse = False

    with tf.variable_scope(scope_name, reuse=reuse):
      body_output_shape = common_layers.shape_list(body_output)
      body_output = common_layers.flatten4d3d(body_output)
      # Transpose to swap the batch and length dimensions to use tf.map_fn.
      body_output = tf.transpose(body_output, [1, 0, 2])
      def to_logits(x):
        # New logit var for each binding prediction.
        var = self._get_weights(body_output_shape[-1])
        return tf.matmul(x, var, transpose_b=True)
      logits = tf.map_fn(to_logits, body_output)
      # Transpose again to revert the aforementioned swap.
      logits = tf.transpose(logits, [1, 0, 2])
      return tf.reshape(logits, body_output_shape[:-1] + [1, self._vocab_size])


@registry.register_problem("genomics_binding_deepsea")
class DeepseaProblem(problem.Problem):
  """N-gram/k-mer embedded deepsea data."""
    
  @property
  def default_chunk_size(self):
    # This is only used if "chunk_size" is
    # not specified in the model hparams.
    return 4  # n-gram/k-mer size.
  
  @property
  def num_output_predictions(self):
    return 919  # DNase, TFs and histones.
  
  @property
  def num_output_classes(self):
    return 2  # Binary classification.
  
  @property
  def input_sequence_length(self):
    return 1000 # Number of residues.
  
  @property
  def input_sequence_depth(self):
    return 4  # ACTG channels (in that order).

  @property
  def position_sensitive_targets(self):
    return True
  
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
      assert len(targets) == self.num_output_predictions
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
    if not model_hparams.get("chunk_size"):
      model_hparams.chunk_size = self.default_chunk_size
    vocab_size = dna_encoder.DNAEncoder(model_hparams.chunk_size).vocab_size
    p = defaults
    # Symbol modality reserves 0 as "padding". Which is why the
    # targets has an extra symbol. :atent targets have yet another
    # symbol for "unknown" -- two extra symbols.
    target_and_latent_modality = registry.Modalities.SYMBOL
    if self.position_sensitive_targets:
      target_and_latent_modality += ":position_sensitive"
    p.input_modality = {"inputs": (registry.Modalities.SYMBOL, vocab_size),
                        "latents": (target_and_latent_modality,
                                    self.num_output_classes + 2)}
    p.target_modality = (target_and_latent_modality,
                         self.num_output_classes + 1)
    p.input_space_id = problem.SpaceID.DNA
    p.target_space_id = problem.SpaceID.GENERIC
  
  def example_reading_spec(self):
    """Reader spec for tf-record examples."""
    data_fields = {
      "index": tf.FixedLenFeature([1], tf.int64),
      "inputs": tf.FixedLenFeature([self.input_sequence_length], tf.int64),
      "targets": tf.FixedLenFeature([self.num_output_predictions], tf.int64),
    }
    data_items_to_decoders = None
    return (data_fields, data_items_to_decoders)
  
  def preprocess_example(self, example, mode, hparams):
    """Preprocess the model inputs."""
    inputs = example["inputs"]
    targets = example["targets"]
    encoder = dna_encoder.DNAEncoder(hparams.chunk_size)
    def to_ids(inputs):
      ids = encoder.encode("".join(map(chr, inputs)))
      return np.array(ids, dtype=np.int64)
    [inputs] = tf.py_func(to_ids, [inputs], [tf.int64], stateful=False)
    # Reshape to the [p0, p1, channels] modality convention.
    out_size = int(np.ceil(self.input_sequence_length / hparams.chunk_size))
    inputs = tf.reshape(inputs, [out_size, 1, 1])
    targets = tf.reshape(targets, [self.num_output_predictions, 1, 1])
    # Targets are natively binary (i.e., 0 or 1), we add one to each
    # (i.e., 0->1 and 1->2) to preserve the meaning of 0 (the "padding" id).
    targets += 1

    # TODO(Alex): Create partially-observed latent targets and set the observed
    # targets to 0 (the "padding" id). We could introduce an hparam called
    # "latent_dropout". That is, the probability of dropping a ground-truth
    # target when computing the latent.

    # The latent is a tensor of 3s (the "unknown" id).
    latents = 3 * tf.ones(common_layers.shape_list(targets), dtype=tf.int64)
    example["inputs"] = inputs
    example["targets"] = targets
    example["latents"] = latents
    return example


@registry.register_model("tfti_transformer")
class TftiTransformer(transformer.Transformer):
  """A stack of transformer layers.
  
  This is a semi-generative model for imputing binding labels.
  That is, given that you know some transcription factors bind,
  can you predict whether other (unobserved) tf's bind.
  
  A latent code is fed into the decoder. Suppose we have N tf binding
  predictions to make, and we only know the ground truth for K of them.
  Then for known tfs, the latent id be 1 for "positive" and 2 for "negative".
  For "unknown" tfs the latent id will be 3 (0 is the "padding" id).
  
  For training, we artificially mask known tfs, get the model to impute,
  and compute the loss w.r.t. the imputed predictions.
  
  This is motivated by the idea that a model should consider all the information
  it has to make a prediction. That is, both the sequence, and the prediction
  being made for other tfs.
  """
  
  def body(self, features):
    """Transformer main model_fn.
    Args:
      features: Map of features to the model. Should contain the following:
          "inputs": Transformer inputs [batch_size, input_length, hidden_dim]
          "targets": Target decoder outputs.
              [batch_size, decoder_length, hidden_dim]
          "latents": Latent decoder inputs.
              [batch_size, decoder_length, hidden_dim]
          "target_space_id"
    Returns:
      Final decoder representation. [batch_size, decoder_length, hidden_dim]
    """
    hparams = self._hparams

    inputs = features["inputs"]
    target_space = features["target_space_id"]
    encoder_output, encoder_decoder_attention_bias = self.encode(
        inputs, target_space, hparams, features=features)

    # No longer call transformer.transformer_prepare_decoder because:
    # - We are using full decoder self-attention (i.e., no attention bias).
    # - We are not adding positional embeddings to the decoder inputs.
    
    latent = features["latents"]
    decoder_input = common_layers.flatten4d3d(latent)
    decoder_self_attention_bias = None

    decoder_output = self.decode(
        decoder_input,
        encoder_output,
        encoder_decoder_attention_bias,
        decoder_self_attention_bias,
        hparams,
        nonpadding=transformer.features_to_nonpadding(features, "targets"))

    expected_attentions = features.get("expected_attentions")
    if expected_attentions is not None:
      attention_loss = common_attention.encoder_decoder_attention_loss(
          expected_attentions, self.attention_weights,
          hparams.expected_attention_loss_type,
          hparams.expected_attention_loss_multiplier)
      return decoder_output, {"attention_loss": attention_loss}

    return decoder_output