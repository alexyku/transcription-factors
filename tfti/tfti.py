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

"""Transformer model from "Attention Is All You Need"."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile

# Dependency imports

import h5py
import scipy

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.data_generators import dna_encoder
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.layers import common_layers
from tensor2tensor.models import transformer
from tensor2tensor.utils import registry

import tensorflow as tf


@registry.register_problem("genomics_binding_deepsea")
class DeepseaProblem(problem.Problem):
  """K-mer embedded deepsea data."""
    
  @property
  def default_chunk_size(self):
    # This is only used if "chunk_size" is
    # not specified in the model hparams.
    return 4  # k-mer size.
  
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
      for i in range(inputs.shape[2]):
        yield (self.stringify(inputs[:, :, i]),
             targets[:, i])
    def valid_generator():
      filename = os.path.join(tmp_dir, "deepsea_train/valid.mat")
      tmp = scipy.io.loadmat(filename)
      inputs = tmp['validxdata']
      targets = tmp['validdata']
      for i in range(inputs.shape[0]):
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
    # The Symbol modality reserves a symbol for "padding". Which is why the
    # targets has one extra symbol. The latent targets have yet another
    # symbol for "unknown" (for a total of two extra symbols).
    p.input_modality = {"inputs": (registry.Modalities.SYMBOL, vocab_size)
                        "latent_targets": (registry.Modalities.SYMBOL,
                                           self.num_output_classes + 2)}
    p.target_modality = (registry.Modalities.SYMBOL,
                         self.num_output_classes + 1)
    p.input_space_id = problem.SpaceID.DNA
    p.target_space_id = problem.SpaceID.GENERIC
  
  def example_reading_spec(self):
    """Reader spec for tf-record examples."""
    data_fields = {
      "index": tf.FixedLenFeature([], tf.int64),
      "inputs": tf.FixedLenFeature([self.input_sequence_length], tf.int64),
      "targets": tf.FixedLenFeature([self.num_output_predictions], tf.int64),
    }
    data_items_to_decoders = None
    return (data_fields, data_items_to_decoders)
  
  def preprocess_example(self, example, mode, hparams):
    """Preprocess the model inputs."""
    encoder = dna_encoder.DNAEncoder(hparams.chunk_size)
    out_size = int(np.ceil(self.input_sequence_length / hparams.chunk_size))
    def to_ids(inputs):
      return np.array([
        encoder.encode("".join([chr(x) for x in row]))
        for row in inputs], dtype=np.int64)
    example["inputs"] = tf.py_func(
      to_ids, [example["inputs"]], [tf.int64], stateful=False)[0]
    targets = example["targets"]
    # Because the targets are natively binary (0 or 1), we add 1 to each to
    # preserve the meaning of 0 (the "padding" id).
    example["targets"] += 1
    # The latent target is a tensor of 3s (the "unknown" id).
    # TODO(Alex): Create partially-observed latent targets and set the observed
    # targets to 0 (the "padding" id).
    # We could add an hparam called "latent_target_dropout" that indicates the
    # probability of dropping a ground-truth target when computing the latent
    # target. We could also explore tempering this parameter. Gradually
    # increasing the it over time.
    example["latent_targets"] = 3 * tf.ones(
        common_layers.shape_list(example["targets"]))
    return example


@registry.register_model("tfti_transformer")
class TftiTransformer(transformer.Transformer):
  """A stack of transformer layers.
  
  This is a semi-generative model for imputing binding labels.
  That is, given that you know some transcription factors bind,
  can you predict whether other (unobserved) tf's bind.
  
  A latent code is fed into the decoder. Suppose we have N tf binding
  predictions to make, and we only know the ground truth for K of them.
  Then for known tfs, the latent will be [1, 0, 0] for "binding" abd
  [0, 1, 0] for "no binding". For unknown tfs the latent will be [0, 0, 1].
  Thus the dimensionality of the latent code is [batch_size, num_tfs, 3].
  
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
	        "tragets": Target decoder outputs.
	            [batch_size, decoder_length, hidden_dim]
	        "target_space_id"
	  Returns:
	    Final decoder representation. [batch_size, decoder_length, hidden_dim]
	  """
	  hparams = self._hparams

	  if self.has_input:
	      inputs = features["inputs"]
	      target_space = features["target_space_id"]
	      encoder_output, encoder_decoder_attention_bias = self.encode(
	          inputs, target_space, hparams, features=features)
	  else:
	      encoder_output, encoder_decoder_attention_bias = (None, None)

	  # Latent tensor to be transformed into logits.
    # TODO(Alex): Remove decoder positional embeddings.
    latent_targets = features["latent_targets"]
    latent_targets = common_layers.flatten4d3d(latent_targets)
	  
	  decoder_input, _ = transformer_prepare_decoder(
	      latent_targets, hparams, features=features)
	  # No masking bias, full decoder self-attention.
	  decoder_self_attention_bias = None

	  decoder_output = self.decode(
	      decoder_input,
	      encoder_output,
	      encoder_decoder_attention_bias,
	      decoder_self_attention_bias,
	      hparams,
	      nonpadding=features_to_nonpadding(features, "targets"))

	  expected_attentions = features.get("expected_attentions")
	  if expected_attentions is not None:
	    attention_loss = common_attention.encoder_decoder_attention_loss(
	        expected_attentions, self.attention_weights,
	        hparams.expected_attention_loss_type,
	        hparams.expected_attention_loss_multiplier)
	    return decoder_output, {"attention_loss": attention_loss}

	  return decoder_output