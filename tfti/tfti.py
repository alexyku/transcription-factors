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
from tensor2tensor.utils import metrics
from tensor2tensor.utils import modality
from tensor2tensor.utils import registry

import tensorflow as tf


################################################################################
################################## UTILITIES ###################################
################################################################################


def keep_first_dims(x, n):
  """Reshapes to the first n dims, assumes the rest are 1.

  Args:
    x: A Tensor with shape [d_1, ..., d_n, 1, ..., 1].
    n: Number of non-trivial trailing dimensions.

  Returns:
    A Tensor with shape [d_1, ..., d_n].
  """
  return tf.reshape(x, common_layers.shape_list(x)[:n])


################################################################################
################################### METRICS ####################################
################################################################################


def set_auc(logits, features, labels, curve, weights_fn=None):
  """Computes the approximate AUC via a Riemann sum.
  
  Args:
    logits : A Tensor of scores of shape [batch, nlabels, 1, 1].
    features: A feature dict mapping keys to Tensors.
    labels: A Tensor of int32s giving true set elements,
      of shape [batch, nlabels, 1, 1].
    curve: Specifies the name of the curve to be computed, "ROC" [default] or
      "PR" for the Precision-Recall-curve.
    weights_fn: A function to weight the elements (unused).
  Returns:
    aucs: A Tensor of shape [nlabels].
    weights: A Tensor of shape [nlabels].
  """
  logits = keep_first_dims(logits, 2)
  labels = keep_first_dims(labels, 2)
  predictions = tf.nn.sigmoid(logits)

  # `metrics_weights` is an alternative to `weights_fn`.
  if features and "metrics_weights" in features:
    metrics_weights = keep_first_dims(features["metrics_weights"], 2)
  else:
    metrics_weights = tf.ones(common_layers.shape_list(labels))  # Uniform weights.

  nlabels = int(labels.shape[1])
  aucs = []
  weights = []

  for i in range(nlabels):
    _, auc_op = tf.metrics.auc(
        curve=curve,
        labels=labels[:, i],
        predictions=predictions[:, i],
        weights=metrics_weights[:, i],
        updates_collections=tf.GraphKeys.METRIC_VARIABLES,
    )
    aucs.append(auc_op)
    weights.append(metrics_weights[:, i])

  return tf.stack(aucs), tf.stack(weights)


def average_auc(logits, features, labels, curve,  weights_fn=None):
  """Weighted average of AUC measurements."""
  aucs, weights = set_auc(logits, features, labels, curve, weights_fn)
  weighted_average_auc = tf.reduce_mean(tf.multiply(aucs, weights))
  return weighted_average_auc, tf.constant(1.0)


def set_auroc(logits, labels, features=None, weights_fn=None):
  """Area under receiver operator curve."""
  return set_auc(logits, features, labels, "ROC", weights_fn)


def set_auprc(logits, labels, features=None, weights_fn=None):
  """Area under precision recall curve."""
  return set_auc(logits, features, labels, "PR", weights_fn)


def average_auroc(logits, labels, features=None, weights_fn=None):
  """Average area under receiver operator curve."""
  return average_auc(logits, features, labels, "ROC", weights_fn)


def average_auprc(logits, labels, features=None, weights_fn=None):
  """Average area under precision recall curve."""
  return average_auc(logits, features, labels, "PR", weights_fn)


# Modify metrics.
metrics.Metrics.SET_AUROC = "set_auroc"
metrics.Metrics.SET_AUPRC = "set_auprc"
metrics.Metrics.AVERAGE_AUROC = "average_auroc"
metrics.Metrics.AVERAGE_AUPRC = "average_auprc"
metrics.METRICS_FNS[metrics.Metrics.SET_AUROC] = set_auroc
metrics.METRICS_FNS[metrics.Metrics.SET_AUPRC] = set_auprc
metrics.METRICS_FNS[metrics.Metrics.AVERAGE_AUROC] = average_auroc
metrics.METRICS_FNS[metrics.Metrics.AVERAGE_AUPRC] = average_auprc


################################################################################
################################ MODALITIES ####################################
################################################################################


def set_embedding(x, vocab_size, dense_size, **kwargs):
  """Each (ID, position) tuple gets a unique embedding.

  Args:
    x: An int Tensor with shape [batch_size, length] whose elements are in
      [0, vocab_size).
    vocab_size: Int. The range of valid ID values elements in x can take.
    dense_size: Int. The dimensionality of an embedding vector.

  Returns:
    A float Tensor with shape [batch_size, length, dense_size].
  """
  x = keep_first_dims(x, 2)
  seq_length = common_layers.shape_list(x)[1]
  x += tf.range(seq_length, dtype=x.dtype) * vocab_size
  new_vocab_size = vocab_size * seq_length
  return common_layers.embedding(x, new_vocab_size, dense_size, **kwargs)


@registry.register_class_label_modality("binary")
class BinaryClassLabelModality(modality.Modality):
  """Class label modality for predicting multiple binary labels.

  Assumes logits and targets are Tensors with shape [batch_size, nlabels].
  Assumes that 0 and 1 are the negative and positive IDs, respectively.
  """
  NEG, NEG_ID = ("<neg>", 0)
  POS, POS_ID = ("<pos>", 1)

  def __init__(self, model_hparams, vocab_size=None):
    self._model_hparams = model_hparams
    self._vocab_size = 2  # Binary predictions in (0, 1).

  @property
  def name(self):
    return "binary_class_label_modality_%d" % self._body_input_depth

  def bottom(self, x):
    """Class label space to embeddings.

    Args:
      x: An int Tensor with shape [batch_size, nlabels, 1, 1] whose elements
        are in [0, self._vocab_size).

    Returns:
      A float Tensor with shape [batch_size, nlabels, 1, hidden_size].
    """
    with tf.variable_scope(self.name):
      res = set_embedding(x, self._vocab_size, self._body_input_depth)
      return tf.expand_dims(res, 2)  # [batch_size, nlabels, 1, hidden_size]

  def top(self, body_output, _):
    """Body output to logits.

    Args:
      body_output: A float Tensor with shape
        [batch_size, nlabels, 1, hidden_size].

    Returns:
      logits: A float Tensor with shape [batch_size, nlabels, 1, 1].

    """
    scope = tf.variable_scope(self.name)
    if self._model_hparams.get("gpu_compat"):
      scope = scope and tf.device("/gpu:0")

    with scope:
      x = body_output  # [batch_size, nlabels, 1, hidden_size]
      x = common_layers.flatten4d3d(x)
      if not self._model_hparams.get("multigpu"):
        # Transpose to apply `tf.map_fn` along the  `nlabels` dimension.
        # i.e., mapping along dimension 0 of [nlabels, batch_size, hidden_size].
        x = tf.transpose(x, [1, 0, 2])
        # Collapse `hidden_size` dimension of `x` with a dense layer
        # to get a Tensor with shape [nlabels, batch_size, 1]
        res = tf.map_fn(lambda y: tf.layers.dense(y, 1), x)
        # Reverse the transposition to get [batch_size, nlabels, 1].
        res =  tf.transpose(res, [1, 0, 2])

      else:
        mask = tf.get_variable("mask",
                                shape=common_layers.shape_list(x)[1:],
                                dtype=tf.float32)
        bias = tf.get_variable("bias",
                                shape=common_layers.shape_list(x)[1],
                                dtype=tf.float32)
        # Take inner product of each label position with a different vector,
        # resulting in a Tensor with shape [batch, nlabels].
        res = tf.reduce_sum(x * mask, axis=-1) + bias
        res = tf.expand_dims(res, -1) # [batch_size, nlabels, 1]
      return tf.expand_dims(res, 3)  # [batch_size, nlabels, 1, 1]

  @property
  def targets_weights_fn(self):
    """Returns a function to weight the eval metrics."""
    return common_layers.weights_all  # Uniform weights.

  @property
  def loss_weights_fn(self):
    """Returns a function to weight the loss."""
    hp = self._model_hparams
    if hp and not hp.get("pos_weight"):
      return common_layers.weights_all
    def pos_weights_fn(targets):
      # Weigh positive examples to correct for class imbalance.
      is_neg = tf.to_float(tf.equal(targets, 0))
      is_pos = tf.to_float(tf.equal(targets, 1))
      return hp.pos_weight * is_pos + is_neg
    return pos_weights_fn

  def loss(self, logits, targets):
    """Logits to loss.

    Args:
      logits: A float Tensor with shape [batch_size, nlabels, 1, 1].
      targets: An int Tensor with shape [batch_size, nlabels, 1, 1] that takes
        values in (0, 1).

    Returns:
      A tuple containing:
        loss_numerator: A Tensor with shape [1].
        loss_denominator: A Tensor with shape [1].
    """
    logits = keep_first_dims(logits, 2)
    targets = keep_first_dims(targets, 2)
    loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=targets,
        logits=logits,
        reduction="none")
    weights = self.loss_weights_fn(targets)
    return tf.reduce_sum(loss * weights), tf.reduce_sum(weights)


@registry.register_class_label_modality("binary_imputation")
class BinaryImputationClassLabelModality(BinaryClassLabelModality):
  """Class label modality for imputing multiple binary labels.

  Assumes that 2 is the unknown ID that will be imputed by the model.
  """
  UNK, UNK_ID = ("<unk>", 2)

  def __init__(self, model_hparams, vocab_size=None):
    self._model_hparams = model_hparams
    self._vocab_size = 3  # Binary predictions with unkowns in (0, 1, ?).

  @property
  def name(self):
    return "binary_imputation_class_label_modality_%d" % self._body_input_depth

  def bottom(self, x):
    """Class label space to embeddings.

    Args:
      x: An int Tensor with shape [batch_size, nlabels, 1, 1] whose elements
        are in [0, self._vocab_size).

    Returns:
      A float Tensor with shape [batch_size, nlabels, 1, hidden_size].
    """
    with tf.variable_scope(self.name):
      res = set_embedding(x, self._vocab_size, self._body_input_depth)
      return tf.expand_dims(res, 2)  # [batch_size, nlabels, 1, hidden_size]

  def loss(self, logits, targets):
    """Logits to loss.

    Args:
      logits: A float Tensor with shape [batch_size, nlabels, 1, 1].
      targets: An int Tensor with shape [batch_size, nlabels, 1, 1] that takes
        values in (0, 1).

    Returns:
      A tuple containing:
        loss_numerator: A Tensor with shape [1].
        loss_denominator: A Tensor with shape [1].
    """
    logits = keep_first_dims(logits, 2)
    targets = keep_first_dims(targets, 2)

    if self._model_hparams.scaled_loss:
      # Get all values that need to be ignored in loss. 
      loss_mask = 1 - tf.cast(tf.equal(targets, self.UNK_ID), tf.float32)
      # Scale the loss by the number of targets that were masked.
      scale_factor_n = tf.to_float(tf.size(loss_mask))
      scale_factor_d = tf.to_float(tf.reduce_sum(loss_mask))
    else:
      scale_factor_n = 1
      scale_factor_d = 1

    loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=targets,
        logits=logits,
        reduction="none")
    # For all self.UNK_ID values in targets, weight will be 0.
    weights = self.loss_weights_fn(targets)
    return tf.reduce_sum(loss * weights) * scale_factor_n, \
      tf.reduce_sum(weights) * scale_factor_d


################################################################################
################################## ENCODERS ####################################
################################################################################


class BinaryClassLabelEncoder(text_encoder.TextEncoder):
    def __init__(self, neg_chr="0", pos_chr="1"):
      self._num_reserved_ids = 0
      self._chr_to_id = {neg_chr: 0, pos_chr: 1}
      self._id_to_chr = {0: neg_chr, 1: pos_chr}
    
    def encode(self, label_str):
      return [self._chr_to_id[x] for x in label_str]
    
    def decode(self, ids):
      return "".join([self._id_to_chr[x] for x in ids])


################################################################################
################################## PROBLEMS ####################################
################################################################################


# Not registered. See subclass `TftiDeepseaProblem`.
class DeepseaProblem(problem.Problem):
  """N-gram/k-mer embedded deepsea data."""
    
  @property
  def chunk_size(self):
    """n-gram/k-mer size."""
    return 4
  
  @property
  def num_binary_predictions(self):
    """Binary predictions: DNase, TFs and histones."""
    return 919
  
  @property
  def input_sequence_length(self):
    return 1000 # Number of residues.

  def preprocess(self, dataset, mode, hparams):
    """Repeats the dataset 10 times if in evaluation mode. 

    Args:
      dataset: the Dataset of already decoded but not yet preprocessed features.
      mode: tf.estimator.ModeKeys
      hparams: HParams, model hyperparameters

    Returns:
      a Dataset
    """
    dataset = super().preprocess(dataset, mode, hparams)
    if mode == tf.estimator.ModeKeys.EVAL:
      # Use --eval_steps to control how long we evaluate.
      dataset = dataset.repeat(count=99999)
    if hparams.get("filter_negatives"):
      dataset = dataset.filter(lambda ex: tf.reduce_any(tf.equal(ex["targets"], tf.constant(1, dtype=tf.int64))))
    return dataset

  def eval_metrics(self):
    """Metrics to run in the eval loop."""
    return [metrics.Metrics.AVERAGE_AUROC,
            metrics.Metrics.AVERAGE_AUPRC]

  def dataset_filename(self):
    """Data set filename (shared among subclasses)."""
    return "genomics_binding_deepsea"

  def feature_encoders(self, data_dir):
    del data_dir
    return {
        "inputs": text_encoder.ByteTextEncoder(num_reserved_ids=0),
        "targets": BinaryClassLabelEncoder(),
    }
  
  def stringify(self, one_hot_seq):
    """One-hot sequence to an ACTG string.

    one_hot_seq: An array with shape [length, 4] representing a one-hot-encoded
      DNA sequence. A column with all zeros is denoted by "N".

    Returns:
      A string decoding one_hot_seq.
    """
    ids = one_hot_seq.dot(np.arange(1, 5))
    # An all-zero column is denoted by "N".
    bases = np.array(list("NACTG"))
    return "".join(bases[ids])
    
  def generator(self, tmp_dir, is_training):
    """Generates example dicts.

    Args:
      tmp_dir: String. Path to working directory where your unprocessed data is.
      is_training: Boolean. Whether to generate data from the training or
        validation set.

    Yields:
      A feature dict containing:
        "index": A singleton list with the iteration index.
        "inputs": A list of ints representing ascii encoded DNA bases in NACTG.
        "targets": A list of ints representing label classes in (0, 1).
    """
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
      inputs = tmp["validxdata"]
      targets = tmp["validdata"]
      for i in xrange(inputs.shape[0]):
        yield (self.stringify(inputs[i].transpose([1, 0])),
             targets[i])
    generator = train_generator if is_training else valid_generator
    for i, (inputs, targets) in enumerate(generator()):
      if (i % 1000 == 0):
        tf.logging.info(f"Generated {i} examples.")
      assert len(inputs) == self.input_sequence_length
      assert len(targets) == self.num_binary_predictions
      yield {"index": [i], "inputs": list(map(ord, inputs)),
             "targets": list(map(int, targets))}
  
  def maybe_download_and_unzip(self, tmp_dir):
    """Downloads deepsea data if it doesn"t already exist.

    Args:
      tmp_dir: String. The directory to maybe download to.

    Returns:
      String. The directory path where the unprocessed data was downloaded and
        unzipped.
    """
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
    """Generates tf-refords for the problem.

    Args:
      data_dir: String. The directory to generate TF-Records to.
      tmp_dir: String. The directory to download the unprocessed data to.

    Returns:
      None.
    """
    self.maybe_download_and_unzip(tmp_dir)
    generator_utils.generate_dataset_and_shuffle(
      self.generator(tmp_dir, is_training=True),
      self.training_filepaths(data_dir, 100, shuffled=True),
      self.generator(tmp_dir, is_training=False),
      self.dev_filepaths(data_dir, 1, shuffled=True))
  
  def hparams(self, defaults, model_hparams):
    """Augment the hparams for this problem.

    Args:
      defaults: The default hparams to augment for this problem.
      model_hparams: The hparams of the model being used. Augment these as 
        needed for this particular problem.

    Returns:
      None.
    """
    p = defaults
    vocab_size = dna_encoder.DNAEncoder(self.chunk_size).vocab_size
    p.input_modality = {"inputs": (registry.Modalities.SYMBOL, vocab_size)}
    p.target_modality = ("%s:binary" % registry.Modalities.CLASS_LABEL, None)
    p.input_space_id = problem.SpaceID.DNA
    p.target_space_id = problem.SpaceID.GENERIC
  
  def example_reading_spec(self):
    """Reader spec for tf-record examples.
    
    Specify the names and types of the features on disk. Specify
      tf.contrib.slim.tfexample_decoder.
    """
    data_fields = {
      "index": tf.FixedLenFeature([], tf.int64),
      "inputs": tf.FixedLenFeature([self.input_sequence_length], tf.int64),
      "targets": tf.FixedLenFeature([self.num_binary_predictions], tf.int64),
    }
    data_items_to_decoders = None
    return (data_fields, data_items_to_decoders)
  
  def preprocess_example(self, example, mode, hparams):
    """Preprocess the model inputs.
    
    Args:
      example: Feature dict from feature name to Tensor or SparseTensor.
      mode: String. Specifies training, eval, and inference.
      hparams: The problem hparams.

    Returns:
      Feature dict from feature name to Tensor or SparseTensor.
    """
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
    example["targets"] = tf.reshape(
        targets, [self.num_binary_predictions, 1, 1])
    return example


@registry.register_problem("genomics_binding_deepsea")
class TftiDeepseaProblem(DeepseaProblem):
  """DeepSEA problem for imputation models, such as TFTI."""

  @property
  def unk_id(self):
    return BinaryImputationClassLabelModality.UNK_ID

  def hparams(self, defaults, model_hparams):
    """See base class."""
    super().hparams(defaults, model_hparams)
    defaults.input_modality["latents"] = (
        "%s:binary_imputation" % registry.Modalities.CLASS_LABEL, None)
    if model_hparams.scaled_loss:
      defaults.target_modality = (
          "%s:binary_imputation" % registry.Modalities.CLASS_LABEL, None)

  def make_latents(self, features, hparams):
    """Generates a partially observed latent target tensor to be imputed.

    Args:
      features: Feature dict mapping keys to Tensors.
      hparams: The problem hparams.

    Returns:
      An int Tensor with values in (0, 1, 2), where 2 represents the unknown
        ID to be imputed by the model. The unknown ID is specified by:
          BinaryImputationClassLabelModality.UNK_ID
    """
    targets = features["targets"]
    targets_shape = common_layers.shape_list(targets)
    if "latent_keep_mask" in features:
      tf.logging.info("Generating latents with a `latent_keep_mask`.")
      # Keep mask must be the same shape as targets. 
      keep_mask = tf.reshape(features["latent_keep_mask"], targets_shape)
    else:
      # The latent_keep_prob is the probability of keeping a ground-truth label.
      keep_prob = hparams.get("latent_keep_prob", 0.0)
      tf.logging.info(f"Generating latents with a `keep_prob` of {keep_prob}")
      keep_mask = tf.random_uniform(targets_shape) < keep_prob
    # Use the keep_mask to create latents.
    keep_mask = tf.to_float(keep_mask)
    return tf.to_int32(keep_mask * tf.to_float(targets)
                       + (1.0 - keep_mask) * self.unk_id), keep_mask

  def preprocess_example(self, example, mode, hparams):
    """See base class."""
    example = super().preprocess_example(example, mode, hparams)
    
    # The keep_mask is ignored if scaled_loss = False.
    latents, keep_mask = self.make_latents(example, hparams)
    example["latents"] = latents
    # Only aggregate metrics (e.g., AUROC, AUPRC) for imputed labels.
    example["metrics_weights"] = tf.to_float(tf.equal(example["latents"],
                                                      self.unk_id))
    if hparams.scaled_loss:
      # Zero out targets for copied labels.
      keep_mask = tf.cast(keep_mask, tf.int64)
      zeroed = example["targets"] * (1 - keep_mask)
      # Add self.unk_id to the zeroed out targets.
      example["targets"] = zeroed + (keep_mask * self.unk_id)
    return example

  def load_names(self, namefile):
    """Loads DeepSEA label names from namefile.
    """
    return np.array(open(namefile).read().split(","))

  def get_overlapping_indices_for_cell_type(self, cell_type_1, cell_type_2):
    """Gets target indices for transcription factors for the intersection 
    of cell_type_1 and cell_type_2.

    Args:
      cell_type_1: Name of cell type 1 as a string.
      cell_type_2: Name of cell type 2 as a string.

    Returns:
      List of indices ~ FOR CELL 1 ~ that overlap with cell 2.
      These indices are listed in alphabetical order for consistency.
    """
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    namefile = dir_path + "/deepsea_label_names.txt"
    names = self.load_names(namefile)
    
    valid_cell_types = list(map(lambda x: x.split("|")[1], names))
    
    # Make sure cell type parameters can be found in our data. 
    assert(cell_type_1 in valid_cell_types, f"{cell_type_1} not in list of valid cell types")
    assert(cell_type_2 in valid_cell_types, f"{cell_type_2} not in list of valid cell types")
        
    # Get positions for both cell lines.
    cell_type_1_pos = [(i, j) for i, j in enumerate(names) \
                        if cell_type_1 in j]
    cell_type_2_pos = [(i, j) for i, j in enumerate(names) \
                        if cell_type_2 in j]

    # Get marks for each cell type
    cell_type_1_marks =  [i[1].split("|")[1] for i in cell_type_1_pos]
    cell_type_2_marks =  [i[1].split("|")[1] for i in cell_type_2_pos]
    
    # Get overlapping marks between both cell types.
    overlapping_marks = list(set(cell_type_1_marks) & set(cell_type_2_marks))

    cell_type_1_final_pos = [(i,j) for i, j in cell_type_1_pos if \
                                    j.split("|")[1] in overlapping_marks]
    cell_type_2_final_pos = [(i,j) for i, j in  cell_type_2_pos if \
                                    j.split("|")[1] in overlapping_marks]

    # Filter out duplicates for both cell types.
    cell_type_1_items = []
    seen = set()
    for item in cell_type_1_final_pos:
      if item[1] not in seen:
        seen.add(item[1])
        cell_type_1_items.append(item)

    cell_type_1_items = sorted(cell_type_1_items, key=lambda i: i[1]) 
    
    # Print out sorted marks.
    tf.logging.info("Marks for CellType %s: %s" 
                    % (cell_type_1, 
                    cell_type_1_items))

    cell_type_2_items = []
    seen = set()
    for item in cell_type_2_final_pos:
      if not item[1] in seen:
        seen.add(item[1])
        cell_type_2_items.append(item)

    cell_type_2_items = sorted(cell_type_2_items, key=lambda i: i[1])

    # Verify that TFs match between cell types.
    for i, item in enumerate(cell_type_2_items):
      assert(cell_type_2_items[i][1].split("|")[1] ==
             cell_type_1_items[i][1].split("|")[1])

    # These are the indices we are using for the cell type 1 model.
    cell_type_1_indices = list(map(lambda x: x[0], cell_type_1_items))
    return cell_type_1_indices


@registry.register_problem("genomics_binding_deepsea_tf")
class TranscriptionFactorDeepseaProblem(TftiDeepseaProblem):
  """DeepSEA Imputation problem for transcription factors (TFs)."""

  def targets_gather_indices(self):
    """Returns indices to gather `targets`, `latents` and `metrics_weights`.

    Returns:
      A list of indices between [0, self.num_binary_predictions).
    """
    return np.arange(125, 815)


  def preprocess_example(self, example, mode, hparams):
    """Slices latents and targets to only include indices of TF labels.

    Indices come from:
    (media.nature.com/original/nature-assets/nmeth/journal/v12/n10/extref/
        nmeth.3547-S3.xlsx)

    They include all TF rows (between 128 to 817). Specifically, the index is
    the row number in the spreadsheet - 3.

    See base class for method signature.
    """
    example = super().preprocess_example(example, mode, hparams)
    gather_indices = self.targets_gather_indices()
    for key in ["targets", "latents", "metrics_weights"]:
      example[key] = tf.gather(example[key], gather_indices)
    return example


################################################################################
########################## Cell Type Specific Problems #########################
################################################################################


@registry.register_problem("genomics_binding_deepsea_gm12878")
class Gm12878DeepseaProblem(TftiDeepseaProblem):
  """GM12878 Cell type specific imputation problem"""

  def targets_gather_indices(self):
    """Returns indices to gather `targets`, `latents` and `metrics_weights`.

    Returns:
      A list of indices between [0, self.num_binary_predictions).
    """
    return self.get_overlapping_indices_for_cell_type("GM12878", "H1-hESC")

  def preprocess_example(self, example, mode, hparams):
    example = super().preprocess_example(example, mode, hparams)
    # Indices for TF labels specific to GM12878 cell type.
    # These are ordered so TFs are alphabetical
    
    gather_indices = self.targets_gather_indices()
    
    # Argsort indices to preserve ordering.
    argsort_indices = np.argsort(gather_indices)
    gather_indices_sorted = np.sort(gather_indices)

    # Keep targets and latents corresponding to GM12878 (LCL cell line).
    targets = tf.gather(example["targets"], gather_indices_sorted)
    latents = tf.gather(example["latents"], gather_indices_sorted)
    metrics_weights = tf.gather(example["metrics_weights"],
                                gather_indices_sorted)
    
    # Ensure sure tensors are sorted by alphabetical TFs.
    example["targets"] = tf.gather(targets, argsort_indices)
    example["latents"] = tf.gather(latents, argsort_indices)
    example["metrics_weights"] = tf.gather(metrics_weights, argsort_indices)
    return example


@registry.register_problem("genomics_binding_deepsea_h1hesc")
class H1hescDeepseaProblem(TftiDeepseaProblem):
  """H1-hESC Cell type specific imputation problem"""

  def targets_gather_indices(self):
    """Returns indices to gather `targets`, `latents` and `metrics_weights`.

    Returns:
      A list of indices between [0, self.num_binary_predictions).
    """
    return self.get_overlapping_indices_for_cell_type("H1-hESC", "GM12878")

  def preprocess_example(self, example, mode, hparams):
    example = super().preprocess_example(example, mode, hparams)
    # Indices for TF labels specific to GM12878 cell type.
    # These are ordered so TFs are alphabetical
    
    gather_indices = self.targets_gather_indices()
    
    # Argsort indices to preserve ordering.
    argsort_indices = np.argsort(gather_indices)
    gather_indices_sorted = np.sort(gather_indices)

    # Keep targets and latents corresponding to H1-hESC.
    targets = tf.gather(example["targets"], gather_indices_sorted)
    latents = tf.gather(example["latents"], gather_indices_sorted)
    metrics_weights = tf.gather(example["metrics_weights"],
                                gather_indices_sorted)
    
    # Ensure sure tensors are sorted by alphabetical TFs.
    example["targets"] = tf.gather(targets, argsort_indices)
    example["latents"] = tf.gather(latents, argsort_indices)
    example["metrics_weights"] = tf.gather(metrics_weights, argsort_indices)
    return example


@registry.register_problem("genomics_binding_deepsea_multicell")
class TftiMulticellProblem(TftiDeepseaProblem):
  """Imputation problem accross multiple cell types"""

  @property
  def cell_types(self):
    return ['HeLa-S3', 'GM12878', 'HepG2', 'K562','H1-hESC']

  @property
  def test_cell_type(self):
    return self.cell_types[-1]

  def get_overlapping_indices_multicell(self):
    """Gets target indices for transcription factors for the intersection
    of call cell types.

    Returns:
      Dict of lists of indices in each cell of the intersection of all cells.
      These indices are listed in alphabetical order for consistency.
    """

    dir_path = os.path.dirname(os.path.realpath(__file__))
    namefile = dir_path + "/deepsea_label_names.txt"
    names = self.load_names(namefile)

    valid_cell_types = list(map(lambda x: x.split("|")[0], names))

    # Make sure cell type parameters can be found in our data.
    for cell_type in self.cell_types:
      assert(cell_type in valid_cell_types,
       f"{cell_type} not in list of valid cell types")

    # Get positions for all cell lines and untreated assays.
    # {cell: [(index, full mark name)]}
    position_dict = {cell_type: [(i, j) for i, j in enumerate(names)
                         if cell_type in j and j.split("|")[2] == "None"]
                         for cell_type in self.cell_types}

    # Get marks for each cell type, to find intersection.
    # {cell: mark type}
    mark_dict = {cell_type:
                     [i[1].split("|")[1] for i in position_dict[cell_type]]
                     for cell_type in self.cell_types}

    # Get overlapping marks between all cell types.
    overlapping_marks = list(set.intersection(*[set(i) for i in mark_dict.values()]))

    # Filter out non-overlapping marks.
    # {cell: [(index, full mark name)]}
    final_position_dict = {cell_type: [(i,j) for i, j in
                                        position_dict[cell_type] if
                                        j.split("|")[1] in overlapping_marks]
                                        for cell_type in self.cell_types}

    # Filter out duplicates for all cell types.
    # {cell: [(index, full mark name)]}
    cell_items_dict = {}
    for cell_type in self.cell_types:
      cell_type_items = []
      seen = set()
      for item in final_position_dict[cell_type]:
        if item[1].split("|")[1] not in seen:
          seen.add(item[1].split("|")[1])
          cell_type_items.append(item)

      cell_items_dict[cell_type] = sorted(cell_type_items, key=lambda i: i[1])

    # Verify that TFs match between cell types.
    for cell_type_1 in self.cell_types:
      for cell_type_2 in self.cell_types:
        for i, _ in enumerate(cell_items_dict[cell_type]):
          assert(cell_items_dict[cell_type_1][i][1].split("|")[1] ==
                 cell_items_dict[cell_type_2][i][1].split("|")[1])

    # These are the indices we are using.
    # {cell: [indices sorted alphabetically]}
    cell_indices = {cell_type: list(map(lambda x:
                     x[0], cell_items_dict[cell_type]))
                     for cell_type in self.cell_types}

    valid_tfs = sorted(list(map(lambda x: x[1].split('|')[1],
                                cell_items_dict[self.cell_types[0]])))

    tf.logging.info("Selected marks for cell types %s: %s" % (self.cell_types, valid_tfs))

    return cell_indices, valid_tfs

  def preprocess_example(self, example, mode, hparams):
    """Makes one example for each cell type, including only intersecting marks.

    See base class for method signature.
    """

    base_example = super().preprocess_example(example, mode, hparams)

    gather_indices, _ = self.get_overlapping_indices_multicell()

    dataset = None

    for cell_type in self.cell_types:
      # Do not put test in train set!
      if cell_type != self.test_cell_type:

        example = base_example.copy()

        for key in ["targets", "latents", "metrics_weights"]:
          example[key] = tf.gather(example[key], gather_indices[cell_type])

        # Each example is added to a new dataset, and the datasets are appended.
        new_data = tf.data.Dataset.from_tensors(example)
        if not dataset:
          dataset = new_data
        else:
          dataset.concatenate(new_data)

    return dataset


@registry.register_problem("genomics_binding_deepsea_multicell_eval")
class TftiMulticellEvalProblem(TftiMulticellProblem):
  """Evaluates TftiMulticellProblem on the held out cell type."""

  def preprocess_example(self, example, mode, hparams):
    """Makes one example for each cell type, including only intersecting marks.

    See base class for method signature.
    """

    example = TftiDeepseaProblem.preprocess_example(self, example, mode, hparams)

    gather_indices, _ = self.get_overlapping_indices_multicell()

    # Only eval on test.
    cell_type = self.test_cell_type

    for key in ["targets", "latents", "metrics_weights"]:
      example[key] = tf.gather(example[key], gather_indices[cell_type])

    dataset = tf.data.Dataset.from_tensors(example)

    return dataset

################################################################################
################################### MODELS #####################################
################################################################################


@registry.register_model("tfti_transformer")
class TftiTransformer(transformer.Transformer):
  """A stack of transformer layers.

  Transforms a set of partially observed latent labels into a complete label
  set. The encoder takes in as inputs embedded DNA sequence. The decoder takes
  in as inputs a latent binary label set in (0, 1, ?) and outputs a complete
  label set in (0, 1).
  """

  def tensorboard_summaries(self, logits, features, labels):
    """A method that is called in `self.top(...)` that logs to TensorBoard.

    This is called in `self.top(...)` because it is the only place where
    both the logits and features (including labels) are available.

    Args:
      logits: A float Tensor with shape [batch_size, nlabels, 1, 1].
      features: A features dict mapping keys to Tensors.
      targets: An int Tensor with shape [batch_size, nlabels, 1, 1] that takes
        values in (0, 1).

    Returns:
      None.
    """
    if not tf.contrib.eager.in_eager_mode():
      weighted_auroc = tf.multiply(*average_auroc(logits, labels, features))
      weighted_auprc = tf.multiply(*average_auprc(logits, labels, features))
      tf.summary.scalar("metrics/average_auroc", weighted_auroc)
      tf.summary.scalar("metrics/average_auprc", weighted_auprc)
      # Logging AUC metrics for individual labels.
      # TODO(alexyku): can we display multple curves on the same plot?
      # tf.summary.scalars("metrics/set_auroc", set_auroc(logits, targets))
      # tf.summary.scalars("metrics/set_auprc", set_auprc(logits, targets))

  def top(self, body_output, features):
    logits = super().top(body_output, features)
    # TensorBoard summaries.
    self.tensorboard_summaries(logits, features, features["targets"])
    return logits
  
  def body(self, features):
    """Transformer main model_fn.

    Args:
      features: Map of features to the model. Should contain the following:
        "inputs": Transformer inputs [batch_size, input_length, hidden_dim]
        "targets": Target decoder outputs.
            [batch_size, decoder_length, hidden_dim]
        "target_space_id"

    Returns:
      Final decoder representation. [batch_size, decoder_length, hidden_dim]
    """
    hparams = self._hparams

    hparams.ffn_layer = "conv_relu_conv"

    encoder_output, encoder_decoder_attention_bias = self.encode(
        inputs=features["inputs"],
        target_space=features["target_space_id"],
        hparams=hparams,
        features=features)
    # No positional embeddings on decoder side.

    hparams.ffn_layer = "dense_relu_dense"

    decoder_output = self.decode(
        decoder_input=common_layers.flatten4d3d(features["latents"]),
        encoder_output=encoder_output,
        encoder_decoder_attention_bias=encoder_decoder_attention_bias,
        decoder_self_attention_bias=None,  # No masking.
        hparams=hparams)
    return decoder_output


################################################################################
################################## HPARAMS #####################################
################################################################################


@registry.register_hparams("tfti_transformer_base")
def tfti_transformer_base():
  """Hparams extends `transformer_base`."""
  hparams = transformer.transformer_base()
  hparams.batch_size = 64
  hparams.add_hparam("multigpu", False)
  hparams.add_hparam("pos_weight", 25)
  hparams.add_hparam("gpu_compat", False)
  hparams.add_hparam("latent_keep_prob", 0.5)
  hparams.add_hparam("filter_negatives", False)
  hparams.add_hparam("scaled_loss", False)
  return hparams


@registry.register_hparams("tfti_transformer_debug")
def tfti_transformer_debug():
  """Hparams for debugging."""
  hparams = tfti_transformer_base()
  hparams.batch_size = 2
  hparams.num_hidden_layers = 2
  hparams.hidden_size = 8
  hparams.num_heads = 2
  hparams.latent_keep_prob = 0.5
  hparams.pos_weight = 10
  return hparams
