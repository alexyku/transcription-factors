from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

# Dependency imports

from tensor2tensor.bin import t2t_trainer
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import decoding
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import usr_dir

import tensorflow as tf
import numpy as np
import tfti

def get_config(problem, model, hparams_set, hparams, checkpoint_path):
  """Returns an inference config HParam object."""
  config = tf.contrib.training.HParams()
  config.problem = problem
  config.model = model
  config.hparams_set = hparams_set
  config.hparams = hparams
  config.data_dir = "."  # Unused.
  config.checkpoint_path = os.path.expanduser(checkpoint_path)
  return config


def get_problem_model_hparams(config):
  """Constructs problem, model, and hparams objects from a config."""
  hparams = trainer_lib.create_hparams(
      config.hparams_set,
      config.hparams,
      data_dir=os.path.expanduser(config.data_dir),
      problem_name=config.problem)
  problem = registry.problem(config.problem)
  model = registry.model(config.model)(hparams, tf.estimator.ModeKeys.EVAL)
  return (problem, model, hparams)


def get_preprocess_example_fn(config):
  """Returns a function that preprocesses a single example."""
  problem, model, hparams = get_problem_model_hparams(config)
  encoders = problem.get_feature_encoders(config.data_dir)
  
  with tf.Graph().as_default():
    inputs_ph = tf.placeholder(dtype=tf.int32)
    targets_ph = tf.placeholder(dtype=tf.int32)
    keep_mask_ph = tf.placeholder(dtype=tf.bool)

    inputs = tf.reshape(inputs_ph, shape=[problem.input_sequence_length])
    targets = tf.reshape(targets_ph, shape=[problem.num_binary_predictions])
    keep_mask = tf.reshape(keep_mask_ph, shape=[problem.num_binary_predictions])
    
    features = {"inputs": inputs, "targets": targets, "latent_keep_mask": keep_mask}
    features = problem.preprocess_example(features, tf.estimator.ModeKeys.EVAL, hparams)
    
    sess = tf.Session()
    
    def preprocess_example_fn(inputs, targets, keep_mask):
      """Preprocesses a single example to the problem specifications.

      Args:
        inputs: An ACTGN string of length 1000 or an ascii-encoded integer list.
          ex. "...NNNNACTGACTGNNNN..." or [..., 78, 65, 67, 84, 71, ...].
        targets: A binary string of length 919 or an binary integer list.
          ex. "...0000100100001100..." or [..., 0, 0, 1, 0, 0, 1, 1, ...].
        keep_mask: A binary string of length 919 or an boolean list.
          ex. "...0000100100001100..." or [..., True, False, True, ...].

      Returns:
        Dictionary containing:
          "inputs": An integer array with shape [preprocessed_length, 1, 1].
          "targets": A binary array with shape [nlabels, 1, 1].
          "latents": An integer array with shape [nlabels, 1, 1]
          "metrics_weights": A float array with shape [nlabels, 1, 1].
      """
      if isinstance(inputs, str):
        inputs = encoders["inputs"].encode(inputs)
      if isinstance(targets, str):
        targets = encoders["targets"].encode(targets)
      if isinstance(keep_mask, str):
        keep_mask = encoders["targets"].encode(keep_mask)
        keep_mask = list(map(bool, keep_mask))
      feed = {inputs_ph: inputs, targets_ph: targets, keep_mask_ph: keep_mask}
      return sess.run(features, feed)
    
    return preprocess_example_fn


def get_preprocess_batch_fn(config):
  """Returns a function that preprocesses a batch of examples."""
  preprocess_example_fn = get_preprocess_example_fn(config)
  
  def preprocess_batch_fn(inputs_iter, targets_iter, keep_mask_iter):
    """Preprocesses a batch of examples to the problem specifications.

    NOTE: assumes iterables are of the same length. This length is the batch size.

    Args:
      inputs_iter: An iterable of ACTGN string of length 1000 or an ascii-encoded integer list.
        ex. [..., "...NNNNACTGACTGNNNN...", ...] or [..., [..., 78, 65, 67, 84, 71, ...], ...].
      targets: An iterable of binary string of length 919 or an binary integer list.
        ex. [..., "...0000100100001100...", ...] or [..., [..., 0, 0, 1, 0, 0, 1, 1, ...], ...].
      keep_mask: An iterable of binary string of length 919 or an boolean list.
        ex. [..., "...0000100100001100...", ...] or [..., [..., True, False, True, ...], ...].

    Returns:
      Dictionary containing:
        "inputs": An integer array with shape [batch_size, preprocessed_length, 1, 1].
        "targets": A binary array with shape [batch_size, nlabels, 1, 1].
        "latents": An integer array with shape [batch_size, nlabels, 1, 1]
        "metrics_weights": A float array with shape [batch_size, nlabels, 1, 1].
    """
    results = []
    for inputs, targets, keep_mask in zip(inputs_iter, targets_iter, keep_mask_iter):
      result = preprocess_example_fn(inputs, targets, keep_mask)
      results.append(result)

    example_batch = {
        "inputs": np.stack([x["inputs"] for x in results]),
        "targets": np.stack([x["targets"] for x in results]),
        "latents": np.stack([x["latents"] for x in results]),
        "metrics_weights": np.stack([x["metrics_weights"] for x in results]),
    }
    return example_batch
  
  return preprocess_batch_fn


def get_inference_fn(config):
  """Returns an inference function from a config."""
  with tf.Graph().as_default():
    problem, model, hparams = get_problem_model_hparams(config)
    encoders = problem.get_feature_encoders(config.data_dir)
    assert isinstance(problem, tfti.TftiDeepseaProblem)
    
    preprocessed_inputs_length = int(np.ceil(problem.input_sequence_length / problem.chunk_size))
    preprocessed_targets_length = len(problem.targets_gather_indices())
    
    # batched
    inputs_ph = tf.placeholder(dtype=tf.int32, shape=[None, preprocessed_inputs_length, 1, 1])
    targets_ph = tf.placeholder(dtype=tf.int32, shape=[None, preprocessed_targets_length, 1, 1])
    latents_ph = tf.placeholder(dtype=tf.int32, shape=[None, preprocessed_targets_length, 1, 1])
    metrics_weights_ph = tf.placeholder(dtype=tf.float32, shape=[None, preprocessed_targets_length, 1, 1])
    
    features = {
        "inputs": inputs_ph,
        "targets": targets_ph,
        "latents": latents_ph,
        "metrics_weights": metrics_weights_ph,
        "target_space_id": tf.constant(0)
    }

    logits, _ = model(features)
    labels = features["targets"]
    predictions = tf.nn.sigmoid(logits)
    
    # Restore variables.
    all_variables = tf.contrib.slim.get_variables_to_restore()
    variables_to_restore = [v for v in all_variables if "global_step" not in v.name]
    global_step_init = [v.initializer for v in all_variables if "global_step" in v.name]

    saver = tf.train.Saver(variables_to_restore)

    sess = tf.Session()
    sess.run(global_step_init)
    saver.restore(sess, config.checkpoint_path)
    
    # Infer function.
    def inference_fn(example_batch):
      """Makes a forward pass through the network.

      Args:
        example_batch: The output of preprocess_batch_fn.

      Returns:
        Dictionary containing:
          "labels": A binary integer array with shape [batch_size, nlabels].
          "logits": A float array with shape [batch_size, nlabels], values in (-inf, inf).
          "predictions": Sigmoid function applied to logits, values in [0, 1].
      """
      feed = {
          inputs_ph: example_batch["inputs"],
          targets_ph: example_batch["targets"],
          latents_ph: example_batch["latents"],
          metrics_weights_ph: example_batch["metrics_weights"]
      }
      fetch = {"labels": labels, "logits": logits, "predictions": predictions}
      return sess.run(fetch, feed)
    
    return inference_fn