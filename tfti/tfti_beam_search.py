from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Dependency imports

import tfti_batched_inference
from tensor2tensor.bin import t2t_decoder
from tensor2tensor.utils import registry

import tensorflow as tf
import numpy as np


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("cache_to_file", None, "")


# Data generators


def get_batch_generator(iterable, batch_size, yield_partial_batch=True):
  """Yields batches from a generator.

  Example for batch size of 2:
    (0, 1, 2, 3, 4) ==> ((0, 1), (2, 3), (4,))
  If `yield_partial_batch=False`:
    (0, 1, 2, 3, 4) ==> ((0, 1), (2, 3))

  Args:
    iterable: a python iterable.
    batch_size: int. batch size to yield.
    yield_partial_batch: bool. if the length of the iterable is not divisible by
      the batch size, yield the remainder if true, and don't otherwise. See the
      example in the docstring.
  Yields:
    chunks of the iterable of size batch_size (or less if partial batch).
  """
  iterator = iter(iterable)
  yield_next = True
  while yield_next:
    buffer = []
    for _ in range(batch_size):
      try:
        example = next(iterator)
        buffer.append(example)
      except StopIteration:
        yield_next = False
        break
    if len(buffer) == batch_size or yield_partial_batch:
      yield buffer


def get_validation_response_generator_fn():
  """Returns a generator function.

  Higher-order fn so we don't load the validation data set multiple times. 

  Returns:
    A generator function that yields labels and predictions.
  """
  config = tfti_batched_inference.get_config(
      problem=FLAGS.problem,
      model=FLAGS.model,
      hparams_set=FLAGS.hparams_set,
      hparams=FLAGS.hparams,
      checkpoint_path=FLAGS.checkpoint_path)

  # Configure the inference pipeline
  (problem, model, hparams
    ) = tfti_batched_inference.get_problem_model_hparams(config)
  preprocess_batch_fn = tfti_batched_inference.get_preprocess_batch_fn(config)
  inference_fn = tfti_batched_inference.get_inference_fn(config)

  # Load the validation data set
  tmp_dir = os.path.expanduser(FLAGS.tmp_dir)
  problem.maybe_download_and_unzip(tmp_dir)
  generator = problem.generator(tmp_dir, is_training=False)
  raw_examples_list = list(generator)

  def validation_response_generator(keep_mask, batch_size):
    """Yields labels and predictions from the validation set.

    Args:
      keep_mask: a boolean mask of length 919 or a binary string.
        [... True, False, True ...] or "... 0100001000100 ..."
      batch_size: int. the batch size to use for batch parallelism.

    Yields:
      tuple containing:
        labels: an int array with shape [nlabels].
        predictions: a float array with shape [nlabels].
    """
    raw_examples_iter = iter(raw_examples_list)  # Reuse the data
    raw_batch_iter = get_batch_generator(raw_examples_iter, batch_size)

    for i, raw_batch in enumerate(raw_batch_iter):
      tf.logging.info(f"Processing batch {i} of size {batch_size}.")

      # Preprocess examples
      example_batch = preprocess_batch_fn(
          [x["inputs"] for x in raw_batch],
          [x["targets"] for x in raw_batch],
          [keep_mask] * batch_size)

      # Fetch your response batch
      response_batch = inference_fn(example_batch)

      # Yield them one at a time (cleaner api?)
      for j in range(response_batch["labels"].shape[0]):
        labels = response_batch["labels"][j]
        predictions = response_batch["predictions"][j]
        yield labels.ravel(), predictions.ravel()


def get_multiclass_auroc(y_test, y_score):
  """Computes multiclass auroc.

  Args:
    y_test: int array with shape [nexamples, nlabels], labels.
    y_score: float array with shape [nexamples, nlabels], predictions.

  Returns:
    tuple containing:
      fpr: dict mapping key --> false pos rate.
      tpr: dict mapping key --> true pos rate.
      thresholds: dict mapping key --> thresholds.
      roc_auc: dict mapping key --> area under roc.

    key contains range(nlabels) ints and "micro" which is the average auc.
  """
  from sklearn.metrics import roc_curve, auc
  # y_test, y_score --> shape: [batch_size, n_classes]
  n_classes = y_test.shape[1]

  # Compute ROC curve and ROC area for each class
  fpr = dict()
  tpr = dict()
  thresholds = dict()
  roc_auc = dict()
  for i in range(n_classes):
    fpr[i], tpr[i], thresholds[i] = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

  # Compute micro-average ROC curve and ROC area
  fpr["micro"], tpr["micro"], thresholds["micro"] = roc_curve(
      y_test.ravel(), y_score.ravel())
  roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
  
  return fpr, tpr, thresholds, roc_auc


# Beam search


# Cached operations
__cache__ = dict()


def get_deepsea_label_names():
  """Deepsea label names.

  Returns:
    A string array with shape [919].
  """
  if "get_deepsea_label_names" in __cache__:
    return __cache__["get_deepsea_label_names"]
  else:
    pwd = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(pwd, "deepsea_label_names.txt")
    deepsea_label_names = np.array(open(path).read().split(","))
    __cache__["get_deepsea_label_names"] = deepsea_label_names
    return deepsea_label_names


def get_names_to_indices():
  """For creating latents.

  Returns:
    A dict mapping label names (string) to indices (int).
    The index corresponds to the position of the label in the 919 label space.
  """
  if "get_names_to_indices" in __cache__:
    return __cache__["get_names_to_indices"]
  else:
    problem = registry.problem(FLAGS.problem)
    indices = problem.targets_gather_indices()

    # Convert: '8988T|DNase|None' --> 'DNase'
    names = get_deepsea_label_names()[indices]
    names = [x.split("|")[1] for x in names]
    names_to_indices = dict(zip(names, indices))
    __cache__["get_names_to_indices"] = names_to_indices
    return names_to_indices


def get_initial_state():
  """Returns the initial beam search state.

  States are frozensets containing mark strings:
    ex. frozenset({'DNase', 'CHD2', 'RFX5', 'EZH2'})

  Returns:
    The empty frozenset: frozenset({})
  """
  return frozenset({})


def get_children_states(parent_state):
  """Returns a list of children state.

  Args:
    parent_state: a frozenset of label names (marks).

  Returns:
    A list of cildren states. Children states have one more label name (mark)
    than their parent state.
  """
  names_to_indices = get_names_to_indices()
  names = frozenset(names_to_indices.keys())
  return [parent_state.union({x}) for x in marks - state]


def get_keep_mask(state):
  """Creates a latent keep mask.

  Args:
    state: a frozenset of label names (marks).

  Returns:
    A boolean string of length 919. ex. "...000010101010...".
  """
  names_to_indices = get_names_to_indices()
  state_indices = [mark_to_index[x] for x in state]
  keep_mask = "".join(str(int(i in indices)) for i in range(919))
  return keep_mask


def get_heuristic_fn(cache=None):
  """Creates a cached heuristic function.

  Args:
    cache: a dictionary mapping states to something used in the heuristic
      compute

  Returns:
    A tuple containing:
      heuristic_fn: mapping a state --> score
      cache: a dict that can reinitilize this function.
  """
  cache = cache or dict()  # cache: states --> something (naively a heuristic score)

  def score_from_cache(x):
    # transforms a cache entry to a score.
    # currently the identity function mapping score --> score. but if we store
    # a dict with more info we might use something more sophisticated.
    return x

  def heuristic_fn(state):
    if state in cache:
      return score_from_cache(cache[state])
    else:
      score = 1.0  # TODO - this.
      cache[state] = score
      return score_from_cache(score)

  return heuristic_fn, cache


def beam_search(num_beams, max_depth, cache=None):
  """Runs beam search.

  Args:
    num_beams: int. number of beams in beam search.
    max_depth: int. how deep in the search tree to go.
    cache: dict. the heuristic_fn cache or None.

  Returns:
    A tuple containing:
      beams: list of (score, state) tuples of length num_beams.
        where each state is size max_depth.
      cache: dict. the heuristic_fn cache.
  """
  heuristic_fn, cache = get_heuristic_fn(cache)
  
  init_state = get_initial_state()
  beams = [(heuristic_fn(init_state), init_state)] 

  for _ in range(max_depth):
    new_beams = []  # Next level in the game tree
    for _, parent_state in beams:
      for child_state in get_children_states(parent_state):
        new_beams.append((heuristic_fn(child_state), child_state))
    beams = sorted(set(new_beams), reverse=True)[:num_beams]

  # Sometime beam search is not monotonic. That is, a state with n labels might
  # be worse than one with n-1. Thus, we keep the cache to be safe.
  return beams, cache
