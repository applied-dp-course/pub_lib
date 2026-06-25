from typing import Any, Callable, Tuple, List

import numpy as np

from dp_accounting import pld
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from libdpy.ml.data_types import LabeledData
from libdpy.ml.models import GradientBasedModel, Model
from libdpy.ml.training_utils import PrivateBatchLearningParameters, calc_accuracy
from libdpy.ml import mnist_utils as _mnist_utils


def create_cnn_model():
    # Backwards-compatible re-export; implementation lives in `libdpy.ml.mnist_utils`.
    return _mnist_utils.create_cnn_model()


def create_dense_model():
    # Backwards-compatible re-export; implementation lives in `libdpy.ml.mnist_utils`.
    return _mnist_utils.create_dense_model()


def load_and_preprocess_data():
    # Backwards-compatible re-export; implementation lives in `libdpy.ml.mnist_utils`.
    return _mnist_utils.load_and_preprocess_data()


def prepare_data(x_train, x_test, use_cnn):
    # Backwards-compatible re-export; implementation lives in `libdpy.ml.mnist_utils`.
    return _mnist_utils.prepare_data(x_train, x_test, use_cnn)


def slice_data(x, y, batch_size=None):
    if batch_size:
        return tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)
    return tf.data.Dataset.from_tensor_slices((x, y))


def check_gradients(gradients, param):
    for g in gradients:
        if g is not None and tf.reduce_max(tf.abs(g)) > param:
            return None
    return gradients


def _global_l2_norm(grad_list: list[tf.Tensor | None]) -> tf.Tensor:
    """Compute global L2 norm across a list of gradient tensors (ignores None)."""
    squares = []
    for g in grad_list:
        if g is None:
            continue
        squares.append(tf.reduce_sum(tf.square(g)))
    if not squares:
        return tf.constant(0.0, dtype=tf.float32)
    return tf.sqrt(tf.add_n(squares))


def _clip_grads(
    grad_list: list[tf.Tensor | None], l2_norm_clip: float
) -> list[tf.Tensor | None]:
    """Clip gradients by global norm to have at most `l2_norm_clip`."""
    norm = _global_l2_norm(grad_list)
    # Avoid division by 0.
    clip = tf.cast(l2_norm_clip, norm.dtype)
    div = tf.maximum(norm, tf.cast(1e-12, norm.dtype))
    factor = tf.minimum(tf.constant(1.0, dtype=norm.dtype), clip / div)
    clipped: list[tf.Tensor | None] = []
    for g in grad_list:
        clipped.append(None if g is None else g * factor)
    return clipped


def tensorflow_DP_SGD(
    labeled_data: LabeledData, model_class, params: PrivateBatchLearningParameters, seed=None
) -> GradientBasedModel:
    if seed is not None:
        # Seed TensorFlow's global RNG so the stateful shuffle/noise ops below are reproducible.
        tf.random.set_seed(seed)
    batch_size = params.batch_size if params.batch_size is not None else len(labeled_data['labels'])
    num_batches = len(labeled_data['labels']) // batch_size
    data = labeled_data['data'][: num_batches * batch_size]
    labels = labeled_data['labels'][: num_batches * batch_size]

    num_classes = len(np.unique(labeled_data['labels']))
    model = model_class.get_model(num_classes)

    # Native DP-SGD implementation (no tensorflow_privacy):
    # - compute per-example gradients
    # - clip each example by global L2 norm (params.clipping_radius)
    # - sum clipped gradients over batch
    # - add Gaussian noise with stddev = noise_multiplier * l2_norm_clip
    # - average and apply via a standard optimizer
    #
    # Note: This is intentionally simple/educational and not optimized for speed.
    optimizer = tf.keras.optimizers.SGD(learning_rate=params.learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction="none")

    weights_arr = [model.get_weights()]

    # Make sure data/labels are tensors for slicing in the loop.
    data_tf = tf.convert_to_tensor(data)
    labels_tf = tf.convert_to_tensor(labels)

    num_samples = int(data_tf.shape[0])
    if num_samples == 0:
        return model_class(weights_arr)

    l2_norm_clip = float(params.clipping_radius)
    noise_multiplier = float(params.noise_factor)
    noise_stddev = noise_multiplier * l2_norm_clip

    for _ in range(params.num_epochs):
        # Shuffle each epoch (similar to Keras default behavior).
        perm = tf.random.shuffle(tf.range(num_samples))
        data_tf = tf.gather(data_tf, perm)
        labels_tf = tf.gather(labels_tf, perm)

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            x_batch = data_tf[start:end]
            y_batch = labels_tf[start:end]

            # Accumulate clipped per-example gradients.
            summed_grads: list[tf.Tensor] = [
                tf.zeros_like(v, dtype=tf.float32) for v in model.trainable_variables
            ]
            bsz = int(x_batch.shape[0])
            if bsz == 0:
                continue

            for i in range(bsz):
                x_i = x_batch[i : i + 1]
                y_i = y_batch[i : i + 1]
                with tf.GradientTape() as tape:
                    preds = model(x_i, training=True)
                    # Ensure shape (1,) loss for sparse labels.
                    per_ex_loss = loss_fn(y_i, preds)
                    loss = tf.reduce_sum(per_ex_loss)
                grads = tape.gradient(loss, model.trainable_variables)
                clipped = _clip_grads(grads, l2_norm_clip)

                for j, g in enumerate(clipped):
                    if g is None:
                        continue
                    summed_grads[j] = summed_grads[j] + tf.cast(g, tf.float32)

            # Add Gaussian noise to the summed gradients.
            noised_grads: list[tf.Tensor] = []
            for g in summed_grads:
                if noise_stddev > 0:
                    noise = tf.random.normal(shape=tf.shape(g), stddev=noise_stddev, dtype=g.dtype)
                    g = g + noise
                noised_grads.append(g / float(bsz))

            optimizer.apply_gradients(zip(noised_grads, model.trainable_variables))

        weights_arr.append(model.get_weights())
    return model_class(weights_arr)


def create_PLD_for_DP_SGD(
    sigma: float, clipping_radius: float, batch_size: int, sample_size: int, num_epochs: int
):
    norm_sigma = sigma / clipping_radius
    sampling_prob = float(batch_size) / sample_size
    num_compositions = int(sample_size / batch_size * num_epochs)
    pl_dist = pld.privacy_loss_distribution.from_gaussian_mechanism(
        norm_sigma,
        pessimistic_estimate=True,
        value_discretization_interval=1e-4,
        sampling_prob=sampling_prob,
        use_connect_dots=True,
    )
    composed_pld = pl_dist.self_compose(num_compositions)
    return composed_pld


def calc_DP_SGD_epsilon(
    sigma: float,
    delta: float,
    clipping_radius: float,
    batch_size: int,
    sample_size: int,
    num_epochs: int,
) -> float:
    if sigma == 0:
        return np.inf
    composed_pld = create_PLD_for_DP_SGD(
        sigma, clipping_radius, batch_size, sample_size, num_epochs
    )
    return composed_pld.get_epsilon_for_delta(delta)


def calc_DP_SGD_delta(
    epsilon: float,
    sigma: float,
    clipping_radius: float,
    batch_size: int,
    sample_size: int,
    num_epochs: int,
) -> float:
    if sigma == 0 or epsilon == 0:
        return 1
    composed_pld = create_PLD_for_DP_SGD(
        sigma, clipping_radius, batch_size, sample_size, num_epochs
    )
    return composed_pld.get_delta_for_epsilon(epsilon)


def calc_DP_SGD_sigma(
    epsilon: float,
    delta: float,
    clipping_radius: float,
    batch_size: int,
    sample_size: int,
    num_epochs: int,
) -> float:
    if epsilon == 0:
        return np.inf
    sigma_upper_bound: float = 100
    search_params = pld.common.BinarySearchParameters(0, sigma_upper_bound)
    return pld.common.inverse_monotone_function(
        lambda sigma: calc_DP_SGD_delta(
            epsilon, (sigma), clipping_radius, batch_size, sample_size, num_epochs
        ),
        delta,
        search_params,
    )


def run_DP_SGD_experiment(
    train_labeled_data: LabeledData,
    test_labeled_data: LabeledData,
    noise_scale: np.ndarray[Any, np.dtype[np.float64]],
    training_params: PrivateBatchLearningParameters,
    model_generator: Callable,
    num_experiments: int,
    delta: float,
    seed=None,
) -> Tuple[
    List[Model], np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]
]:
    if seed is not None:
        # Seed TensorFlow's global RNG so model training across the experiment is reproducible.
        tf.random.set_seed(seed)
    params_size = len(noise_scale)
    models = []
    accuracy_array = np.zeros(params_size)
    epsilon_array = np.zeros(params_size)
    sample_size = len(train_labeled_data['labels'])
    batch_size = (
        training_params.batch_size if training_params.batch_size is not None else sample_size
    )

    for i in range(params_size):
        params = PrivateBatchLearningParameters(
            learning_rate=training_params.learning_rate,
            num_epochs=training_params.num_epochs,
            batch_size=batch_size,
            noise_factor=noise_scale[i],
            clipping_radius=training_params.clipping_radius,
        )
        model = model_generator(train_labeled_data, params)
        models.append(model)
        accuracy_array[i] = np.mean(
            [
                np.mean(
                    calc_accuracy(model_generator(train_labeled_data, params), test_labeled_data)
                )
                for _ in range(num_experiments)
            ]
        )
        epsilon_array[i] = calc_DP_SGD_epsilon(
            noise_scale[i],
            delta,
            params.clipping_radius / batch_size,
            batch_size,
            sample_size,
            params.num_epochs,
        )
    return models, accuracy_array, epsilon_array
