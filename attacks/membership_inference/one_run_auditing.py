import scipy.stats
import numpy as np
import pandas as pd
import math

from libdpy.ml.requires_tensorflow.dnn_models import FullyDeepConnectedModel
from libdpy.ml.training_utils import (
    calc_accuracy,
    shuffle_data,
    PrivateBatchLearningParameters,
)
from libdpy.ml.requires_tensorflow.dp_sgd import tensorflow_DP_SGD, calc_DP_SGD_epsilon
from libdpy.attacks.membership_inference.classical_auditing_utils import threshold_guesser


def one_run_auditing_experiement(
    labeled_data, score_func, parameters_to_loop_over, error_prob, delta, seed=None
):
    rng = np.random.default_rng(seed)
    all_results = pd.DataFrame(
        columns=[
            'subset_size',
            'noise_factor',
            'batch_size',
            'num_epochs',
            'learning_rate',
            'lower_eps',
            'upper_eps',
            'acc_synthetic_train',
            'acc_synthetic_test',
            'acc_labeled_train',
            'acc_labeled_test',
        ]
    )
    synthetic_data = generate_adversarial_patterns(num_samples=2048, seed=rng)

    for subset_size, noise_factor, batch_size, num_epochs, learning_rate, clipping_radius in zip(
        parameters_to_loop_over['subset_size_vec'],
        parameters_to_loop_over['noise_factor_vec'],
        parameters_to_loop_over['batch_size_vec'],
        parameters_to_loop_over['num_epochs_vec'],
        parameters_to_loop_over['learning_rate_vec'],
        parameters_to_loop_over['clipping_radius_vec'],
    ):
        FC_training_params = PrivateBatchLearningParameters(
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            batch_size=batch_size,
            clipping_radius=clipping_radius,
            noise_factor=noise_factor,
        )
        FC_model_generator = lambda data: tensorflow_DP_SGD(
            data, FullyDeepConnectedModel, FC_training_params
        )
        upper_eps = calc_DP_SGD_epsilon(
            noise_factor,
            delta,
            clipping_radius,
            batch_size,
            (int(len(synthetic_data['labels']) / 2) + subset_size),
            num_epochs,
        )
        lower_eps, acc_synthetic_train, acc_synthetic_test, acc_labeled_train, acc_labeled_test = (
            one_run_auditing(
                labeled_data,
                synthetic_data,
                FC_model_generator,
                score_func,
                subset_size,
                error_prob,
                delta,
                rng=rng,
            )
        )
        new_row = {
            'subset_size': subset_size,
            'noise_factor': noise_factor,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'lower_eps': lower_eps,
            'upper_eps': upper_eps,
            'acc_synthetic_train': acc_synthetic_train,
            'acc_synthetic_test': acc_synthetic_test,
            'acc_labeled_train': acc_labeled_train,
            'acc_labeled_test': acc_labeled_test,
        }
        all_results = pd.concat([all_results, pd.DataFrame([new_row])], ignore_index=True)
    return all_results


def one_run_auditing(
    labeled_data,
    synthetic_data,
    FC_model_generator,
    score_func,
    subset_size,
    error_prob,
    delta,
    rng: np.random.Generator | None = None,
):
    (
        scores_neg,
        scores_pos,
        acc_synthetic_train,
        acc_synthetic_test,
        acc_labeled_train,
        acc_labeled_test,
    ) = randomize_train_and_return_score(
        labeled_data, synthetic_data, FC_model_generator, score_func, subset_size, rng=rng
    )
    score_vec = np.sort(np.concatenate((scores_neg, scores_pos)))
    # compute the values in score_vec such that 100 datapoint are strictly smaller than it, another value that 100 are strictly larger than it
    interval = score_vec[200], np.percentile(score_vec, 50), score_vec[-200]
    # remove from both the negative and positive scores the scores that are in the interval
    scores_neg = [score for score in scores_neg if score < interval[0] or score > interval[2]]
    scores_pos = [score for score in scores_pos if score < interval[0] or score > interval[2]]
    threshold = interval[1]
    guesser = lambda x: threshold_guesser(x, threshold=threshold)
    eps = epsilon_lower_bound_given_responses(
        scores_neg,
        scores_pos,
        guesser,
        error_prob,
        delta,
        synthetic_db_size=len(synthetic_data['labels']),
    )
    return eps, acc_synthetic_train, acc_synthetic_test, acc_labeled_train, acc_labeled_test


def generate_adversarial_patterns(num_samples, seed=None):
    import numpy as np
    from scipy.ndimage import rotate

    rng = np.random.default_rng(seed)

    images = []
    seen_patterns = set()

    def hash_image(img):
        return tuple(map(tuple, img))

    def create_checkerboard_pattern(frequency=2, random_offset=True):
        img = np.zeros((28, 28))
        offset_x = rng.integers(0, frequency) if random_offset else 0
        offset_y = rng.integers(0, frequency) if random_offset else 0
        for i in range(28):
            for j in range(28):
                if ((i + offset_x) + (j + offset_y)) % frequency == 0:
                    img[i, j] = 1
        return img

    def create_sparse_pattern(num_pixels=5, cluster_size=2):
        img = np.zeros((28, 28))
        center_positions = rng.choice(28 * 28, num_pixels, replace=False)
        for pos in center_positions:
            i, j = pos // 28, pos % 28
            # Create small clusters around each point
            for di in range(-cluster_size, cluster_size + 1):
                for dj in range(-cluster_size, cluster_size + 1):
                    ni, nj = i + di, j + dj
                    if (
                        0 <= ni < 28 and 0 <= nj < 28 and rng.random() < 0.7
                    ):  # 70% chance to fill
                        img[ni, nj] = 1
        return img

    def create_symmetric_pattern(num_points=3):
        img = np.zeros((28, 28))
        center = (14, 14)
        for _ in range(num_points):
            # Generate point in first quadrant
            x = rng.integers(0, 14)
            y = rng.integers(0, 14)
            # Mirror across all quadrants
            points = [
                (center[0] + x, center[1] + y),
                (center[0] - x, center[1] + y),
                (center[0] + x, center[1] - y),
                (center[0] - x, center[1] - y),
            ]
            for px, py in points:
                if 0 <= px < 28 and 0 <= py < 28:
                    img[int(px), int(py)] = 1
        return img

    def create_fractal_pattern(depth=3):
        img = np.zeros((28, 28))

        def recursive_pattern(x, y, size, depth):
            if depth == 0 or size < 2:
                return
            img[int(x) : int(x + size), int(y) : int(y + size)] = 1
            new_size = size // 3
            for i in range(3):
                for j in range(3):
                    if (i + j) % 2 == 0:  # Skip some squares for pattern
                        recursive_pattern(x + i * new_size, y + j * new_size, new_size, depth - 1)

        recursive_pattern(0, 0, 27, depth)
        return img

    def create_periodic_pattern():
        img = np.zeros((28, 28))
        frequency_x = rng.uniform(0.5, 2.0)
        frequency_y = rng.uniform(0.5, 2.0)
        phase = rng.uniform(0, 2 * np.pi)
        for i in range(28):
            for j in range(28):
                if np.sin(frequency_x * i + phase) * np.cos(frequency_y * j) > 0:
                    img[i, j] = 1
        return img

    def create_random_walk(steps=100):
        img = np.zeros((28, 28))
        x, y = rng.integers(0, 28), rng.integers(0, 28)
        for _ in range(steps):
            img[int(x), int(y)] = 1
            dx, dy = rng.choice([-1, 0, 1], 2)
            x = np.clip(x + dx, 0, 27)
            y = np.clip(y + dy, 0, 27)
        return img

    def apply_random_transformation(img):
        # Randomly rotate
        angle = rng.uniform(0, 360)
        img = rotate(img, angle, reshape=False)

        # Randomly flip
        if rng.random() > 0.5:
            img = np.fliplr(img)
        if rng.random() > 0.5:
            img = np.flipud(img)

        # Threshold to ensure binary image
        img = (img > 0.5).astype(float)
        return img

    # Calculate patterns per type
    patterns_per_type = num_samples // 8
    duplicates_found = 0

    # Define base pattern generators
    pattern_generators = [
        lambda: create_checkerboard_pattern(frequency=rng.integers(2, 6), random_offset=True),
        lambda: create_sparse_pattern(
            num_pixels=rng.integers(3, 8), cluster_size=rng.integers(1, 4)
        ),
        lambda: create_symmetric_pattern(num_points=rng.integers(2, 6)),
        lambda: create_fractal_pattern(depth=rng.integers(2, 4)),
        lambda: create_periodic_pattern(),
        lambda: create_random_walk(steps=rng.integers(50, 150)),
        lambda: apply_random_transformation(create_checkerboard_pattern()),
        lambda: apply_random_transformation(create_symmetric_pattern()),
    ]

    images = []
    seen_patterns = set()

    # Generate patterns by randomly selecting generators
    while len(images) < num_samples:
        # Randomly choose a pattern generator
        generator = pattern_generators[rng.integers(len(pattern_generators))]
        img = generator()
        img_hash = hash_image(img)

        if img_hash in seen_patterns:
            # Generate a new pattern until we get a unique one
            while img_hash in seen_patterns:
                generator = pattern_generators[
                    rng.integers(len(pattern_generators))
                ]  # Choose a new random generator
                img = generator()
                img_hash = hash_image(img)

        seen_patterns.add(img_hash)
        images.append(img)

    # Convert to numpy array and assign random labels
    images = np.array(images)
    labels = rng.choice([0, 1], size=len(images))
    return {'data': images, 'labels': labels}


def randomize_train_and_return_score(
    labeled_data,
    synthetic_data,
    FC_model_generator,
    score_func,
    subset_size,
    rng: np.random.Generator | None = None,
):
    if rng is None:
        rng = np.random.default_rng()
    labeled_data = shuffle_data(labeled_data, rng=rng)
    synthetic_data = shuffle_data(synthetic_data, rng=rng)

    # train-test split
    labeled_data_train = {
        'data': labeled_data['data'][:subset_size],
        'labels': labeled_data['labels'][:subset_size],
    }
    labeled_data_test = {
        'data': labeled_data['data'][subset_size:],
        'labels': labeled_data['labels'][subset_size:],
    }

    # train-test split of the synthetic data
    synthetic_data_size = len(synthetic_data['labels'])
    half_ind = int(synthetic_data_size / 2)
    synthetic_data_train = {
        'data': synthetic_data['data'][:half_ind],
        'labels': synthetic_data['labels'][:half_ind],
    }
    synthetic_data_test = {
        'data': synthetic_data['data'][(half_ind + 1) :],
        'labels': synthetic_data['labels'][(half_ind + 1) :],
    }

    # concatenate the subset of synthetic data with the labeled data
    labeled_data_train_with_syn = labeled_data_train.copy()
    labeled_data_train_with_syn['data'] = np.concatenate(
        (labeled_data_train['data'], synthetic_data_train['data'])
    )
    labeled_data_train_with_syn['labels'] = np.concatenate(
        (labeled_data_train['labels'], synthetic_data_train['labels'])
    )
    # train the model
    model = FC_model_generator(labeled_data_train_with_syn)
    # compute the score of the model on the synthetic data
    score = score_func(model, synthetic_data)
    # return the accuracy of the model on the synthetic data that we trained on, did not train on, on the train data and the test data
    acc_synthetic_train = np.mean(calc_accuracy(model, synthetic_data_train))
    acc_synthetic_test = np.mean(calc_accuracy(model, synthetic_data_test))
    acc_labeled_train = np.mean(calc_accuracy(model, labeled_data_train))
    acc_labeled_test = np.mean(calc_accuracy(model, labeled_data_test))
    return (
        score[(half_ind + 1) :],
        score[:half_ind],
        acc_synthetic_train,
        acc_synthetic_test,
        acc_labeled_train,
        acc_labeled_test,
    )


def epsilon_lower_bound_given_responses(
    responses_neg, responses_pos, guesser, error_prob: float, delta, synthetic_db_size
) -> float:
    number_of_guesses = len(responses_pos) + len(responses_neg)
    num_pos_guesses_pos = np.sum([guesser(response) for response in responses_pos])
    num_neg_guesses_neg = len(responses_neg) - np.sum(
        [guesser(response) for response in responses_neg]
    )
    # print(f"Number of guesses: {number_of_guesses}")
    # print(f"Number of positive guesses on positive examples: {num_pos_guesses_pos}")
    # print(f"Number of negative guesses on negative examples: {num_neg_guesses_neg}")
    # print(f"Synthetic db size: {synthetic_db_size}")
    return get_eps_audit(
        synthetic_db_size,
        number_of_guesses,
        (num_pos_guesses_pos + num_neg_guesses_neg),
        delta,
        error_prob,
    )


def p_value_DP_audit(m, r, v, eps, delta):
    # Ensure inputs are valid
    assert 0 <= v <= r <= m, "Invalid input: 0 <= v <= r <= m must hold."
    assert eps >= 0, "Epsilon must be non-negative."
    assert 0 <= delta <= 1, "Delta must be between 0 and 1."
    q = 1 / (1 + math.exp(-eps))
    beta = scipy.stats.binom.sf(v - 1, r, q)
    alpha = 0
    summation = 0
    for i in range(1, v + 1):
        summation += scipy.stats.binom.pmf(v - i, r, q)
        if summation > i * alpha:
            alpha = summation / i

    # Final p-value
    p = beta + alpha * delta * 2 * m
    return min(p, 1)


# m =number of examples, each included independently with probability 0.5
# r =number of guesses (i.e. excluding abstentions)
# v =number of correct guesses by auditor
# p =1-confidence e.g. p=0.05 corresponds to 95%
# output:lower bound on eps i.e. algorithm is not(eps,delta)-DP
def get_eps_audit(m, r, v, delta, p):
    # Ensure inputs are valid
    assert 0 <= v <= r <= m, "Invalid input: 0 <= v <= r <= m must hold."
    assert 0 <= delta <= 1, "Delta must be between 0 and 1."
    assert 0 < p < 1, "p must be between 0 and 1."

    eps_min = 0  # Maintain p_value_DP_audit(eps_min) < p
    eps_max = 1  # Maintain p_value_DP_audit(eps_max) >= p

    # Increment eps_max until p_value_DP_audit(eps_max) >= p
    while p_value_DP_audit(m, r, v, eps_max, delta) < p:
        eps_max += 1

    # Binary search to find eps_min
    for _ in range(30):
        eps = (eps_min + eps_max) / 2
        if p_value_DP_audit(m, r, v, eps, delta) < p:
            eps_min = eps
        else:
            eps_max = eps

    return eps_min
