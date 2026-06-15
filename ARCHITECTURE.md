# libdpy Architecture

A per-module reference for the `libdpy` package: what each subpackage and file
provides. For installation and a high-level overview see [README.md](README.md).

`libdpy/__init__.py` exposes the package version (`__version__`). Subpackages have
no `__init__.py` files — setuptools `find-packages` handles discovery. Course
datasets ship inside the package under `resources/*.csv` (declared as
`package-data` in `pyproject.toml`).

## `privacy_mechanisms/` — core DP mechanisms

| File | Provides |
|------|----------|
| `noise.py` | `laplace_noise`, noisy sum/median helpers (`Laplace_sum`, `Gaussian_sum`, …) |
| `exponential_mechanism.py` | Generic exponential mechanism, `get_exact_median` |
| `above_threshold.py` | Batch Above Threshold + `simulate_above_threshold` demo |
| `above_threshold_online.py` | Stateful `AboveThresholdOnline` class |
| `synthetic_data.py` | DP histograms, synthetic sampling, TV-distance comparison |

## `ml/` — machine learning with privacy (NumPy core, TensorFlow optional)

| File | Provides |
|------|----------|
| `data_types.py` | `LabeledData` type alias |
| `models.py` | Abstract `Model` / `GradientBasedModel`; `LogisticModel`, `NaiveModel` |
| `training_utils.py` | SGD, DP-SGD, DP naive training, batching, metrics, parameter dataclasses |
| `dp_utils.py` | Config dataclasses + re-exports from `mnist_utils` |
| `mnist_utils.py` | MNIST loading/preprocessing (OpenML + optional Keras) |
| `requires_tensorflow/dp_sgd.py` | Native TF DP-SGD loop, ε/δ/σ accounting (`calc_DP_SGD_*`), experiment runner |
| `requires_tensorflow/dnn_models.py` | `DNNModel`, `FullyConnectedModel`, `FullyDeepConnectedModel` (Keras wrappers) |
| `requires_tensorflow/utils.py` | TF SGD, output-perturbation σ/ε, naive-DP experiment |
| `requires_tensorflow/early_stopping_utils.py` | Train/val split, per-epoch DP-SGD step, non-DP early stopping (DP variant is a student exercise) |
| `requires_tensorflow/visualizations.py` | Per-epoch loss curves for TF models |

Everything under `requires_tensorflow/` needs the `[ml]` extra installed.

## `attacks/` — privacy attacks and auditing

| File | Provides |
|------|----------|
| `membership_inference/classical_auditing_utils.py` | Clopper–Pearson-style ε lower bounds, estimation error |
| `membership_inference/elements_selection.py` | Membership scores (log-loss, gradient norm, angle), extreme-element selection, dataset construction |
| `membership_inference/one_run_auditing.py` | One-run DP auditing experiment (TF), parameter sweeps |
| `reconstruction/reconstruction_attacks.py` | White/black/API-box reconstruction queries and orchestration |
| `reconstruction/solvers.py` | Brute-force int-prog, LR-rounding, LP solvers |
| `reconstruction/instances.py` | Reconstruction instance generation, candidate enumeration/elimination, 2D slab region, and the OLS out-of-cube counterexample |
| `reconstruction/geometry.py` | 3D lecture geometry: cube corners, slab classification, plane/slab polygons, feasible clouds |

## `k_anon/`

| File | Provides |
|------|----------|
| `anonymizer.py` | `Anonymizer` (wraps `anonypy`), `generate_synthetic_data` |

## `visualization/` — plotting (matplotlib / plotly / ipywidgets)

Excluded from the coverage scope (plotting code).

| File | Provides |
|------|----------|
| `roc_plots.py` | Interactive ROC curves, distribution comparisons, threshold sliders |
| `statistical_plots.py` | Histograms, CLT demos, KDE, interactive statistical plots |
| `privacy_plots.py` | Privacy bounds, ROC from distributions, interactive ε widgets |
| `ml_plots.py` / `ml_visualization.py` | Weight images, prediction grids, confusion matrices |
| `private_visualization.py` | Accuracy/ε/noise comparisons, multi-model weight grids |
| `one_run_auditing.py` | ε lower/upper bound plots from auditing experiments |
| `interactive_plot.py` | Reusable `InteractivePlot` widget framework |

## `assignment_specific/` — per-assignment scaffolding

Part of the package, fully implemented (no student stubs live here — those belong
only in the assignment notebooks).

| Subfolder | Supports |
|-----------|----------|
| `exponential_mechanism/` | week 7 class + lecture |
| `smarter_noise_addition/` | week 5 class |
| `beyond_noise_addition/` | week 6 class (above threshold) |
| `dp_sgd/` | week 10 class (`default_noise` identity = intentional "no noise" baseline) |
| `early_stopping/` | week 11 class (includes `secretly_test_sensitivity` hash-check autograder) |
| `ml_and_overfitting/` | week 11 lecture |
| `privacy_auditing/` | week 4 class |
| `reconstruction/` | week 2 class (`mysterious_predictor`/`reconstructor` are implemented, intentionally opaque); `reconstruction_lecture_visualization.py` (matplotlib: query matrix, 2D slab widget, candidate elimination) and `reconstruction_3d_visualization.py` (Plotly: 3D slab explorer, OLS out-of-cube) for the reconstruction lecture |
| `k_anon/` | week 1 class |

## `resources/`

Course datasets bundled into the installed package (`FultonPUMS5full.csv`,
`Healthcare_preprocess.csv`). Other course datasets live under
`class_assignments/resources/` in the development repo and are not shipped here.
