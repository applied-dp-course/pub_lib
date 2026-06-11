# libdpy

A Python library for **Differential Privacy** and **Privacy-Preserving Machine Learning**, used in the Applied Privacy course.

## Installation

Install from the public repository:

```bash
pip install "libdpy @ git+https://github.com/applied-dp-course/pub_lib.git"
```

### Optional extras

```bash
# Machine learning features (TensorFlow, dp-accounting)
pip install "libdpy[ml] @ git+https://github.com/applied-dp-course/pub_lib.git"

# Development tools (includes [ml])
pip install "libdpy[dev] @ git+https://github.com/applied-dp-course/pub_lib.git"
```

Requires Python 3.10.11 or newer.

## Modules

- **`privacy_mechanisms/`** — Laplace, Gaussian, Above Threshold, Exponential Mechanism
- **`ml/`** — DP-SGD, private training, neural network helpers
- **`attacks/`** — Reconstruction and membership-inference utilities
- **`k_anon/`** — K-anonymity helpers
- **`visualization/`** — Privacy and ML plotting utilities
- **`assignment_specific/`** — Course-specific helpers

## Quick check

```bash
python -c "import libdpy; print(libdpy.__version__)"
```

## Development

This directory is the standalone publishable package. It is automatically synced from the private [code_base_dev](https://github.com/applied-dp-course/code_base_dev) repository when changes merge to `main`.

To work on the full course project (notebooks, tests, assignments), clone `code_base_dev` instead.

## License

Educational use as part of the Applied Privacy course.