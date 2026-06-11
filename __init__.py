"""libdpy: Shared library for course code.

Subpackages:
- visualization: plotting utilities re-exported from utilities/visualization
- privacy_mechanisms: basic DP mechanisms wrappers
- attacks: classical auditing and reconstruction utilities
- k_anon: anonymization helpers
- ml: machine-learning related helpers (skeleton)
"""

__version__ = "0.1.0"

import sys
import os

# Try to include the class_assignment_solutions
SOLUTIONS_PATH = 'class_assignment_solutions'
if SOLUTIONS_PATH in os.listdir():
    sys.path.append(os.path.abspath('.'))
if SOLUTIONS_PATH in os.listdir('..'):
    sys.path.append(os.path.abspath('..'))
