"""Compatibility imports for hypothesis-testing helpers.

The canonical location is ``libdpy.hypothesis_testing``. This package is kept so
older notebooks that imported from ``assignment_specific`` continue to run.
"""

from libdpy.hypothesis_testing import (
    FPR_from_threshold,
    TPR_from_threshold,
    decision_rule_params,
    desicion_rule_params,
    threshold_from_FPR,
)

__all__ = [
    "FPR_from_threshold",
    "TPR_from_threshold",
    "threshold_from_FPR",
    "desicion_rule_params",
    "decision_rule_params",
]
