"""Helpers for binary hypothesis testing and ROC visualizations."""

from scipy.stats import rv_continuous as Distribution


def FPR_from_threshold(dist_neg: Distribution, threshold: float):
    """
    Calculate the false positive rate when values above ``threshold`` are classified positive.
    """
    return 1 - dist_neg.cdf(threshold)


def TPR_from_threshold(dist_pos: Distribution, threshold: float):
    """
    Calculate the true positive rate when values above ``threshold`` are classified positive.
    """
    return 1 - dist_pos.cdf(threshold)


def threshold_from_FPR(dist_neg: Distribution, FPR: float):
    """
    Calculate the threshold that gives the requested false positive rate.

    This assumes the positive distribution is separated by shifting mass to larger values, so
    the optimal decision rule classifies values above the threshold as positive.
    """
    return dist_neg.ppf(1 - FPR)


def desicion_rule_params(
    dist0: Distribution,
    dist1: Distribution,
    param: float,
    param_type,
):
    """
    Return FPR, TPR, and thresholds for threshold decision rules.

    ``param`` is interpreted as either an FPR or a raw threshold according to ``param_type``.
    The function name keeps the original notebook spelling for compatibility.
    """
    from libdpy.visualization.roc_plots import ComparisonType

    kind = getattr(param_type, "value", param_type)
    if kind == ComparisonType.SAME_VAR.value:
        threshold = threshold_from_FPR(dist0, param)
        tpr = TPR_from_threshold(dist1, threshold)
        return param, tpr, [threshold]
    if kind == ComparisonType.GENERAL.value:
        fpr = FPR_from_threshold(dist0, param)
        tpr = TPR_from_threshold(dist1, param)
        return fpr, tpr, [param]
    raise ValueError("This setting is not supported")


def decision_rule_params(
    dist0: Distribution,
    dist1: Distribution,
    param: float,
    param_type,
):
    """Correctly-spelled alias for ``desicion_rule_params``."""
    return desicion_rule_params(dist0, dist1, param, param_type)
