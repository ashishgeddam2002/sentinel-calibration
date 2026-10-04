"""Split conformal prediction for two classes (0 = off, 1 = on).

Nonconformity score: s = 1 - p(true class). Small score = the model agrees with the label.
Calibration threshold: the ceil((n+1)(1-alpha))/n empirical quantile of the calibration scores.
Prediction set: every class c with 1 - p(c) <= threshold.
"""
from __future__ import annotations

import math
from typing import Dict, Mapping

import numpy as np


def class_probs(p_on) -> np.ndarray:
    """Turn p(class 1) of shape (n,) into an (n, 2) matrix [p(off), p(on)]."""
    p_on = np.asarray(p_on, dtype=float)
    return np.stack([1.0 - p_on, p_on], axis=1)


def nonconformity(p_on, y) -> np.ndarray:
    """s = 1 - p(true class)."""
    probs = class_probs(p_on)
    y = np.asarray(y, dtype=int)
    return 1.0 - probs[np.arange(len(y)), y]


def conformal_quantile(scores, alpha: float) -> float:
    """Finite-sample-corrected quantile: the k-th smallest score, k = ceil((n+1)(1-alpha)).

    Equivalent to the quantile at level ceil((n+1)(1-alpha))/n. If k > n (too few
    calibration points for this alpha) the answer is +inf, i.e. the set is everything.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    scores = np.sort(np.asarray(scores, dtype=float))
    n = len(scores)
    if n == 0:
        return math.inf
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return math.inf
    return float(scores[k - 1])


def prediction_sets(p_on, qhat) -> np.ndarray:
    """Boolean (n, 2) matrix: entry [i, c] is True if class c is in the set for sample i.

    `qhat` is one number or an array of n thresholds (one per sample).
    """
    probs = class_probs(p_on)
    return (1.0 - probs) <= np.asarray(qhat, dtype=float).reshape(-1, 1) + 1e-12


def set_sizes(sets) -> np.ndarray:
    return np.asarray(sets).sum(axis=1)


def covered(sets, y) -> np.ndarray:
    """True where the true label is inside the prediction set."""
    y = np.asarray(y, dtype=int)
    return np.asarray(sets)[np.arange(len(y)), y]


def pooled_calibrate(p_on, y, alpha: float) -> float:
    """One threshold from all calibration data, ignoring conditions."""
    return conformal_quantile(nonconformity(p_on, y), alpha)


def mondrian_calibrate(p_on, y, cond, alpha: float) -> Dict[int, float]:
    """One threshold per condition (Mondrian = condition-stratified conformal)."""
    p_on, y, cond = np.asarray(p_on), np.asarray(y), np.asarray(cond)
    scores = nonconformity(p_on, y)
    return {int(c): conformal_quantile(scores[cond == c], alpha) for c in np.unique(cond)}


def mondrian_thresholds_for(cond, qhats: Mapping[int, float]) -> np.ndarray:
    """Look up each sample's threshold; a condition never seen in calibration gets +inf."""
    return np.array([qhats.get(int(c), math.inf) for c in np.asarray(cond)], dtype=float)


def predict_pooled(p_on, qhat: float) -> np.ndarray:
    return prediction_sets(p_on, qhat)


def predict_mondrian(p_on, cond, qhats: Mapping[int, float]) -> np.ndarray:
    return prediction_sets(p_on, mondrian_thresholds_for(cond, qhats))
