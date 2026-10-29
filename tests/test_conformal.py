import math

import numpy as np
import pytest

from sentinel_calibration import conformal as cf


def test_nonconformity_is_one_minus_true_class_prob():
    s = cf.nonconformity([0.9, 0.9, 0.2], [1, 0, 0])
    assert np.allclose(s, [0.1, 0.9, 0.2])


def test_quantile_finite_sample_correction():
    scores = np.arange(1, 10) / 10  # n = 9: 0.1 ... 0.9
    # k = ceil(10 * 0.9) = 9 -> the largest score
    assert cf.conformal_quantile(scores, 0.1) == pytest.approx(0.9)
    # k = ceil(10 * 0.8) = 8
    assert cf.conformal_quantile(scores, 0.2) == pytest.approx(0.8)


def test_quantile_infinite_when_too_few_points():
    assert cf.conformal_quantile([0.1] * 5, 0.1) == math.inf  # k = ceil(6*0.9) = 6 > 5
    assert cf.conformal_quantile([], 0.1) == math.inf


def test_quantile_rejects_bad_alpha():
    with pytest.raises(ValueError):
        cf.conformal_quantile([0.1], 1.5)


def test_prediction_sets_and_sizes():
    sets = cf.prediction_sets([0.95, 0.5, 0.05], 0.3)
    # p=0.5 gives both scores 0.5 > 0.3: an EMPTY set is possible with a small threshold
    assert sets.tolist() == [[False, True], [False, False], [True, False]]
    assert cf.set_sizes(sets).tolist() == [1, 0, 1]
    assert cf.set_sizes(cf.prediction_sets([0.5], 0.6)).tolist() == [2]
    assert cf.prediction_sets([0.5], math.inf).all()


def test_covered():
    sets = cf.prediction_sets([0.95, 0.05], 0.3)
    assert cf.covered(sets, [1, 1]).tolist() == [True, False]


def test_mondrian_gives_one_threshold_per_condition():
    p = np.array([0.9, 0.8, 0.7, 0.6] * 5)
    y = np.ones(20, dtype=int)
    cond = np.array([0, 0, 1, 1] * 5)
    q = cf.mondrian_calibrate(p, y, cond, 0.1)
    assert set(q) == {0, 1}
    assert q[0] < q[1]  # condition 1 has less confident scores -> larger threshold
    th = cf.mondrian_thresholds_for([0, 1, 7], q)
    assert th[0] == q[0] and th[1] == q[1] and th[2] == math.inf


def test_pooled_coverage_guarantee_when_exchangeable():
    rng = np.random.default_rng(1)
    cover = []
    for _ in range(300):
        p_cal, p_test = rng.random(100), rng.random(400)
        y_cal, y_test = (rng.random(100) < p_cal).astype(int), (rng.random(400) < p_test).astype(int)
        q = cf.pooled_calibrate(p_cal, y_cal, 0.1)
        cover.append(cf.covered(cf.predict_pooled(p_test, q), y_test).mean())
    assert np.mean(cover) >= 0.9 - 0.01
