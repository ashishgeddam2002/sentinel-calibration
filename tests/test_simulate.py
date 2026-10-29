import numpy as np

from sentinel_calibration import conformal as cf
from sentinel_calibration.simulate import DAY, NIGHT, SimConfig, sample_condition, sample_mixed


def test_same_seed_same_data_different_seed_different_data():
    cfg = SimConfig(seed=3)
    a = sample_mixed(cfg, 50, cfg.rng(0))
    b = sample_mixed(cfg, 50, cfg.rng(0))
    c = sample_mixed(cfg, 50, cfg.rng(1))
    assert np.array_equal(a.p_on, b.p_on) and np.array_equal(a.y, b.y)
    assert not np.array_equal(a.p_on, c.p_on)


def test_shapes_and_ranges():
    cfg = SimConfig()
    s = sample_mixed(cfg, 200, cfg.rng())
    assert len(s) == 200
    assert ((s.p_on >= 0) & (s.p_on <= 1)).all()
    assert set(np.unique(s.y)) <= {0, 1} and set(np.unique(s.cond)) <= {0, 1}


def test_night_is_worse_and_more_overconfident_than_day():
    cfg = SimConfig()
    rng = cfg.rng()
    day, night = (sample_condition(cfg, 20000, c, rng) for c in (DAY, NIGHT))
    acc = lambda s: ((s.p_on > 0.5) == s.y).mean()
    conf = lambda s: np.maximum(s.p_on, 1 - s.p_on).mean()
    assert acc(night) < acc(day) - 0.1
    # over-confidence = confidence minus accuracy
    assert conf(night) - acc(night) > conf(day) - acc(day) + 0.1


def test_zero_shift_makes_conditions_identical_in_law():
    cfg = SimConfig(shift=0.0)
    assert cfg.condition_params(DAY) == cfg.condition_params(NIGHT)


def test_pooled_undercovers_at_night_and_mondrian_does_not():
    cfg = SimConfig()
    rng = cfg.rng()
    pooled_night, mond_night = [], []
    for _ in range(40):
        cal = sample_mixed(cfg, 500, rng)
        qp = cf.pooled_calibrate(cal.p_on, cal.y, 0.1)
        qm = cf.mondrian_calibrate(cal.p_on, cal.y, cal.cond, 0.1)
        t = sample_condition(cfg, 2000, NIGHT, rng)
        pooled_night.append(cf.covered(cf.predict_pooled(t.p_on, qp), t.y).mean())
        mond_night.append(cf.covered(cf.predict_mondrian(t.p_on, t.cond, qm), t.y).mean())
    assert np.mean(pooled_night) < 0.8
    assert abs(np.mean(mond_night) - 0.9) < 0.03


def test_quick_experiment_runs(tmp_path):
    from sentinel_calibration.experiments import run_all
    out = run_all(str(tmp_path / "f"), str(tmp_path / "r"), quick=True)
    assert len(out["figures"]) == 4
    assert all((tmp_path / "f" / p.split("/")[-1]).exists() for p in out["figures"])
    assert (tmp_path / "r" / "main_table.csv").exists()
