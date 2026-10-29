"""Seeded simulator of a day/night two-class classifier with a distribution shift.

The classifier sees a 1-D raw score z drawn from Normal(+margin, noise) when the fan is on
and Normal(-margin, noise) when it is off, and answers p(on) = sigmoid(scale * z).
By day `scale` equals the true log-likelihood ratio, so the model is honest. At night the
margin shrinks (less signal), the noise grows, and the scale is inflated (over-confidence).
All of it is controlled by `SimConfig.shift` in [0, 1]; shift = 0 means no shift.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

DAY, NIGHT = 0, 1
CONDITIONS = ("day", "night")
STATES = ("fan-off", "fan-on")


@dataclass(frozen=True)
class SimConfig:
    # daylight signal
    day_margin: float = 1.5          # half-distance between the two class means
    day_noise: float = 1.0           # standard deviation of the raw score
    # how the classifier turns a score into a probability (fitted for daylight)
    score_scale: float = 3.0         # = 2 * margin / noise**2 -> honest by day
    # night distribution shift
    shift: float = 1.0               # 0 = no shift, 1 = full shift
    night_margin_loss: float = 0.5   # fraction of the margin lost at shift = 1
    night_noise_gain: float = 1.0    # extra noise (relative) at shift = 1
    night_overconf_gain: float = 0.5 # extra probability sharpness (relative) at shift = 1
    # data mix
    night_fraction: float = 0.3      # share of night samples when mixing conditions
    p_on: float = 0.5                # prior probability of "fan-on"
    seed: int = 0

    def with_(self, **kw) -> "SimConfig":
        return replace(self, **kw)

    def condition_params(self, cond: int):
        """Return (margin, noise, scale) used for a condition."""
        if cond == DAY:
            return self.day_margin, self.day_noise, self.score_scale
        s = self.shift
        return (
            self.day_margin * (1.0 - s * self.night_margin_loss),
            self.day_noise * (1.0 + s * self.night_noise_gain),
            self.score_scale * (1.0 + s * self.night_overconf_gain),
        )

    def rng(self, offset: int = 0) -> np.random.Generator:
        return np.random.default_rng([self.seed, offset])


@dataclass
class Samples:
    p_on: np.ndarray   # classifier's probability of "on"
    y: np.ndarray      # true state, 0 = off, 1 = on
    cond: np.ndarray   # 0 = day, 1 = night

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, mask) -> "Samples":
        return Samples(self.p_on[mask], self.y[mask], self.cond[mask])


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))  # numerically stable


def sample_condition(cfg: SimConfig, n: int, cond: int, rng: np.random.Generator) -> Samples:
    """n samples from one condition."""
    margin, noise, scale = cfg.condition_params(cond)
    y = (rng.random(n) < cfg.p_on).astype(int)
    z = np.where(y == 1, margin, -margin) + noise * rng.standard_normal(n)
    return Samples(_sigmoid(scale * z), y, np.full(n, cond, dtype=int))


def sample_mixed(cfg: SimConfig, n: int, rng: np.random.Generator) -> Samples:
    """n samples where each is night with probability cfg.night_fraction."""
    cond = (rng.random(n) < cfg.night_fraction).astype(int)
    parts_p = np.empty(n)
    parts_y = np.empty(n, dtype=int)
    for c in (DAY, NIGHT):
        m = cond == c
        s = sample_condition(cfg, int(m.sum()), c, rng)
        parts_p[m], parts_y[m] = s.p_on, s.y
    return Samples(parts_p, parts_y, cond)
