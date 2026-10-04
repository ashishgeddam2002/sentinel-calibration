"""Audio features from WAV files.

All levels are in dBFS: 0 dBFS = full scale (sample value 1.0), silence is floored at -120.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
from scipy.io import wavfile

FLOOR_DB = -120.0
SKIP_S = 0.5            # ignore the first 0.5 s (button press, handling noise)
WINDOW_S = 0.25         # level window length
BUMP_DBFS = -10.0       # any peak above this counts as a bump
BANDS: Tuple[Tuple[int, int], ...] = ((20, 100), (100, 300), (300, 1000), (1000, 3000), (3000, 8000))
PULSE_RANGE_HZ = (8.0, 30.0)
PULSE_RATIO_MIN = 6.0   # peak / median of the envelope spectrum needed to call it a pulse
ENV_CLIP_X_MEDIAN = 3.0  # envelope values above 3x the median are clipped
ENV_HOP_S = 0.005       # envelope sampled at 200 Hz


def to_db(x: float) -> float:
    """20*log10(x), floored at FLOOR_DB."""
    return FLOOR_DB if x <= 0 else max(FLOOR_DB, 20.0 * math.log10(x))


def read_wav(path: str) -> Tuple[np.ndarray, int]:
    """Read a WAV as mono float samples in [-1, 1] plus the sample rate."""
    rate, data = wavfile.read(path)
    if data.dtype == np.uint8:
        x = (data.astype(np.float64) - 128.0) / 128.0
    elif data.dtype == np.int16:
        x = data / 32768.0
    elif data.dtype == np.int32:
        x = data / 2147483648.0
    elif data.dtype in (np.float32, np.float64):
        x = data.astype(np.float64)
    else:
        raise ValueError(f"unsupported WAV sample type {data.dtype}")
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x, int(rate)


def window_levels(x: np.ndarray, rate: int, skip_s: float = SKIP_S, window_s: float = WINDOW_S) -> np.ndarray:
    """RMS level (dBFS) of each non-overlapping window after the first `skip_s` seconds."""
    x = x[int(skip_s * rate):]
    w = int(window_s * rate)
    if w <= 0 or len(x) < w:
        return np.array([])
    n = len(x) // w
    frames = x[: n * w].reshape(n, w)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    return np.array([to_db(r) for r in rms])


def steady_level(x: np.ndarray, rate: int) -> float:
    """25th percentile of the window levels: the quiet floor, robust to short bumps."""
    lv = window_levels(x, rate)
    return float(np.percentile(lv, 25)) if len(lv) else float("nan")


def has_bump(x: np.ndarray) -> bool:
    """True if any single sample peaks above -10 dBFS."""
    return bool(len(x)) and to_db(float(np.max(np.abs(x)))) > BUMP_DBFS


def band_levels(x: np.ndarray, rate: int) -> Dict[str, float]:
    """Level (dB re full-scale mean-square) in each band. Bands sum to about the total level."""
    out = {}
    if len(x) < 2:
        return {f"band_{lo}_{hi}": FLOOR_DB for lo, hi in BANDS}
    w = np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x * w)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1.0 / rate)
    total = spec.sum()
    ms = float(np.mean(x ** 2))
    for lo, hi in BANDS:
        sel = (freqs >= lo) & (freqs < hi)
        p = ms * spec[sel].sum() / total if total > 0 else 0.0
        out[f"band_{lo}_{hi}"] = FLOOR_DB if p <= 0 else max(FLOOR_DB, 10.0 * math.log10(p))
    return out


def amplitude_envelope(x: np.ndarray, rate: int) -> Tuple[np.ndarray, float]:
    """Frame-RMS envelope sampled every ENV_HOP_S seconds; returns (envelope, envelope rate)."""
    hop = max(1, int(ENV_HOP_S * rate))
    n = len(x) // hop
    if n == 0:
        return np.array([]), 1.0 / ENV_HOP_S
    env = np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(axis=1))
    return env, rate / hop


def find_pulse(x: np.ndarray, rate: int) -> Dict[str, float]:
    """Look for a periodic pulse (8-30 Hz) in the amplitude envelope, after the first 0.5 s.

    Returns dict(pulse_found, pulse_hz, pulse_ratio). `pulse_ratio` is the strongest
    envelope-spectrum peak in 8-30 Hz divided by the median of the 1-90 Hz spectrum.
    Needs at least ~1 s of audio; otherwise it reports not found.
    """
    none = dict(pulse_found=False, pulse_hz=float("nan"), pulse_ratio=0.0)
    env, env_rate = amplitude_envelope(x[int(SKIP_S * rate):], rate)
    if len(env) < env_rate:  # < 1 s
        return none
    env = np.minimum(env, ENV_CLIP_X_MEDIAN * np.median(env))  # one loud click must not swamp the spectrum
    env = env - env.mean()
    nfft = 1 << int(math.ceil(math.log2(len(env) * 4)))   # zero-pad for a finer frequency grid
    spec = np.abs(np.fft.rfft(env * np.hanning(len(env)), nfft))
    freqs = np.fft.rfftfreq(nfft, 1.0 / env_rate)
    ref = spec[(freqs >= 1) & (freqs <= min(90, env_rate / 2 - 1))]
    median = float(np.median(ref)) if len(ref) else 0.0
    sel = (freqs >= PULSE_RANGE_HZ[0]) & (freqs <= PULSE_RANGE_HZ[1])
    if median <= 0 or not sel.any():
        return none
    i = int(np.argmax(np.where(sel, spec, -1)))
    ratio = float(spec[i] / median)
    return dict(pulse_found=ratio >= PULSE_RATIO_MIN, pulse_hz=float(freqs[i]), pulse_ratio=ratio)


def extract_features(path: str) -> Dict[str, object]:
    """All features for one WAV file as a flat dict."""
    x, rate = read_wav(path)
    feats: Dict[str, object] = dict(
        duration_s=len(x) / rate, sample_rate=rate,
        rms_dbfs=to_db(float(np.sqrt(np.mean(x ** 2)))) if len(x) else FLOOR_DB,
        steady_dbfs=steady_level(x, rate),
        peak_dbfs=to_db(float(np.max(np.abs(x)))) if len(x) else FLOOR_DB,
        bump=has_bump(x),
    )
    feats.update(band_levels(x, rate))
    feats.update(find_pulse(x, rate))
    return feats
