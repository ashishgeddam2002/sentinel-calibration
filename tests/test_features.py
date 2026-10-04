import numpy as np
import pytest

from sentinel_calibration import features as ft
from sentinel_calibration.synth_audio import synth_signal, write_wav

RATE = 16000


def sine(freq, dbfs_rms, seconds=3.0):
    t = np.arange(int(RATE * seconds)) / RATE
    return np.sqrt(2) * 10 ** (dbfs_rms / 20) * np.sin(2 * np.pi * freq * t)


def test_steady_level_of_constant_sine():
    assert ft.steady_level(sine(440, -30), RATE) == pytest.approx(-30, abs=0.2)


def test_steady_level_ignores_first_half_second_and_short_bumps():
    x = sine(440, -50, seconds=4)
    x[: RATE // 2] = 0.9                     # loud start: must be ignored
    x[2 * RATE: 2 * RATE + 200] += 0.8       # one short click: 25th percentile ignores it
    assert ft.steady_level(x, RATE) == pytest.approx(-50, abs=0.5)


def test_steady_level_nan_for_too_short_audio():
    assert np.isnan(ft.steady_level(np.zeros(RATE // 2), RATE))


def test_bump_threshold():
    assert not ft.has_bump(np.full(100, 0.3))   # -10.46 dBFS
    assert ft.has_bump(np.full(100, 0.35))      # -9.12 dBFS
    assert not ft.has_bump(np.array([]))


def test_band_levels_put_energy_in_the_right_band():
    lv = ft.band_levels(sine(500, -20), RATE)
    assert lv["band_300_1000"] == pytest.approx(-20, abs=0.5)
    assert all(lv[k] < -50 for k in lv if k != "band_300_1000")
    assert set(lv) == {f"band_{a}_{b}" for a, b in ft.BANDS}


def test_pulse_found_in_modulated_noise():
    for hz in (10.0, 20.0, 28.0):
        res = ft.find_pulse(synth_signal(noise_dbfs=-40, pulse_hz=hz, seconds=4), RATE)
        assert res["pulse_found"]
        assert res["pulse_hz"] == pytest.approx(hz, abs=0.5)


def test_pulse_not_found_in_plain_noise_or_outside_range():
    for seed in range(10):
        assert not ft.find_pulse(synth_signal(noise_dbfs=-40, seed=seed), RATE)["pulse_found"]
    # 50 Hz modulation is outside 8-30 Hz
    assert not ft.find_pulse(synth_signal(noise_dbfs=-40, pulse_hz=50.0), RATE)["pulse_found"]


def test_pulse_needs_enough_audio():
    assert not ft.find_pulse(synth_signal(seconds=1.2, pulse_hz=20.0), RATE)["pulse_found"]


def test_extract_features_from_wav_file(tmp_path):
    p = tmp_path / "t.wav"
    write_wav(str(p), synth_signal(noise_dbfs=-40, pulse_hz=15.0, bump_at_s=2.0), RATE)
    f = ft.extract_features(str(p))
    assert f["steady_dbfs"] == pytest.approx(-40, abs=1.5)
    assert f["bump"] is True and f["pulse_found"] is True
    assert f["duration_s"] == pytest.approx(4.0)
    assert "band_20_100" in f


def test_read_wav_stereo_and_float(tmp_path):
    from scipy.io import wavfile
    x = sine(300, -20, 1.0).astype(np.float32)
    wavfile.write(str(tmp_path / "s.wav"), RATE, np.stack([x, x], axis=1))
    y, rate = ft.read_wav(str(tmp_path / "s.wav"))
    assert y.ndim == 1 and rate == RATE and len(y) == len(x)
