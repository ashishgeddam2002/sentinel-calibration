"""Generate synthetic WAV files (no real recordings are ever used or stored)."""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
from scipy.io import wavfile


def synth_signal(rate: int = 16000, seconds: float = 4.0, noise_dbfs: float = -50.0,
                 pulse_hz: Optional[float] = None, pulse_depth: float = 0.8,
                 bump_at_s: Optional[float] = None, tone_hz: Optional[float] = None,
                 tone_dbfs: float = -40.0, seed: int = 0) -> np.ndarray:
    """White noise at a given rms level, optionally amplitude-modulated, with a tone or a bump."""
    rng = np.random.default_rng(seed)
    n = int(rate * seconds)
    t = np.arange(n) / rate
    x = rng.standard_normal(n) * 10 ** (noise_dbfs / 20)
    if pulse_hz:
        x = x * (1 + pulse_depth * np.sin(2 * np.pi * pulse_hz * t)) / np.sqrt(1 + pulse_depth ** 2 / 2)
    if tone_hz:
        x = x + np.sqrt(2) * 10 ** (tone_dbfs / 20) * np.sin(2 * np.pi * tone_hz * t)
    if bump_at_s is not None:
        i = int(bump_at_s * rate)
        x[i:i + 40] += 0.6 * np.hanning(40)
    return np.clip(x, -1, 1)


def write_wav(path: str, x: np.ndarray, rate: int = 16000) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    wavfile.write(path, rate, (np.clip(x, -1, 1) * 32767).astype(np.int16))


def make_demo_folder(folder: str, per_class: int = 12, seed: int = 0) -> int:
    """Write a synthetic folder of recordings with sidecar .txt files.

    Two rooms (roomA, roomB). Fan-ON is louder, has a low-frequency tone and a ~20 Hz pulse in
    some files; fan-OFF is quiet. Room B is a bit noisier, so room-to-room shift is visible.
    Returns the number of WAV files written.
    """
    rng = np.random.default_rng(seed)
    count = 0
    for room, room_noise in (("roomA", 0.0), ("roomB", 4.0)):
        for state in ("on", "off"):
            for i in range(per_class):
                base = -55.0 + room_noise + rng.normal(0, 2.0)
                if state == "on":
                    base += rng.normal(8.0, 3.0)
                x = synth_signal(noise_dbfs=base, seed=int(rng.integers(1 << 30)),
                                 pulse_hz=20.0 if (state == "on" and i % 2 == 0) else None,
                                 tone_hz=120.0 if state == "on" else None, tone_dbfs=base - 6)
                stem = f"2026-10-04_17-{count // 60:02d}-{count % 60:02d}_audio_fan-{state.upper()}-3m-{room}_dist-TBD_nord"
                write_wav(os.path.join(folder, stem + ".wav"), x)
                with open(os.path.join(folder, stem + ".txt"), "w") as f:
                    f.write(f"label=fan-{state}-3m-{room}\nphone=nord\nsource=synthetic\n")
                count += 1
    return count
