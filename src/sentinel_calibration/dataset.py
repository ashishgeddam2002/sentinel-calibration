"""Parse recording file names, labels and sidecar .txt files. Never crashes on odd names.

File name pattern:  DATE_TIME_TYPE_LABEL_FLAGS..._PHONE.ext
  e.g. 2026-10-04_17-32-56_audio_fan-ON-step3_dist-TBD_has-thump_nord.wav
Label pattern:      DEVICE-STATE[-extra-tokens]   e.g. fan-off-3m, fan-ON-step3, fan-off-3m-roomA
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

AUDIO_EXTS = (".wav",)
DEFAULT_GROUP = "all"


@dataclass
class Label:
    raw: str
    device: Optional[str] = None
    state: Optional[str] = None          # "on" / "off" / None if unknown
    distance_m: Optional[float] = None   # from a token like "3m"
    step: Optional[int] = None           # from a token like "step3"
    room: Optional[str] = None           # from "roomA" or "room-A"
    extras: List[str] = field(default_factory=list)  # any token we did not recognise (kept)

    @property
    def y(self) -> Optional[int]:
        return {"on": 1, "off": 0}.get(self.state)


@dataclass
class Recording:
    path: str
    parse_ok: bool
    date: Optional[str] = None
    time: Optional[str] = None
    type: Optional[str] = None
    label: Label = field(default_factory=lambda: Label(""))
    flags: List[str] = field(default_factory=list)      # raw flags, unknown ones included
    flag_values: Dict[str, str] = field(default_factory=dict)  # "dist-TBD" -> {"dist": "TBD"}
    phone: Optional[str] = None
    sidecar: Dict[str, str] = field(default_factory=dict)

    @property
    def group(self) -> str:
        return self.label.room or DEFAULT_GROUP


_LABEL_RE = re.compile(r"^(?P<device>[A-Za-z]+)-(?P<state>on|off)(?:-(?P<rest>.*))?$", re.IGNORECASE)
_DIST_RE = re.compile(r"^(\d+(?:\.\d+)?)m$", re.IGNORECASE)
_STEP_RE = re.compile(r"^step(\d+)$", re.IGNORECASE)
_ROOM_RE = re.compile(r"^room(.+)$", re.IGNORECASE)


def parse_label(raw: str) -> Label:
    """Parse e.g. 'fan-ON-step3' or 'fan-off-3m-roomA'. Unrecognised tokens go to `extras`."""
    label = Label(raw=raw)
    m = _LABEL_RE.match(raw.strip())
    if not m:
        label.extras = [t for t in raw.split("-") if t]
        return label
    label.device = m.group("device").lower()
    label.state = m.group("state").lower()
    tokens = [t for t in (m.group("rest") or "").split("-") if t]
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if (d := _DIST_RE.match(t)):
            label.distance_m = float(d.group(1))
        elif (s := _STEP_RE.match(t)):
            label.step = int(s.group(1))
        elif t.lower() == "room" and i + 1 < len(tokens):
            label.room = tokens[i + 1]
            i += 1
        elif (r := _ROOM_RE.match(t)):
            label.room = r.group(1)
        else:
            label.extras.append(t)
        i += 1
    return label


def parse_filename(path: str) -> Recording:
    """Parse DATE_TIME_TYPE_LABEL_FLAGS..._PHONE.ext. Fewer than 5 parts -> parse_ok=False."""
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.split("_")
    if len(parts) < 5:
        # best effort: keep what we can, never raise
        rec = Recording(path=path, parse_ok=False)
        if len(parts) >= 4:
            rec.date, rec.time, rec.type = parts[0], parts[1], parts[2]
            rec.label = parse_label(parts[3])
        return rec
    flags = parts[4:-1]
    kv = {}
    for fl in flags:
        key, _, val = fl.partition("-")
        kv[key] = val
    return Recording(path=path, parse_ok=True, date=parts[0], time=parts[1], type=parts[2],
                     label=parse_label(parts[3]), flags=flags, flag_values=kv, phone=parts[-1])


def read_sidecar(path: str) -> Dict[str, str]:
    """Read key=value lines from the .txt next to a recording. Missing file -> {}.

    Blank lines and lines starting with # are skipped; lines without '=' are kept under
    the key '_line<N>' so nothing is silently lost.
    """
    side = os.path.splitext(path)[0] + ".txt"
    out: Dict[str, str] = {}
    if not os.path.isfile(side):
        return out
    with open(side, encoding="utf-8", errors="replace") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, sep, val = line.partition("=")
            if sep:
                out[key.strip()] = val.strip()
            else:
                out[f"_line{n}"] = line
    return out


def load_recording(path: str) -> Recording:
    rec = parse_filename(path)
    rec.sidecar = read_sidecar(path)
    # fall back to the sidecar when the file name did not give a usable label / phone
    if rec.label.state is None and "label" in rec.sidecar:
        rec.label = parse_label(rec.sidecar["label"])
    if rec.phone is None:
        rec.phone = rec.sidecar.get("phone")
    return rec


def load_folder(folder: str) -> List[Recording]:
    """All audio recordings in a folder (non-recursive), sorted by file name."""
    names = sorted(n for n in os.listdir(folder) if n.lower().endswith(AUDIO_EXTS))
    return [load_recording(os.path.join(folder, n)) for n in names]
