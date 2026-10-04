from sentinel_calibration.dataset import (load_folder, load_recording, parse_filename,
                                          parse_label, read_sidecar)

NAME = "2026-10-04_17-32-56_audio_fan-ON-step3_dist-TBD_has-thump_nord.wav"


def test_parse_example_filename():
    r = parse_filename(NAME)
    assert r.parse_ok
    assert (r.date, r.time, r.type, r.phone) == ("2026-10-04", "17-32-56", "audio", "nord")
    assert r.label.state == "on" and r.label.y == 1 and r.label.step == 3
    assert r.flags == ["dist-TBD", "has-thump"]
    assert r.flag_values == {"dist": "TBD", "has": "thump"}


def test_label_case_insensitive_and_distance():
    for raw, state in (("fan-off-3m", "off"), ("fan-OFF-3m", "off"), ("Fan-On-3m", "on")):
        assert parse_label(raw).state == state
    assert parse_label("fan-off-3m").distance_m == 3.0


def test_room_tag_and_unknown_tokens_kept():
    a = parse_label("fan-off-3m-roomA")
    assert a.room == "A"
    b = parse_label("fan-on-room-kitchen-weird")
    assert b.room == "kitchen" and b.extras == ["weird"]


def test_unknown_flags_are_kept_and_nothing_crashes():
    r = parse_filename("2026-10-04_17-32-56_audio_fan-ON_zzz_foo-bar_qux_nord.wav")
    assert r.flags == ["zzz", "foo-bar", "qux"]
    for bad in ("weird.wav", "", "a_b.wav", "___.wav"):
        r = parse_filename(bad)
        assert r.parse_ok is False
    assert parse_label("???").state is None


def test_sidecar_read(tmp_path):
    (tmp_path / "x.wav").write_bytes(b"")
    (tmp_path / "x.txt").write_text("label=fan-off-3m\nphone = nord\n# comment\n\nrms_dbfs=-47.5\nstray line\n")
    side = read_sidecar(str(tmp_path / "x.wav"))
    assert side["label"] == "fan-off-3m" and side["phone"] == "nord" and side["rms_dbfs"] == "-47.5"
    assert "stray line" in side.values()
    assert read_sidecar(str(tmp_path / "missing.wav")) == {}


def test_sidecar_fallback_for_bad_filename(tmp_path):
    (tmp_path / "odd.wav").write_bytes(b"")
    (tmp_path / "odd.txt").write_text("label=fan-ON-3m\nphone=nord\n")
    r = load_recording(str(tmp_path / "odd.wav"))
    assert r.label.state == "on" and r.phone == "nord" and not r.parse_ok


def test_load_folder_sorted_and_only_wav(tmp_path):
    for n in ("b.wav", "a.wav", "c.txt"):
        (tmp_path / n).write_bytes(b"")
    assert [r.path.split("/")[-1] for r in load_folder(str(tmp_path))] == ["a.wav", "b.wav"]
