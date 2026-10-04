import numpy as np

from sentinel_calibration import evaluate as ev
from sentinel_calibration.__main__ import main
from sentinel_calibration.synth_audio import make_demo_folder


def test_logistic_separates_and_is_monotone():
    x = np.r_[np.linspace(-60, -50, 20), np.linspace(-45, -35, 20)]
    y = np.r_[np.zeros(20), np.ones(20)].astype(int)
    f = ev.fit_logistic_1d(x, y)
    assert f(-60) < 0.1 < 0.9 < f(-35)
    assert np.all(np.diff(f(np.sort(x))) >= 0)


def test_analyse_folder_end_to_end(tmp_path):
    folder = tmp_path / "rec"
    assert make_demo_folder(str(folder), per_class=8) == 32
    (folder / "garbage.wav").write_bytes(b"not a wav")        # must not crash the table
    out = ev.analyse_folder(str(folder), str(tmp_path / "fig"), str(tmp_path / "res"))
    assert len(out["rows"]) == 33
    assert any(r["error"] for r in out["rows"])
    assert (tmp_path / "fig" / "level_distributions.png").exists()
    assert out["logo"]["mode"] == "leave-one-group-out"
    assert {f["group"] for f in out["logo"]["folds"]} == {"A", "B"}
    assert out["logo"]["overall"]["accuracy"] > 0.6


def test_single_group_falls_back_to_leave_one_recording_out():
    rows = [dict(y=i % 2, group="all", steady_dbfs=-50.0 + 10 * (i % 2) + (i % 5) * 0.3) for i in range(30)]
    res = ev.leave_one_group_out(rows)
    assert "only one group" in res["mode"]
    assert len(res["folds"]) == 30


def test_cli_analyse(tmp_path, capsys):
    folder = tmp_path / "rec"
    make_demo_folder(str(folder), per_class=5)
    rc = main(["analyse", str(folder), "--figures", str(tmp_path / "f"), "--results", str(tmp_path / "r")])
    assert rc == 0 and "leave-one-group-out" in capsys.readouterr().out
