# sentinel-calibration

SENTINEL is an M.Tech robotics project that answers yes/no home-status questions
("is the fan on?") at night from a phone's camera, flash and microphone. It must say
**"not sure"** instead of guessing.

This repository tests one claim, in simulation:

> A confidence threshold calibrated in daylight **under-covers at night**, but calibrating
> **separately for each condition** (Mondrian = condition-stratified conformal prediction)
> keeps the coverage guarantee in **every** condition.

It also contains a small audio-feature pipeline (level, bumps, band levels, a 8-30 Hz pulse
search) and a folder analyser for real recordings. Everything runs offline; all data used here
is synthetic. New to conformal prediction? Read [HOW_IT_WORKS.md](HOW_IT_WORKS.md) first.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"        # numpy, scipy, matplotlib (+ pytest)
```

## Reproduce every figure (one command)

```bash
python -m sentinel_calibration reproduce          # ~5 seconds on a laptop
python -m sentinel_calibration reproduce --quick  # fewer seeds, even faster
```

| File | What it shows | Made by |
|---|---|---|
| `figures/fig1_coverage_by_condition.png` | **Main figure.** Coverage by condition, pooled vs per-condition | `experiments` |
| `figures/fig2_coverage_vs_shift.png` | Coverage as the day/night shift grows | `experiments` |
| `figures/fig3_night_coverage_vs_ncal.png` | Night coverage vs calibration-set size | `experiments` |
| `figures/fig4_setsize_abstention_vs_shift.png` | Cost: set size and abstention at night | `experiments` |
| `figures/level_distributions.png` | Fan ON vs OFF level histogram (synthetic demo audio) | `analyse` |
| `results/*.csv, *.md` | Tables behind the figures | both |

The pieces can also be run separately:

```bash
python -m sentinel_calibration experiments [--quick]     # simulation only
python -m sentinel_calibration make-demo-data demo_data  # synthetic WAVs + sidecar .txt
python -m sentinel_calibration analyse demo_data         # table, plot, leave-one-group-out
python -m sentinel_calibration analyse <your-folder>     # same, for your own recordings
pytest                                                   # all tests
```

`analyse` writes `results/recordings_table.csv`, `figures/level_distributions.png` and
`results/logo_report.txt`. Recordings must be WAVs named
`DATE_TIME_TYPE_LABEL_FLAGS..._PHONE.wav`, e.g.
`2026-10-04_17-32-56_audio_fan-ON-step3_dist-TBD_has-thump_nord.wav`. Add `roomA` (or
`room-A`) to the label, e.g. `fan-off-3m-roomA`, to make rooms the groups held out one at a time.

## Main result (simulation, alpha = 0.1, 200 seeds, shift = 1, 500 calibration points)

| Method | Day coverage | Night coverage | Night abstention |
|---|---|---|---|
| One pooled threshold | 0.978 | **0.713** | 0.14 |
| Per-condition (Mondrian) | 0.899 | **0.903** | 0.61 |

The target is 1 - alpha = 0.90.

**Interpretation.** In this simulation the classifier is accurate by day and noisier and more
over-confident at night. A single threshold calibrated on the pooled data is dominated by the
easy daytime examples, so it is too permissive for night: it over-covers by day (0.98) and
covers only 0.71 of true night states, well under the promised 0.90. Calibrating each condition
on its own data restores about 0.90 in both. The price is honesty about uncertainty: at night the
per-condition method answers "not sure" for about 61% of questions, because the night classifier
really is much less informative. **Limits:** this is a simulation whose shift was built to
produce exactly this effect, with only two conditions, independent samples and a one-number
classifier output; real recordings can shift in more ways, and the guarantee needs the
calibration data of each condition to be representative of that condition's test data
(exchangeability) and a large-enough calibration set per condition. The result shows the
mechanism works; it does not show SENTINEL's real-world coverage.

## Layout

```
src/sentinel_calibration/
  conformal.py   split conformal, pooled + Mondrian helpers
  simulate.py    seeded day/night simulator (one SimConfig dataclass)
  experiments.py Monte-Carlo runs, sweeps, figures, tables
  features.py    WAV features (level, bump, bands, pulse)
  dataset.py     file-name / label / sidecar parsing
  evaluate.py    feature table, plots, leave-one-group-out
  synth_audio.py synthetic WAV generator (tests and demo)
  __main__.py    command line
tests/           pytest suite;  .github/workflows/tests.yml runs it
```
