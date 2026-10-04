# REPORT

Status: **Definition of Done met.** Stretch goal not attempted.

## Checklist
- [x] Scaffold (pyproject, package, .gitignore), ASSUMPTIONS.md
- [x] A. `conformal.py` - split conformal, finite-sample quantile, pooled + Mondrian helpers
- [x] B. `simulate.py` - seeded day/night simulator, one `SimConfig` dataclass
- [x] C. `experiments.py` - 200 seeds, sweeps over shift and calibration size, figures/ and results/
- [x] D. `features.py` - steady level, bump, 5 band levels, 8-30 Hz pulse search
- [x] E. `dataset.py` - file-name, label and sidecar parsing; unknown flags kept
- [x] F. `evaluate.py` + CLI (`analyse`, `experiments`, `make-demo-data`, `reproduce`)
- [x] G. pytest suite (35 tests) + `.github/workflows/tests.yml`
- [x] H. README.md, HOW_IT_WORKS.md (with worked example)
- [x] Final test run recorded, PR opened
- [ ] Stretch: multi-step stopping - NOT done (kept scope small; see Next steps)

## What is done
Everything A-H. `python -m sentinel_calibration reproduce` regenerates all five figures and all
tables from scratch in about 5 seconds. Only synthetic data is used; no network, accounts or secrets.

**Main result** (alpha = 0.1, 200 seeds, shift = 1, 500 calibration points, 2000 test points per condition):

| Method | Day coverage | Night coverage | Night abstention |
|---|---|---|---|
| One pooled threshold | 0.978 | 0.713 | 0.14 |
| Per-condition (Mondrian) | 0.899 | 0.903 | 0.61 |

Pooled coverage at night is well below 1-alpha = 0.90; per-condition is about 0.90 in both
conditions. Night coverage vs shift: pooled falls 0.90 -> 0.84 -> 0.79 -> 0.75 -> 0.71 for shift
0 -> 1, Mondrian stays 0.903. With small calibration sets Mondrian is conservative (night
coverage 0.99 at 20 total points, because the night cell has ~6 points and the quantile is
infinite) and reaches about 0.90 from roughly 200 points.

**Interpretation (limits of a simulation).** The simulated classifier is accurate by day and
noisier and over-confident at night, so a threshold fixed mostly on easy daytime scores is too
permissive at night; calibrating per condition restores the promise at the cost of abstaining on
about 61% of night questions. The shift was built to cause exactly this, there are only two
conditions, samples are independent, and the classifier output is one number, so this
demonstrates the mechanism, not SENTINEL's real-world coverage. The guarantee also needs each
condition's calibration data to be representative of its test data and large enough.

## What is not done and why
- Stretch (multi-step stopping with up to K extra sensing actions): not attempted; the core
  Definition of Done came first and the brief said do less. HOW_IT_WORKS.md notes that
  coverage guarantees under adaptive stopping need extra care; no proof is claimed.
- Real recordings were not used (none provided; rules forbid real data). The audio pipeline is
  tested on generated WAVs only; thresholds such as the pulse ratio (6.0) are heuristics.
- WAV only; 24-bit WAV is unsupported by the reader and shows up as an `error` in the table.

## How to run
```bash
pip install -e ".[test]"
python -m sentinel_calibration reproduce      # all figures + tables
pytest                                        # tests
python -m sentinel_calibration analyse <folder-of-wavs>
```

## Test results (exact)
`pytest -v` -> **35 passed in 2.26s** (Python 3.11.15, numpy/scipy/matplotlib current at install).
Test files: test_conformal (8), test_simulate (6), test_dataset (7), test_features (10), test_evaluate (4).
`python -m sentinel_calibration reproduce --quick` also runs OK (it is a CI smoke test).
The GitHub Actions workflow (Python 3.10 and 3.12) had not yet been observed running when this
was written; check the PR's checks.

## Assumptions
See ASSUMPTIONS.md (11 items): e.g. alpha = 0.1; night = 30% of the calibration mix; empty Mondrian
cell quantile = infinity; room tag = `roomX` token in the label; one group falls back to
leave-one-recording-out.

## Risks
- Simulated results depend on the simulator's parameters (`SimConfig`); other shifts may differ.
- Mondrian needs enough calibration data per condition (>= ~9 at alpha 0.1 just to be non-trivial).
- Pulse detection can still miss pulses under heavy noise or false-alarm on strongly modulated noise.
- Mondrian night coverage is identical across shift values in fig 2: with common random numbers,
  conformal coverage depends only on score ranks, which do not change when night is
  rescaled. This is expected (distribution-free guarantee), not a bug.

## Next steps
1. Collect real labelled day/night recordings and repeat with the same `analyse` command.
2. Stretch goal: multi-step stopping, calibrated as a whole procedure.
3. Try finer condition strata (phone, distance) and check cells stay large enough.
