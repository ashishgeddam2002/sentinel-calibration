# How it works (in plain words)

## The problem

A classifier says "fan on with probability 0.8". Can we trust that 0.8? At night the camera
sees less and the microphone hears more room noise, so the same classifier is often *more
wrong while sounding just as sure*. We want the system to say "not sure" when it should, and we
want a **promise** we can test: *"the true answer is in my answer set at least 90% of the time."*

## Conformal prediction, step by step

Instead of one answer, the system returns a **set** of answers: `{on}`, `{off}`, or
`{off, on}` ("not sure"). (A set can even be empty `{}` when neither answer looks plausible.)

1. **Score the surprise.** For a labelled example, `s = 1 - p(true class)`. If the model gave
   the true answer probability 0.9, then s = 0.1 (not surprised). If it gave 0.2, then s = 0.8.
2. **Calibrate.** Collect scores on `n` labelled examples the model has never trained on.
   Pick the threshold `q` as the `k`-th smallest score, with `k = ceil((n+1)(1-alpha))`.
   (This is the quantile at level `ceil((n+1)(1-alpha))/n`; the "+1" is the small-sample
   correction that makes the promise exact.) If `k > n` there is not enough data and `q = infinity`:
   the system always says "not sure".
3. **Predict.** For a new example keep every class whose score `1 - p(class)` is `<= q`.

**Tiny worked example** (alpha = 0.1, so we want 90%). Nine calibration scores:
`0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.40, 0.60`. Here n = 9, so
k = ceil(10 x 0.9) = 9: the threshold is the 9th (largest) score, **q = 0.60**.

| New photo: p(on) | score if "on" = 1-p(on) | score if "off" = p(on) | set |
|---|---|---|---|
| 0.70 | 0.30 <= 0.60 keep | 0.70 > 0.60 drop | `{on}` |
| 0.50 | 0.50 keep | 0.50 keep | `{off, on}` = "not sure" |
| 0.30 | 0.70 drop | 0.30 keep | `{off}` |

Why 90%? If the new example is "like" the calibration ones, its true-class score is equally
likely to land anywhere among the 10 scores (9 old + 1 new), so it is at or below the 9th
smallest with probability at least 9/10. That is the whole trick, and it needs no assumption
about *how good* the model is. It *does* need the new example to look like the calibration data.

## Why night breaks a single threshold, and what Mondrian does

If the calibration set mixes day and night, but most examples are easy daytime ones, `q` ends up
small. At night the true-class scores are bigger (the model is wrong and sure of itself), so a
small `q` drops the right answer too often: **coverage below 90% at night**. **Mondrian**
(condition-stratified) conformal prediction computes one `q` per condition, using only that
condition's calibration examples. Each condition is then "like its own calibration data", so
each gets its own 90% promise. Cost: each condition needs enough calibration examples (with
night n = 8 and alpha = 0.1, `k = 9 > 8`, so `q = infinity`).

## The modules

- **`conformal.py`** - the three steps above: `nonconformity`, `conformal_quantile`,
  `prediction_sets`, plus `pooled_calibrate` (one q) and `mondrian_calibrate` (a dictionary,
  one q per condition). Classes: 0 = off, 1 = on.
- **`simulate.py`** - a fake classifier for testing the idea. The raw score is
  `z ~ Normal(+-margin, noise)` (plus for on, minus for off) and the output is
  `p(on) = sigmoid(scale * z)`. By day `scale` is exactly right. At night the margin shrinks,
  the noise grows and `scale` grows (over-confidence); `SimConfig.shift` (0 to 1) dials this in.
  Everything is seeded: same seed, same numbers. All parameters live in one dataclass,
  `SimConfig`.
- **`experiments.py`** - for 200 random seeds: draw a mixed calibration set, calibrate pooled and
  per-condition, draw fresh day and night test sets, and record *coverage* (true answer inside
  the set), *set size*, and *abstention* (set size not equal to 1). Sweeps shift strength and
  calibration size; saves the four figures and the tables.
- **`features.py`** - turns a WAV file into numbers. *Steady level*: the 25th percentile of the
  levels of 0.25 s windows after the first 0.5 s (the quiet floor; a short thump will not move
  it). *Bump*: any sample peak above -10 dBFS. *Band levels*: energy in 20-100, 100-300,
  300-1000, 1000-3000 and 3000-8000 Hz (from an FFT). *Pulse*: take the loudness envelope
  (200 values per second), take its spectrum, and look for a clear peak between 8 and 30 Hz -
  a fan or motor that "chops" the sound. dBFS means decibels relative to the loudest possible
  sample (0 dBFS), so real recordings are negative numbers.
- **`dataset.py`** - reads `DATE_TIME_TYPE_LABEL_FLAGS..._PHONE.ext` names and labels like
  `fan-off-3m` or `fan-ON-step3` (ON/OFF in any case), and the sidecar `.txt` (`key=value`
  lines). Anything it does not understand (flags, extra label words) is kept, never an error.
- **`evaluate.py`** - builds a table from a folder, plots ON vs OFF levels, and runs
  *leave-one-group-out*: hold out all recordings of one room, train on the others (a tiny
  logistic regression on one feature, then conformal calibration), and test on the held-out room.
  This imitates "a new room we have never seen". With only one room it holds out one
  recording at a time.
- **`__main__.py`** - the command line (`reproduce`, `experiments`, `analyse`, `make-demo-data`).

## Things to be careful about

- A simulation shows the mechanism, not real-world performance.
- Conformal promises are **averages** over many questions in a condition, not a guarantee for one
  particular answer, and they need exchangeable (similar) calibration and test data.
- **Future idea, not built:** letting the system ask for an extra sensing action up to K
  times before answering. If the *number* of extra looks depends on what the data showed, the
  simple guarantee above no longer applies as-is and needs extra care (for example calibrating
  the whole stopping procedure, or using sequential tests). No proof is claimed here.
