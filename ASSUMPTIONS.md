# ASSUMPTIONS

Nobody was available to answer questions, so these choices were made and recorded.

1. Branch: the task named `claude/sweet-einstein-8q3uwh`; all work is on it.
2. Two classes are coded 0 = "off", 1 = "on". Conditions are coded 0 = day, 1 = night.
3. Default miscoverage alpha = 0.1 (target coverage 90%).
4. Simulator: the classifier sees a 1-D raw score `z ~ Normal(+-margin, noise)` and outputs
   `p(on) = sigmoid(scale * z)`. `scale` is fitted for daylight (it equals the true
   log-likelihood ratio there), so at night, where margin shrinks, noise grows and the
   scale is inflated, the same classifier is noisier AND over-confident. One number,
   `shift`, from 0 (no shift) to 1 (full shift) controls how strong this is.
5. "Calibration-set size" in the sweep means the TOTAL calibration size; night makes up
   `night_fraction` (default 30%) of it, so Mondrian night calibration uses fewer points.
6. When a Mondrian cell has too few points for the finite-sample quantile (k > n), the
   quantile is +infinity, so the prediction set is {off, on} (always valid, never useful).
7. Audio: WAV only (scipy reader: 8/16/32-bit int and 32/64-bit float; 24-bit is not
   supported and is reported as an error for that file). Multi-channel audio is averaged to mono.
   dBFS = 20*log10(rms or peak) with samples scaled to [-1, 1]; silence is floored at -120 dBFS.
8. File names: split on "_" ; first two parts are date and time, third is the type, fourth is the
   label, the LAST is the phone, everything in between is flags. Names with fewer than 5 parts
   do not crash; they are marked `parse_ok=False` and whatever can be read is kept.
9. Room tag: a label token `room<name>` or `room-<name>` (e.g. `fan-off-3m-roomA`). No room tag
   means group "all".
10. Leave-one-group-out with a single group falls back to leave-one-recording-out (otherwise
    there would be nothing to hold out). The classifier is a 1-D logistic regression on one
    feature (default `steady_dbfs`), followed by split conformal prediction.
11. Pulse detection: threshold on peak/median ratio of the envelope spectrum = 6.0 (chosen by
    testing on synthetic noise; it is a heuristic, not a calibrated detector).
