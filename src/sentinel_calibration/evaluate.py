"""Build a feature table from a folder of recordings, plot ON vs OFF levels, and run
leave-one-group-out evaluation (1-D logistic regression + split conformal prediction)."""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import conformal as cf
from .dataset import Recording, load_folder
from .features import extract_features


def build_table(folder: str) -> List[Dict[str, object]]:
    """One row per recording: metadata + features. A file that cannot be read gets an `error`."""
    rows = []
    for rec in load_folder(folder):
        row: Dict[str, object] = dict(
            file=os.path.basename(rec.path), state=rec.label.state, y=rec.label.y,
            group=rec.group, phone=rec.phone, distance_m=rec.label.distance_m,
            step=rec.label.step, parse_ok=rec.parse_ok, flags="|".join(rec.flags),
            extras="|".join(rec.label.extras), error="")
        try:
            row.update(extract_features(rec.path))
        except Exception as e:  # unreadable / unsupported WAV: keep the row, record why
            row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
    return rows


def save_table(rows: List[Dict[str, object]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cols: List[str] = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


def plot_levels(rows, path: str, feature: str = "steady_dbfs") -> bool:
    """Histogram of a level feature for ON vs OFF. Returns False if there is nothing to plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vals = {s: np.array([r[feature] for r in rows if r.get("state") == s and _finite(r.get(feature))])
            for s in ("on", "off")}
    if all(len(v) == 0 for v in vals.values()):
        return False
    allv = np.concatenate([v for v in vals.values() if len(v)])
    bins = np.linspace(allv.min() - 1, allv.max() + 1, 20)
    fig, ax = plt.subplots(figsize=(6, 4))
    for s, c in (("off", "#1f6fb2"), ("on", "#c0392b")):
        if len(vals[s]):
            ax.hist(vals[s], bins=bins, alpha=0.6, color=c, label=f"fan {s.upper()} (n={len(vals[s])})")
    ax.set_xlabel(f"{feature} (dBFS)")
    ax.set_ylabel("Number of recordings")
    ax.set_title("Level distribution: fan ON vs OFF")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def _finite(v) -> bool:
    return isinstance(v, (int, float, np.floating)) and np.isfinite(v)


def fit_logistic_1d(x: np.ndarray, y: np.ndarray, ridge: float = 1e-2, iters: int = 50):
    """Newton's method for p(on) = sigmoid(a*z + b), z = standardised x. Returns a predict fn."""
    mu, sd = float(x.mean()), float(x.std()) or 1.0
    z = (x - mu) / sd
    a = b = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(a * z + b)))
        w = p * (1 - p) + 1e-9
        g = np.array([np.sum((p - y) * z) + ridge * a, np.sum(p - y)])
        H = np.array([[np.sum(w * z * z) + ridge, np.sum(w * z)], [np.sum(w * z), np.sum(w)]])
        step = np.linalg.solve(H, g)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-8:
            break
    return lambda xn: 1 / (1 + np.exp(-(a * (np.asarray(xn, float) - mu) / sd + b)))


def leave_one_group_out(rows, feature: str = "steady_dbfs", alpha: float = 0.1, seed: int = 0) -> Dict[str, object]:
    """Hold out each group in turn; train on the others (half fit, half conformal calibration).

    With a single group, every recording is its own group (leave-one-recording-out).
    """
    usable = [r for r in rows if r.get("y") in (0, 1) and _finite(r.get(feature))]
    x = np.array([r[feature] for r in usable], float)
    y = np.array([r["y"] for r in usable], int)
    groups = np.array([r["group"] for r in usable])
    mode = "leave-one-group-out"
    if len(set(groups)) < 2:
        groups = np.array([str(i) for i in range(len(usable))])
        mode = "leave-one-recording-out (only one group)"
    rng = np.random.default_rng(seed)
    folds, skipped = [], []
    for g in sorted(set(groups)):
        te, tr = groups == g, np.where(groups != g)[0]
        tr = rng.permutation(tr)
        fit, cal = tr[: len(tr) // 2], tr[len(tr) // 2:]
        if len(set(y[fit])) < 2 or len(cal) == 0:
            skipped.append(g)
            continue
        predict = fit_logistic_1d(x[fit], y[fit])
        q = cf.pooled_calibrate(predict(x[cal]), y[cal], alpha)
        p = predict(x[te])
        sets = cf.prediction_sets(p, q)
        size = cf.set_sizes(sets)
        folds.append(dict(group=g, n=int(te.sum()), accuracy=float(((p > 0.5).astype(int) == y[te]).mean()),
                          coverage=float(cf.covered(sets, y[te]).mean()),
                          set_size=float(size.mean()), abstain=float((size != 1).mean())))
    total = sum(f["n"] for f in folds)
    agg = {k: (sum(f[k] * f["n"] for f in folds) / total if total else float("nan"))
           for k in ("accuracy", "coverage", "set_size", "abstain")}
    return dict(mode=mode, feature=feature, alpha=alpha, n_usable=len(usable),
                folds=folds, skipped=skipped, overall=agg)


def format_logo(res: Dict[str, object]) -> str:
    lines = [f"{res['mode']} on '{res['feature']}', alpha={res['alpha']}, usable recordings={res['n_usable']}"]
    if len(res["folds"]) <= 12:
        for f in res["folds"]:
            lines.append(f"  held-out {f['group']:>10}: n={f['n']:3d} acc={f['accuracy']:.2f} "
                         f"cov={f['coverage']:.2f} size={f['set_size']:.2f} abstain={f['abstain']:.2f}")
    o = res["overall"]
    lines.append(f"  overall: acc={o['accuracy']:.3f} coverage={o['coverage']:.3f} "
                 f"set size={o['set_size']:.3f} abstain={o['abstain']:.3f}")
    if res["skipped"]:
        lines.append(f"  skipped folds (not enough training data): {res['skipped']}")
    return "\n".join(lines)


def analyse_folder(folder: str, fig_dir: str = "figures", res_dir: str = "results",
                   feature: str = "steady_dbfs", alpha: float = 0.1, seed: int = 0) -> Dict[str, object]:
    rows = build_table(folder)
    table_path = os.path.join(res_dir, "recordings_table.csv")
    save_table(rows, table_path)
    fig_path = os.path.join(fig_dir, "level_distributions.png")
    plotted = plot_levels(rows, fig_path, feature)
    logo = leave_one_group_out(rows, feature, alpha, seed)
    text = format_logo(logo)
    with open(os.path.join(res_dir, "logo_report.txt"), "w") as f:
        f.write(text + "\n")
    return dict(rows=rows, table=table_path, figure=fig_path if plotted else None, logo=logo, text=text)
