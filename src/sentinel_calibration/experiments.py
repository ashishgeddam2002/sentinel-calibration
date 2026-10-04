"""Monte-Carlo experiments: pooled vs per-condition (Mondrian) conformal calibration.

For every random seed we draw a calibration set (mixed day/night), calibrate both ways,
then draw a fresh test set for each condition and measure
  * coverage        - how often the true state is inside the prediction set,
  * set size        - average number of answers kept (1 = decisive, 2 = "not sure"),
  * abstention rate - share of questions where the set size is not 1.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Sequence

import numpy as np

from . import conformal as cf
from .simulate import CONDITIONS, DAY, NIGHT, SimConfig, sample_condition, sample_mixed

METHODS = ("pooled", "mondrian")
FIELDS = ["method", "condition", "n_cal", "shift", "alpha", "n_seeds",
          "coverage", "coverage_sd", "set_size", "abstain", "abstain_sd", "empty_rate"]


def run_once(cfg: SimConfig, n_cal: int, alpha: float, n_test: int, seed: int) -> List[dict]:
    """One seed -> one row per (method, condition)."""
    rng = cfg.rng(seed)
    cal = sample_mixed(cfg, n_cal, rng)
    q_pooled = cf.pooled_calibrate(cal.p_on, cal.y, alpha)
    q_mond = cf.mondrian_calibrate(cal.p_on, cal.y, cal.cond, alpha)
    rows = []
    for cond in (DAY, NIGHT):
        test = sample_condition(cfg, n_test, cond, rng)
        for method in METHODS:
            sets = (cf.predict_pooled(test.p_on, q_pooled) if method == "pooled"
                    else cf.predict_mondrian(test.p_on, test.cond, q_mond))
            size = cf.set_sizes(sets)
            rows.append(dict(method=method, condition=CONDITIONS[cond],
                             coverage=cf.covered(sets, test.y).mean(),
                             set_size=size.mean(), abstain=(size != 1).mean(),
                             empty_rate=(size == 0).mean()))
    return rows


def run_setting(cfg: SimConfig, n_cal: int, alpha: float, n_seeds: int, n_test: int) -> List[dict]:
    """Average run_once over seeds 0..n_seeds-1 (the seed is also mixed into cfg.seed)."""
    per_seed = [run_once(cfg, n_cal, alpha, n_test, s) for s in range(n_seeds)]
    out = []
    for i, first in enumerate(per_seed[0]):
        col = lambda k: np.array([ps[i][k] for ps in per_seed])
        out.append(dict(method=first["method"], condition=first["condition"],
                        n_cal=n_cal, shift=cfg.shift, alpha=alpha, n_seeds=n_seeds,
                        coverage=col("coverage").mean(), coverage_sd=col("coverage").std(ddof=1),
                        set_size=col("set_size").mean(),
                        abstain=col("abstain").mean(), abstain_sd=col("abstain").std(ddof=1),
                        empty_rate=col("empty_rate").mean()))
    return out


def sweep(cfg: SimConfig, shifts: Sequence[float], n_cals: Sequence[int], alpha: float,
          n_seeds: int, n_test: int) -> List[dict]:
    rows = []
    for shift in shifts:
        for n_cal in n_cals:
            rows += run_setting(cfg.with_(shift=shift), n_cal, alpha, n_seeds, n_test)
    return rows


def save_csv(rows: List[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k]) for k in FIELDS})


def save_markdown(rows: List[dict], path: str) -> None:
    cols = ["method", "condition", "n_cal", "shift", "coverage", "coverage_sd", "set_size", "abstain"]
    with open(path, "w") as f:
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in rows:
            f.write("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c])
                                       for c in cols) + " |\n")


def _select(rows, **kw):
    return [r for r in rows if all(np.isclose(r[k], v) if isinstance(v, float) else r[k] == v
                                   for k, v in kw.items())]


def make_figures(main_rows, shift_rows, ncal_rows, alpha, fig_dir, main_shift, main_ncal, ncal_shift):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(fig_dir, exist_ok=True)
    colors = {"pooled": "#c0392b", "mondrian": "#1f6fb2"}
    labels = {"pooled": "One pooled threshold", "mondrian": "Per-condition (Mondrian)"}
    paths = []

    # Figure 1 (main): coverage per condition
    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.36
    for j, m in enumerate(METHODS):
        vals = [_select(main_rows, method=m, condition=c)[0] for c in CONDITIONS]
        x = np.arange(2) + (j - 0.5) * width
        ax.bar(x, [v["coverage"] for v in vals], width, yerr=[v["coverage_sd"] for v in vals],
               capsize=3, color=colors[m], label=labels[m])
        for xi, v in zip(x, vals):
            ax.text(xi, v["coverage"] + v["coverage_sd"] + 0.008, f"{v['coverage']:.2f}", ha="center", fontsize=9)
    ax.axhline(1 - alpha, color="black", ls="--", lw=1, label=f"Target 1-alpha = {1 - alpha:.2f}")
    ax.set_xticks(range(2), CONDITIONS)
    ax.set_ylim(0.5, 1.2)
    ax.set_ylabel("Empirical coverage (mean over seeds, bar = 1 sd)")
    ax.set_title(f"Coverage by condition (shift={main_shift}, n_cal={main_ncal})")
    ax.legend(loc="upper center", fontsize=8, ncol=1)
    fig.tight_layout()
    paths.append(os.path.join(fig_dir, "fig1_coverage_by_condition.png"))
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    # Figure 2: night coverage vs shift strength
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in METHODS:
        for cond, ls in (("night", "-"), ("day", ":")):
            r = sorted(_select(shift_rows, method=m, condition=cond), key=lambda r: r["shift"])
            ax.plot([x["shift"] for x in r], [x["coverage"] for x in r], ls, marker="o",
                    color=colors[m], label=f"{labels[m]} - {cond}")
    ax.axhline(1 - alpha, color="black", ls="--", lw=1)
    ax.set_xlabel("Shift strength (0 = none, 1 = full)")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(f"Coverage vs shift strength (n_cal={main_ncal})")
    ax.legend(fontsize=7)
    fig.tight_layout()
    paths.append(os.path.join(fig_dir, "fig2_coverage_vs_shift.png"))
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    # Figure 3: night coverage vs calibration-set size
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in METHODS:
        r = sorted(_select(ncal_rows, method=m, condition="night"), key=lambda r: r["n_cal"])
        n = np.array([x["n_cal"] for x in r])
        c = np.array([x["coverage"] for x in r])
        s = np.array([x["coverage_sd"] for x in r])
        ax.plot(n, c, marker="o", color=colors[m], label=labels[m])
        ax.fill_between(n, c - s, c + s, color=colors[m], alpha=0.15)
    ax.axhline(1 - alpha, color="black", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("Total calibration-set size (night share = "
                  f"{SimConfig().night_fraction:.0%})")
    ax.set_ylabel("Night coverage")
    ax.set_title(f"Night coverage vs calibration size (shift={ncal_shift})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    paths.append(os.path.join(fig_dir, "fig3_night_coverage_vs_ncal.png"))
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    # Figure 4: set size and abstention vs shift, night only
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for m in METHODS:
        r = sorted(_select(shift_rows, method=m, condition="night"), key=lambda r: r["shift"])
        sh = [x["shift"] for x in r]
        axes[0].plot(sh, [x["set_size"] for x in r], marker="o", color=colors[m], label=labels[m])
        axes[1].plot(sh, [x["abstain"] for x in r], marker="o", color=colors[m], label=labels[m])
    axes[0].set_ylabel("Average set size (night)")
    axes[1].set_ylabel("Abstention rate (night)")
    for a in axes:
        a.set_xlabel("Shift strength")
    axes[0].legend(fontsize=8)
    fig.suptitle("The price of validity: more 'not sure' answers at night")
    fig.tight_layout()
    paths.append(os.path.join(fig_dir, "fig4_setsize_abstention_vs_shift.png"))
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)
    return paths


def run_all(fig_dir="figures", res_dir="results", alpha=0.1, n_seeds=200, n_test=2000,
            quick=False, cfg: SimConfig = None) -> Dict[str, object]:
    """Run every experiment, write tables to res_dir and figures to fig_dir."""
    cfg = cfg or SimConfig()
    if quick:
        n_seeds, n_test = 20, 500
    main_ncal, main_shift = 500, 1.0
    shifts = [0.0, 0.25, 0.5, 0.75, 1.0]
    n_cals = [20, 50, 100, 200, 500, 1000]
    main_rows = run_setting(cfg.with_(shift=main_shift), main_ncal, alpha, n_seeds, n_test)
    shift_rows = sweep(cfg, shifts, [main_ncal], alpha, n_seeds, n_test)
    ncal_rows = sweep(cfg, [main_shift], n_cals, alpha, n_seeds, n_test)
    save_csv(main_rows, os.path.join(res_dir, "main_table.csv"))
    save_csv(shift_rows, os.path.join(res_dir, "sweep_shift.csv"))
    save_csv(ncal_rows, os.path.join(res_dir, "sweep_ncal.csv"))
    save_markdown(main_rows, os.path.join(res_dir, "main_table.md"))
    save_markdown(shift_rows + ncal_rows, os.path.join(res_dir, "sweeps.md"))
    figs = make_figures(main_rows, shift_rows, ncal_rows, alpha, fig_dir, main_shift, main_ncal, main_shift)
    return dict(main=main_rows, shift=shift_rows, ncal=ncal_rows, figures=figs)


if __name__ == "__main__":
    out = run_all()
    for r in out["main"]:
        print(r["method"], r["condition"], round(r["coverage"], 3), round(r["set_size"], 3))
