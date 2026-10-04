"""Command line: python -m sentinel_calibration {analyse,experiments,make-demo-data,reproduce}"""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="sentinel_calibration", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyse", help="feature table, level plot and leave-one-group-out for a folder")
    a.add_argument("folder")
    a.add_argument("--figures", default="figures")
    a.add_argument("--results", default="results")
    a.add_argument("--feature", default="steady_dbfs")
    a.add_argument("--alpha", type=float, default=0.1)

    e = sub.add_parser("experiments", help="simulation experiments (figures/ and results/)")
    e.add_argument("--figures", default="figures")
    e.add_argument("--results", default="results")
    e.add_argument("--quick", action="store_true", help="fewer seeds, for a fast check")

    d = sub.add_parser("make-demo-data", help="write a synthetic folder of WAV recordings")
    d.add_argument("folder", nargs="?", default="demo_data")

    r = sub.add_parser("reproduce", help="make every figure from scratch")
    r.add_argument("--quick", action="store_true")

    args = p.parse_args(argv)

    if args.cmd == "analyse":
        from .evaluate import analyse_folder
        out = analyse_folder(args.folder, args.figures, args.results, args.feature, args.alpha)
        print(f"{len(out['rows'])} recordings -> {out['table']}")
        print(f"figure: {out['figure']}")
        print(out["text"])
    elif args.cmd == "experiments":
        from .experiments import run_all
        out = run_all(args.figures, args.results, quick=args.quick)
        print("figures:", *out["figures"], sep="\n  ")
    elif args.cmd == "make-demo-data":
        from .synth_audio import make_demo_folder
        print(f"wrote {make_demo_folder(args.folder)} synthetic recordings to {args.folder}")
    elif args.cmd == "reproduce":
        from .evaluate import analyse_folder
        from .experiments import run_all
        from .synth_audio import make_demo_folder
        out = run_all(quick=args.quick)
        print("figures:", *out["figures"], sep="\n  ")
        make_demo_folder("demo_data")
        res = analyse_folder("demo_data")
        print(res["text"])
        print("figure:", res["figure"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
