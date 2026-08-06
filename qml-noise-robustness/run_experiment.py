#!/usr/bin/env python3
"""Entry point for the noise-robustness sweep.

    python run_experiment.py                 # full sweep, writes to results/
    python run_experiment.py --seed 7        # different seed
    python run_experiment.py --quiet         # no per-condition output
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.experiment import run
from src.plots import plot_depth_tradeoff, plot_noise_sweeps

RESULTS_DIR = Path(__file__).parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="seed for data, init and sampling")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR, help="output directory")
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    parser.add_argument("--no-plots", action="store_true", help="skip figure generation")
    args = parser.parse_args()

    payload = run(args.output, seed=args.seed, verbose=not args.quiet)

    if not args.no_plots:
        sweeps = plot_noise_sweeps(payload, args.output / "noise_sweeps.png")
        tradeoff = plot_depth_tradeoff(payload, args.output / "depth_tradeoff.png")
        print(f"wrote {sweeps}\nwrote {tradeoff}")
    print(f"wrote {args.output / 'results.json'}\nwrote {args.output / 'results.csv'}")


if __name__ == "__main__":
    main()
