#!/usr/bin/env python3
"""Entry point for the noise-robustness sweep.

    python run_experiment.py                      # 8 seeds, parallel
    python run_experiment.py --seeds 42 43        # fewer seeds
    python run_experiment.py --workers 1          # serial
    python run_experiment.py --quiet --no-plots
"""

from __future__ import annotations

import os

# Set before numpy or qiskit are imported, because both read these at import
# time. For two-qubit circuits Aer's internal threading costs more than it
# saves -- measured 118 ms vs 85 ms per objective evaluation on this workload --
# and single-threaded workers then parallelise cleanly across seeds instead of
# fighting each other for cores. setdefault so an explicit environment wins.
for _threads_var in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS",
):
    os.environ.setdefault(_threads_var, "1")

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

from src.experiment import DEFAULT_SEEDS, run  # noqa: E402
from src.plots import plot_depth_tradeoff, plot_noise_sweeps  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help=f"seeds to sweep (default: {DEFAULT_SEEDS})")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1),
                        help="parallel seed workers (default: 4)")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR, help="output directory")
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    parser.add_argument("--no-plots", action="store_true", help="skip figure generation")
    args = parser.parse_args()

    payload = run(args.output, seeds=args.seeds, workers=args.workers, verbose=not args.quiet)

    if not args.no_plots:
        sweeps = plot_noise_sweeps(payload, args.output / "noise_sweeps.png")
        tradeoff = plot_depth_tradeoff(payload, args.output / "depth_tradeoff.png")
        print(f"wrote {sweeps}\nwrote {tradeoff}")
    print(f"wrote {args.output / 'results.json'}\nwrote {args.output / 'results.csv'}")


if __name__ == "__main__":
    main()
