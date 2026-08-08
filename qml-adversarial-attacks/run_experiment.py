#!/usr/bin/env python3
"""Entry point for the adversarial attack sweep.

    python run_experiment.py                  # 8 seeds, parallel
    python run_experiment.py --seeds 42 43    # fewer seeds
    python run_experiment.py --workers 1      # serial
"""

from __future__ import annotations

import os

# Set before numpy or qiskit are imported, because both read these at import
# time. On two-qubit circuits Aer's internal threading costs more than it saves
# (measured in qml-noise-robustness: 118 ms vs 85 ms per evaluation), and
# single-threaded workers parallelise cleanly across seeds instead.
for _threads_var in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS",
):
    os.environ.setdefault(_threads_var, "1")

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

# The shared ``qml_lab`` package lives at the repository root, which Python does
# not put on the path when this script is run from inside its own directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adversarial.experiment import DEFAULT_SEEDS, run  # noqa: E402
from adversarial.plots import plot_flip_rates, plot_model_comparison  # noqa: E402

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
        for figure in (
            plot_flip_rates(payload, args.output / "flip_rates.png"),
            plot_model_comparison(payload, args.output / "model_comparison.png"),
        ):
            print(f"wrote {figure}")
    for name in ("results.json", "results.csv"):
        print(f"wrote {args.output / name}")


if __name__ == "__main__":
    main()
