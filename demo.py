#!/usr/bin/env python3
"""A ninety-second tour of what this lab found.

    python demo.py            # about 90 seconds
    python demo.py --quick    # about 10 seconds, visibly noisier

Runs the three headline results end to end at deliberately reduced fidelity --
fewer shots, one seed, a short training budget. The numbers it prints are
therefore noisier than the ones in the modules' READMEs, which come from 8-seed
sweeps. This is an illustration you can watch happen, not a reproduction; every
section names the module that carries the real measurement.
"""

from __future__ import annotations

import os

# Must precede numpy and qiskit: both read these at import time. See
# qml-noise-robustness/README.md section 5 for why single-threaded is faster
# here rather than slower on two-qubit circuits.
for _threads_var in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS",
):
    os.environ.setdefault(_threads_var, "1")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent
for _module_dir in (ROOT, ROOT / "qml-noise-robustness", ROOT / "qml-adversarial-attacks"):
    sys.path.insert(0, str(_module_dir))

import numpy as np  # noqa: E402
from adversarial.attacks import attack_outcome, fgsm_attack, random_attack  # noqa: E402
from adversarial.gradients import loss_gradient  # noqa: E402
from noise_robustness.mitigation import (  # noqa: E402
    ReadoutMitigator,
    build_zne_classifiers,
    mitigation_variants,
)
from scipy.optimize import minimize  # noqa: E402

from qml_lab.data import load_moons  # noqa: E402
from qml_lab.model import VariationalClassifier, log_loss  # noqa: E402
from qml_lab.noise_models import device_like, readout  # noqa: E402

LAYERS = 3
SEED = 42


def banner(number: int, title: str, module: str) -> None:
    print(f"\n{'=' * 74}\n  {number}. {title}\n     full measurement: {module}\n{'=' * 74}")


def train(x_train, y_train, shots: int, iterations: int, restarts: int) -> np.ndarray:
    """Same protocol as the modules, with the budget turned down."""
    classifier = VariationalClassifier(layers=LAYERS, shots=shots, seed=SEED)

    def objective(weights: np.ndarray) -> float:
        return log_loss(classifier.predict_proba(x_train, weights), y_train)

    best, best_loss = None, np.inf
    for restart in range(restarts):
        start = np.random.default_rng(SEED + restart).uniform(-1, 1, classifier.n_weights)
        result = minimize(objective, start, method="COBYLA",
                          options={"maxiter": iterations, "rhobeg": 0.5, "tol": 1e-4})
        if result.fun < best_loss:
            best_loss, best = float(result.fun), np.asarray(result.x)
    assert best is not None
    return best


def section_accuracy_is_blind(x_test, y_test, weights, shots: int) -> None:
    banner(1, "Accuracy does not see what noise does to the outputs",
           "qml-noise-robustness section 7.2")

    clean = VariationalClassifier(layers=LAYERS, shots=shots, seed=SEED)
    noisy = VariationalClassifier(
        layers=LAYERS, noise_model=device_like().model, shots=shots, seed=SEED
    )
    clean_proba = clean.predict_proba(x_test, weights)
    noisy_proba = noisy.predict_proba(x_test, weights)

    clean_accuracy = float(((clean_proba >= 0.5).astype(int) == y_test).mean())
    noisy_accuracy = float(((noisy_proba >= 0.5).astype(int) == y_test).mean())
    shift = float(np.mean(np.abs(noisy_proba - clean_proba)))

    print(f"\n  test accuracy, noiseless simulator     {clean_accuracy:.3f}")
    print(f"  test accuracy, device-like noise       {noisy_accuracy:.3f}")
    print(f"  -> accuracy changed by                 {noisy_accuracy - clean_accuracy:+.3f}")
    print(f"\n  mean |probability shift|               {shift:.3f}")
    print("\n  The labels barely move. The probabilities behind them do. An")
    print("  acceptance test built on accuracy alone reports nothing wrong.")


def section_mitigation_is_specific(x_test, weights, shots: int) -> None:
    banner(2, "Each mitigation corrects its own mechanism, and only its own",
           "qml-noise-robustness section 10.1")

    reference = VariationalClassifier(layers=LAYERS, shots=shots, seed=SEED).predict_proba(
        x_test, weights
    )

    for label, condition in (("pure readout error", readout(0.1)),
                             ("device-like composite", device_like())):
        classifiers = build_zne_classifiers(LAYERS, condition.model, shots, SEED)
        mitigator = ReadoutMitigator(condition.model, shots=shots * 2, seed=SEED)
        variants = mitigation_variants(classifiers, mitigator, x_test, weights)

        base = float(np.mean(np.abs(variants["none"] - reference)))
        print(f"\n  {label}   (unmitigated shift {base:.4f})")
        for strategy in ("readout", "zne", "readout+zne"):
            shift = float(np.mean(np.abs(variants[strategy] - reference)))
            removed = (base - shift) / base * 100 if base > 0 else 0.0
            print(f"    {strategy:<14} shift {shift:.4f}   removed {removed:+6.1f}%")

    print("\n  Zero-noise extrapolation does nothing to a measurement-time error:")
    print("  folding repeats the circuit, and readout error is applied once")
    print("  regardless. That zero is what separates a real correction from a")
    print("  technique that would merely smooth the outputs.")


def section_attacks_beat_noise(x_test, y_test, weights, shots: int) -> None:
    banner(3, "A gradient attack beats random noise of the same size",
           "qml-adversarial-attacks section 7.1")

    classifier = VariationalClassifier(layers=LAYERS, shots=shots, seed=SEED)

    def predict(x: np.ndarray) -> np.ndarray:
        return (classifier.predict_proba(x, weights) >= 0.5).astype(int)

    def gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return loss_gradient(classifier, x, y, weights, shots=shots)

    clean_prediction = predict(x_test)
    rng = np.random.default_rng(SEED)

    # FGSM's gradient is taken at the clean input and does not depend on the
    # budget, so it is computed once for both rows. On a quantum model a
    # gradient costs 2 * layers * n_features circuit runs, which is most of
    # this section's runtime.
    clean_gradient = gradient(x_test, y_test)

    print(f"\n  {'epsilon':>8}  {'random':>9}  {'FGSM':>9}")
    for epsilon in (0.2, 0.4):
        random_flip = attack_outcome(
            x_test, y_test, random_attack(x_test, epsilon, rng), predict, clean_prediction
        )["flip_rate"]
        adversarial = fgsm_attack(
            x_test, y_test, epsilon, lambda _x, _y, g=clean_gradient: g
        )
        fgsm_flip = attack_outcome(
            x_test, y_test, adversarial, predict, clean_prediction
        )["flip_rate"]
        print(f"  {epsilon:>8.2f}  {random_flip:>9.3f}  {fgsm_flip:>9.3f}")

    print("\n  Same budget, very different effect: the direction matters more")
    print("  than the size. Without the random control a flip rate would not")
    print("  distinguish a fragile model from a gradient too noisy to follow.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="even lower fidelity; used by the test suite")
    args = parser.parse_args()

    shots = 256 if args.quick else 2048
    iterations = 15 if args.quick else 150
    restarts = 1 if args.quick else 2
    n_test = 20 if args.quick else 80

    started = time.perf_counter()
    print(__doc__.strip())

    x_train, x_test, y_train, y_test = load_moons(seed=SEED)
    x_test, y_test = x_test[:n_test], y_test[:n_test]

    print(f"\ntraining a {LAYERS}-layer classifier on {len(x_train)} samples "
          f"({shots} shots, {iterations} iterations, {restarts} restart(s)) ...")
    weights = train(x_train, y_train, shots, iterations, restarts)

    section_accuracy_is_blind(x_test, y_test, weights, shots)
    section_mitigation_is_specific(x_test, weights, shots)
    section_attacks_beat_noise(x_test, y_test, weights, shots)

    print(f"\n{'=' * 74}")
    print(f"  done in {time.perf_counter() - started:.0f}s. Reduced fidelity throughout —")
    print("  see each module's README for the 8-seed measurements with error bars.")
    print(f"{'=' * 74}\n")


if __name__ == "__main__":
    main()
