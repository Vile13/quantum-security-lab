"""Training and the depth x noise sweep.

Protocol: train once per ansatz depth on the *noiseless* simulator, then freeze
the weights and evaluate them under every noise condition.

Training noiseless and evaluating noisy is a deliberate choice. It isolates the
question being asked -- how much accuracy is lost when a model meets a noisy
device -- from the separate question of whether training can compensate for
noise it can observe. It also matches the realistic deployment case: a model
shipped to a device whose error profile differs from the one it was tuned on.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from .data import load_moons
from .model import VariationalClassifier, log_loss
from .noise_models import NoiseCondition, all_conditions

# Shots are lower during training than during evaluation: the optimiser
# tolerates a noisy objective, but the reported accuracies should not be
# dominated by sampling error.
TRAIN_SHOTS = 512
EVAL_SHOTS = 4096
# The re-uploading model carries 6 weights per layer, so the optimiser needs a
# budget that scales with the layer count rather than the 120 evaluations that
# sufficed for the 6-parameter encode-once model.
MAX_ITERATIONS = 600
LAYER_COUNTS = [2, 3, 4, 5]
# COBYLA on a shot-noisy objective converges to whatever basin it starts in,
# and a single unlucky start is indistinguishable in the results table from a
# genuine capacity limit of that layer count. Restarting and keeping the best
# run makes the depth axis mean what the plot claims it means.
N_RESTARTS = 3


@dataclass
class TrainingResult:
    layers: int
    weights: list[float]
    final_loss: float
    iterations: int
    circuit_depth: int
    two_qubit_gates: int
    train_accuracy: float
    seconds: float
    restart_losses: list[float]  # every restart, so run-to-run spread stays visible


@dataclass
class EvaluationRow:
    layers: int
    label: str
    mechanism: str
    strength: float
    test_accuracy: float
    accuracy_drop: float
    mean_abs_proba_shift: float


def train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    layers: int,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[np.ndarray, TrainingResult]:
    """Fit the weights on the noiseless simulator using COBYLA.

    COBYLA is gradient-free, which suits a shot-noise objective: finite
    differences on a stochastic function would mostly measure sampling error.
    """
    classifier = VariationalClassifier(
        layers=layers, noise_model=None, shots=TRAIN_SHOTS, seed=seed
    )

    def objective(weights: np.ndarray) -> float:
        return log_loss(classifier.predict_proba(x_train, weights), y_train)

    started = time.perf_counter()
    best_weights: np.ndarray | None = None
    best_loss = np.inf
    restart_losses: list[float] = []
    total_evaluations = 0

    for restart in range(N_RESTARTS):
        # Initialised in [-1, 1] rather than [0, 2*pi]: half of each layer's
        # weights multiply an input, and a large initial scale starts the model
        # in a high-frequency regime the optimiser has to climb back out of.
        rng = np.random.default_rng(seed + restart)
        initial = rng.uniform(-1.0, 1.0, classifier.n_weights)
        result = minimize(
            objective, initial, method="COBYLA",
            options={"maxiter": MAX_ITERATIONS, "rhobeg": 0.5, "tol": 1e-4},
        )
        restart_losses.append(float(result.fun))
        total_evaluations += int(result.nfev)
        if result.fun < best_loss:
            best_loss, best_weights = float(result.fun), np.asarray(result.x)

    elapsed = time.perf_counter() - started
    assert best_weights is not None  # N_RESTARTS >= 1

    summary = TrainingResult(
        layers=layers,
        weights=best_weights.tolist(),
        final_loss=best_loss,
        iterations=total_evaluations,
        circuit_depth=classifier.depth,
        two_qubit_gates=classifier.two_qubit_gate_count,
        train_accuracy=classifier.accuracy(x_train, y_train, best_weights, shots=EVAL_SHOTS),
        seconds=elapsed,
        restart_losses=restart_losses,
    )
    if verbose:
        spread = " / ".join(f"{loss:.4f}" for loss in restart_losses)
        print(
            f"  layers={layers}  best loss {best_loss:.4f} (restarts: {spread})  "
            f"train acc {summary.train_accuracy:.3f}  "
            f"({total_evaluations} evals, {elapsed:.1f}s, "
            f"circuit depth {summary.circuit_depth}, {summary.two_qubit_gates} 2q gates)"
        )
    return best_weights, summary


def evaluate_under_noise(
    x_test: np.ndarray,
    y_test: np.ndarray,
    weights: np.ndarray,
    layers: int,
    conditions: list[NoiseCondition],
    seed: int = 42,
    verbose: bool = True,
) -> list[EvaluationRow]:
    """Score frozen weights under each noise condition."""
    baseline_classifier = VariationalClassifier(
        layers=layers, noise_model=None, shots=EVAL_SHOTS, seed=seed
    )
    baseline_proba = baseline_classifier.predict_proba(x_test, weights)
    baseline_accuracy = float(((baseline_proba >= 0.5).astype(int) == y_test).mean())

    rows: list[EvaluationRow] = []
    for condition in conditions:
        classifier = VariationalClassifier(
            layers=layers, noise_model=condition.model, shots=EVAL_SHOTS, seed=seed
        )
        proba = classifier.predict_proba(x_test, weights)
        accuracy = float(((proba >= 0.5).astype(int) == y_test).mean())
        rows.append(
            EvaluationRow(
                layers=layers,
                label=condition.label,
                mechanism=condition.mechanism,
                strength=condition.strength,
                test_accuracy=accuracy,
                accuracy_drop=baseline_accuracy - accuracy,
                # Accuracy is a thresholded metric and hides erosion of
                # confidence. This tracks how far the probabilities moved even
                # when the argmax did not change -- an early warning signal
                # that accuracy alone would miss.
                mean_abs_proba_shift=float(np.mean(np.abs(proba - baseline_proba))),
            )
        )
        if verbose:
            print(f"    {condition.label:<28} acc {accuracy:.3f}  "
                  f"(drop {rows[-1].accuracy_drop:+.3f}, |dp| {rows[-1].mean_abs_proba_shift:.3f})")
    return rows


def run(output_dir: Path, seed: int = 42, verbose: bool = True) -> dict:
    """Run the full depth x noise sweep and persist results as JSON and CSV."""
    x_train, x_test, y_train, y_test = load_moons(seed=seed)
    conditions = all_conditions()
    if verbose:
        print(f"dataset: {len(x_train)} train / {len(x_test)} test samples")
        print(f"sweep:   {len(LAYER_COUNTS)} layer counts x {len(conditions)} noise conditions\n")

    trainings: list[TrainingResult] = []
    evaluations: list[EvaluationRow] = []
    for layers in LAYER_COUNTS:
        if verbose:
            print(f"training layers={layers}")
        weights, summary = train(x_train, y_train, layers, seed=seed, verbose=verbose)
        trainings.append(summary)
        if verbose:
            print(f"  evaluating {len(conditions)} noise conditions")
        evaluations += evaluate_under_noise(
            x_test, y_test, weights, layers, conditions, seed=seed, verbose=verbose
        )
        if verbose:
            print()

    payload = {
        "config": {
            "seed": seed,
            "layer_counts": LAYER_COUNTS,
            "train_shots": TRAIN_SHOTS,
            "eval_shots": EVAL_SHOTS,
            "max_iterations": MAX_ITERATIONS,
            "n_restarts": N_RESTARTS,
            "n_train": len(x_train),
            "n_test": len(x_test),
        },
        "training": [asdict(t) for t in trainings],
        "evaluation": [asdict(e) for e in evaluations],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2))

    csv_lines = ["layers,label,mechanism,strength,test_accuracy,accuracy_drop,mean_abs_proba_shift"]
    csv_lines += [
        f"{e.layers},\"{e.label}\",{e.mechanism},{e.strength:g},"
        f"{e.test_accuracy:.4f},{e.accuracy_drop:.4f},{e.mean_abs_proba_shift:.4f}"
        for e in evaluations
    ]
    (output_dir / "results.csv").write_text("\n".join(csv_lines) + "\n")

    return payload
