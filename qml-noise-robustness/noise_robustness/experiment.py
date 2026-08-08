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
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from qml_lab.data import load_moons
from qml_lab.model import VariationalClassifier, log_loss
from qml_lab.noise_models import (
    NoiseCondition,
    all_conditions,
    depolarizing,
    device_like,
    readout,
    thermal_relaxation,
)

from .mitigation import (
    STRATEGIES,
    ZNE_SCALES,
    ReadoutMitigator,
    build_zne_classifiers,
    mitigation_variants,
)

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
# Eight seeds is a compromise: a sample standard deviation from three or four
# runs is itself so noisy that the error bar misleads more than it informs,
# while every added seed costs a full sweep.
DEFAULT_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49]

# Calibration is measured once per condition and reused for every sample, so it
# can afford more shots than an individual evaluation. A noisy calibration
# matrix would be inverted into the corrected result of every sample at once.
CALIBRATION_SHOTS = 16384


def _mitigation_conditions() -> list[NoiseCondition]:
    """Conditions the mitigation comparison runs on.

    One realistic composite, plus the strongest setting of one mechanism per
    type: a measurement-time error that readout mitigation should fix and ZNE
    should not touch, and two gate-accumulated errors where the reverse holds.
    Choosing conditions where each mitigation is expected to fail is what makes
    the comparison informative rather than merely favourable.
    """
    return [device_like(), readout(0.1), depolarizing(0.02), thermal_relaxation(25.0)]


MITIGATION_CONDITIONS = _mitigation_conditions()


@dataclass
class TrainingResult:
    seed: int
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
class MitigationRow:
    seed: int
    layers: int
    label: str
    mechanism: str
    strategy: str
    test_accuracy: float
    accuracy_drop: float
    mean_abs_proba_shift: float
    # Fraction of the unmitigated shift removed. Positive is an improvement,
    # negative means the mitigation made the outputs worse than doing nothing.
    shift_reduction: float


@dataclass
class EvaluationRow:
    seed: int
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
        seed=seed,
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
                seed=seed,
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


def evaluate_mitigations(
    x_test: np.ndarray,
    y_test: np.ndarray,
    weights: np.ndarray,
    layers: int,
    seed: int = 42,
) -> list[MitigationRow]:
    """Compare mitigation strategies on a focused subset of noise conditions.

    The subset is deliberate rather than a cost compromise: it pairs the
    realistic composite model with the strongest setting of one mechanism per
    *type*, so each mitigation can be checked against a condition it should fix
    and a condition it should not touch. A mitigation that improves everything
    equally would be smoothing the numbers, not correcting a mechanism.
    """
    baseline = VariationalClassifier(layers=layers, noise_model=None, shots=EVAL_SHOTS, seed=seed)
    ideal_proba = baseline.predict_proba(x_test, weights)
    ideal_accuracy = float(((ideal_proba >= 0.5).astype(int) == y_test).mean())

    rows: list[MitigationRow] = []
    for condition in MITIGATION_CONDITIONS:
        classifiers = build_zne_classifiers(
            layers, condition.model, EVAL_SHOTS, seed, ZNE_SCALES
        )
        mitigator = ReadoutMitigator(condition.model, shots=CALIBRATION_SHOTS, seed=seed)
        variants = mitigation_variants(classifiers, mitigator, x_test, weights, ZNE_SCALES)

        unmitigated_shift = float(np.mean(np.abs(variants["none"] - ideal_proba)))
        for strategy in STRATEGIES:
            proba = variants[strategy]
            shift = float(np.mean(np.abs(proba - ideal_proba)))
            rows.append(
                MitigationRow(
                    seed=seed,
                    layers=layers,
                    label=condition.label,
                    mechanism=condition.mechanism,
                    strategy=strategy,
                    test_accuracy=float(((proba >= 0.5).astype(int) == y_test).mean()),
                    accuracy_drop=ideal_accuracy
                    - float(((proba >= 0.5).astype(int) == y_test).mean()),
                    mean_abs_proba_shift=shift,
                    shift_reduction=(
                        (unmitigated_shift - shift) / unmitigated_shift
                        if unmitigated_shift > 0
                        else 0.0
                    ),
                )
            )
    return rows


def run_single_seed(
    seed: int, verbose: bool = True
) -> tuple[list[TrainingResult], list[EvaluationRow], list[MitigationRow]]:
    """One complete depth x noise sweep at one seed.

    The seed drives the dataset and its split, the weight initialisation of
    every restart, and the sampler's shot noise. Varying all three together is
    deliberate: the spread it produces is the spread a reader would see on
    re-running the experiment, which is what an error bar should mean.
    """
    x_train, x_test, y_train, y_test = load_moons(seed=seed)
    conditions = all_conditions()

    trainings: list[TrainingResult] = []
    evaluations: list[EvaluationRow] = []
    mitigations: list[MitigationRow] = []
    for layers in LAYER_COUNTS:
        if verbose:
            print(f"[seed {seed}] training layers={layers}")
        weights, summary = train(x_train, y_train, layers, seed=seed, verbose=verbose)
        trainings.append(summary)
        evaluations += evaluate_under_noise(
            x_test, y_test, weights, layers, conditions, seed=seed, verbose=False
        )
        mitigations += evaluate_mitigations(x_test, y_test, weights, layers, seed=seed)
    if verbose:
        print(f"[seed {seed}] done")
    return trainings, evaluations, mitigations


def _run_seed_worker(seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    """Process-pool entry point. Returns plain dicts because dataclasses defined
    in this module pickle fine, but dicts keep the boundary explicit."""
    trainings, evaluations, mitigations = run_single_seed(seed, verbose=False)
    return (
        [asdict(t) for t in trainings],
        [asdict(e) for e in evaluations],
        [asdict(m) for m in mitigations],
    )


def _summarise(values: list[float]) -> dict:
    """Mean, sample standard deviation and standard error over seeds.

    ``ddof=1`` because these seeds are a sample of possible runs, not the
    population. With a handful of seeds the distinction is not pedantic: ddof=0
    would understate the spread by around 7% at n=8.
    """
    array = np.asarray(values, dtype=float)
    n = len(array)
    std = float(array.std(ddof=1)) if n > 1 else 0.0
    return {
        "mean": float(array.mean()),
        "std": std,
        "sem": std / np.sqrt(n) if n > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
        "n": n,
    }


def aggregate(trainings: list[dict], evaluations: list[dict]) -> dict:
    """Collapse per-seed rows into mean/spread statistics."""
    training_rows = []
    for layers in LAYER_COUNTS:
        matching = [t for t in trainings if t["layers"] == layers]
        training_rows.append({
            "layers": layers,
            "circuit_depth": matching[0]["circuit_depth"],
            "two_qubit_gates": matching[0]["two_qubit_gates"],
            "train_accuracy": _summarise([t["train_accuracy"] for t in matching]),
            "final_loss": _summarise([t["final_loss"] for t in matching]),
        })

    evaluation_rows = []
    keys = []
    for row in evaluations:
        key = (row["layers"], row["label"])
        if key not in keys:
            keys.append(key)
    for layers, label in keys:
        matching = [e for e in evaluations if e["layers"] == layers and e["label"] == label]
        evaluation_rows.append({
            "layers": layers,
            "label": label,
            "mechanism": matching[0]["mechanism"],
            "strength": matching[0]["strength"],
            "test_accuracy": _summarise([e["test_accuracy"] for e in matching]),
            # Paired within each seed against that seed's own noiseless run, so
            # the dataset-split variance cancels. This is why the drop resolves
            # far more sharply than the absolute accuracy it is derived from.
            "accuracy_drop": _summarise([e["accuracy_drop"] for e in matching]),
            "mean_abs_proba_shift": _summarise([e["mean_abs_proba_shift"] for e in matching]),
        })
    return {"training": training_rows, "evaluation": evaluation_rows}


def aggregate_mitigations(mitigations: list[dict]) -> list[dict]:
    """Collapse the seed axis of the mitigation comparison.

    Layers are pooled: the mitigation question is about mechanisms, and v2
    established that depth changes how much error accumulates but not which
    mechanism produced it. Pooling therefore trades a distinction that carries
    little information here for error bars over four times as many samples.
    """
    rows = []
    keys = []
    for row in mitigations:
        key = (row["label"], row["strategy"])
        if key not in keys:
            keys.append(key)
    for label, strategy in keys:
        matching = [
            m for m in mitigations if m["label"] == label and m["strategy"] == strategy
        ]
        rows.append({
            "label": label,
            "mechanism": matching[0]["mechanism"],
            "strategy": strategy,
            "test_accuracy": _summarise([m["test_accuracy"] for m in matching]),
            "accuracy_drop": _summarise([m["accuracy_drop"] for m in matching]),
            "mean_abs_proba_shift": _summarise([m["mean_abs_proba_shift"] for m in matching]),
            "shift_reduction": _summarise([m["shift_reduction"] for m in matching]),
        })
    return rows


def depth_monotonicity(evaluations: list[dict], seeds: list[int]) -> dict:
    """In how many seeds does probability shift increase with every added layer?

    The headline claim of v1 was that confidence erosion scales with circuit
    depth. With one seed that was an observation; counting how often the strict
    ordering survives across seeds is the closest this design gets to testing it.
    """
    results = {}
    mechanisms = sorted({e["mechanism"] for e in evaluations} - {"none"})
    for mechanism in mechanisms:
        strengths = sorted({e["strength"] for e in evaluations if e["mechanism"] == mechanism})
        strongest = strengths[-1]
        held = 0
        for seed in seeds:
            series = []
            for layers in LAYER_COUNTS:
                match = [
                    e for e in evaluations
                    if e["seed"] == seed and e["layers"] == layers
                    and e["mechanism"] == mechanism and e["strength"] == strongest
                ]
                if match:
                    series.append(match[0]["mean_abs_proba_shift"])
            if len(series) == len(LAYER_COUNTS) and all(
                a < b for a, b in pairwise(series)
            ):
                held += 1
        results[mechanism] = {"strictly_increasing_in": held, "of_seeds": len(seeds)}
    return results


def run(
    output_dir: Path,
    seeds: list[int] | None = None,
    workers: int | None = None,
    verbose: bool = True,
) -> dict:
    """Run the sweep across seeds and persist per-seed and aggregated results."""
    seeds = list(seeds) if seeds else list(DEFAULT_SEEDS)
    conditions = all_conditions()
    if verbose:
        print(f"seeds:   {seeds}")
        print(f"sweep:   {len(LAYER_COUNTS)} layer counts x {len(conditions)} noise conditions")
        print(f"workers: {workers or 1}\n")

    trainings: list[dict] = []
    evaluations: list[dict] = []
    mitigations: list[dict] = []
    if workers and workers > 1 and len(seeds) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_seed_worker, s): s for s in seeds}
            for future in as_completed(futures):
                seed_trainings, seed_evaluations, seed_mitigations = future.result()
                trainings += seed_trainings
                evaluations += seed_evaluations
                mitigations += seed_mitigations
                if verbose:
                    print(f"  seed {futures[future]} finished "
                          f"({len({t['seed'] for t in trainings})}/{len(seeds)})")
    else:
        for seed in seeds:
            seed_trainings, seed_evaluations, seed_mitigations = run_single_seed(
                seed, verbose=verbose
            )
            trainings += [asdict(t) for t in seed_trainings]
            evaluations += [asdict(e) for e in seed_evaluations]
            mitigations += [asdict(m) for m in seed_mitigations]

    # Process completion order is nondeterministic; sort so the written files
    # are byte-identical across runs with the same seeds.
    trainings.sort(key=lambda t: (t["seed"], t["layers"]))
    evaluations.sort(key=lambda e: (e["seed"], e["layers"], e["mechanism"], e["strength"]))
    mitigations.sort(key=lambda m: (m["seed"], m["layers"], m["label"], m["strategy"]))

    payload = {
        "config": {
            "seeds": seeds,
            "layer_counts": LAYER_COUNTS,
            "train_shots": TRAIN_SHOTS,
            "eval_shots": EVAL_SHOTS,
            "max_iterations": MAX_ITERATIONS,
            "n_restarts": N_RESTARTS,
        },
        "per_seed": {
            "training": trainings,
            "evaluation": evaluations,
            "mitigation": mitigations,
        },
        "aggregate": aggregate(trainings, evaluations),
        "mitigation_aggregate": aggregate_mitigations(mitigations),
        "depth_monotonicity": depth_monotonicity(evaluations, seeds),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2))

    header = (
        "layers,label,mechanism,strength,n_seeds,"
        "test_accuracy_mean,test_accuracy_std,"
        "accuracy_drop_mean,accuracy_drop_std,accuracy_drop_sem,"
        "proba_shift_mean,proba_shift_std"
    )
    csv_lines = [header]
    for row in payload["aggregate"]["evaluation"]:
        accuracy = row["test_accuracy"]
        drop = row["accuracy_drop"]
        shift = row["mean_abs_proba_shift"]
        csv_lines.append(
            f"{row['layers']},\"{row['label']}\",{row['mechanism']},{row['strength']:g},"
            f"{accuracy['n']},{accuracy['mean']:.4f},{accuracy['std']:.4f},"
            f"{drop['mean']:.4f},{drop['std']:.4f},{drop['sem']:.4f},"
            f"{shift['mean']:.4f},{shift['std']:.4f}"
        )
    (output_dir / "results.csv").write_text("\n".join(csv_lines) + "\n")

    mitigation_lines = [
        "label,mechanism,strategy,n,"
        "proba_shift_mean,proba_shift_std,shift_reduction_mean,shift_reduction_std,"
        "accuracy_drop_mean,accuracy_drop_std"
    ]
    for row in payload["mitigation_aggregate"]:
        shift = row["mean_abs_proba_shift"]
        reduction = row["shift_reduction"]
        drop = row["accuracy_drop"]
        mitigation_lines.append(
            f"\"{row['label']}\",{row['mechanism']},{row['strategy']},{shift['n']},"
            f"{shift['mean']:.4f},{shift['std']:.4f},"
            f"{reduction['mean']:.4f},{reduction['std']:.4f},"
            f"{drop['mean']:.4f},{drop['std']:.4f}"
        )
    (output_dir / "mitigation.csv").write_text("\n".join(mitigation_lines) + "\n")

    return payload
