"""Attack sweep: quantum vs classical, across epsilon, across seeds.

Both models are trained on the same split of the same data, attacked with the
same three perturbations at the same budgets, and scored with the same metric.
The quantum model is additionally attacked under device noise, which asks
whether the noise measured in ``qml-noise-robustness`` acts as an accidental
defence or as an additional weakness.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from qml_lab.data import load_moons
from qml_lab.model import VariationalClassifier, log_loss
from qml_lab.noise_models import device_like

from .attacks import attack_outcome, fgsm_attack, pgd_attack, random_attack
from .classical import ClassicalReference
from .gradients import loss_gradient

LAYERS = 3  # the layer count that trained best in qml-noise-robustness
TRAIN_SHOTS = 512
EVAL_SHOTS = 4096
# Gradients are shot-noisy and PGD consumes one per step, so this is the knob
# that decides the module's runtime. 4096 keeps the gradient's sign reliable --
# only the sign is used -- without making the sweep an overnight job.
GRADIENT_SHOTS = 4096
MAX_ITERATIONS = 600
N_RESTARTS = 3
DEFAULT_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49]

# Inputs are encoded into [0, pi], so epsilon is in radians. 0.05 rad is 1.6%
# of the encoding range, 0.4 rad is 12.7%.
EPSILONS = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4]
ATTACKS = ("random", "fgsm", "pgd")
MODELS = ("quantum", "quantum_noisy", "classical")


@dataclass
class AttackRow:
    seed: int
    model: str
    attack: str
    epsilon: float
    clean_accuracy: float
    adversarial_accuracy: float
    flip_rate: float
    n_correct_before: int
    mean_perturbation: float


def train_quantum(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> np.ndarray:
    """Same protocol as qml-noise-robustness: noiseless COBYLA, best of restarts."""
    classifier = VariationalClassifier(
        layers=LAYERS, noise_model=None, shots=TRAIN_SHOTS, seed=seed
    )

    def objective(weights: np.ndarray) -> float:
        return log_loss(classifier.predict_proba(x_train, weights), y_train)

    best_weights, best_loss = None, np.inf
    for restart in range(N_RESTARTS):
        initial = np.random.default_rng(seed + restart).uniform(-1.0, 1.0, classifier.n_weights)
        result = minimize(
            objective, initial, method="COBYLA",
            options={"maxiter": MAX_ITERATIONS, "rhobeg": 0.5, "tol": 1e-4},
        )
        if result.fun < best_loss:
            best_loss, best_weights = float(result.fun), np.asarray(result.x)
    assert best_weights is not None
    return best_weights


def _quantum_interface(classifier: VariationalClassifier, weights: np.ndarray):
    """Prediction and loss-gradient callables bound to one classifier."""

    def predict(x: np.ndarray) -> np.ndarray:
        return (classifier.predict_proba(x, weights, shots=EVAL_SHOTS) >= 0.5).astype(int)

    def gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return loss_gradient(classifier, x, y, weights, shots=GRADIENT_SHOTS)

    return predict, gradient


def run_single_seed(seed: int, verbose: bool = True) -> list[AttackRow]:
    """Train both models on one split and attack them across the epsilon grid."""
    x_train, x_test, y_train, y_test = load_moons(seed=seed)
    if verbose:
        print(f"[seed {seed}] training quantum model")
    weights = train_quantum(x_train, y_train, seed)

    classical = ClassicalReference(x_train, y_train)
    noiseless = VariationalClassifier(
        layers=LAYERS, noise_model=None, shots=EVAL_SHOTS, seed=seed
    )
    noisy = VariationalClassifier(
        layers=LAYERS, noise_model=device_like().model, shots=EVAL_SHOTS, seed=seed
    )

    interfaces = {
        "quantum": _quantum_interface(noiseless, weights),
        "quantum_noisy": _quantum_interface(noisy, weights),
        "classical": (classical.predict, classical.loss_gradient),
    }

    rows: list[AttackRow] = []
    for model in MODELS:
        predict, gradient = interfaces[model]
        clean_prediction = predict(x_test)

        # FGSM evaluates the gradient at the clean input, which does not depend
        # on the budget -- only the step length does. Caching it and handing
        # ``fgsm_attack`` a closure removes five sixths of this attack's cost
        # (a quantum gradient is 2 * layers * n_features circuit runs) without
        # duplicating the attack's definition here.
        clean_gradient = gradient(x_test, y_test)

        # Bound as a default argument rather than closed over: the loop
        # variable would otherwise be looked up at call time, which is correct
        # today only because the closure is consumed in the same iteration.
        def cached_gradient(_x: np.ndarray, _y: np.ndarray, cached=clean_gradient) -> np.ndarray:
            return cached

        for epsilon in EPSILONS:
            # One generator per (model, epsilon) so the random control sees the
            # same perturbation pattern for every model at a given budget --
            # otherwise the control's variance would be confounded with the
            # model comparison.
            rng = np.random.default_rng(seed * 1000 + int(epsilon * 1000))
            adversarial = {
                "random": random_attack(x_test, epsilon, rng),
                "fgsm": fgsm_attack(x_test, y_test, epsilon, cached_gradient),
                "pgd": pgd_attack(x_test, y_test, epsilon, gradient),
            }
            for attack in ATTACKS:
                outcome = attack_outcome(
                    x_test, y_test, adversarial[attack], predict, clean_prediction
                )
                rows.append(AttackRow(seed=seed, model=model, attack=attack,
                                      epsilon=epsilon, **outcome))
        if verbose:
            print(f"[seed {seed}] {model} done")
    return rows


def _run_seed_worker(seed: int) -> list[dict]:
    return [asdict(r) for r in run_single_seed(seed, verbose=False)]


def _summarise(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    n = len(array)
    std = float(array.std(ddof=1)) if n > 1 else 0.0
    return {
        "mean": float(array.mean()),
        "std": std,
        "sem": std / np.sqrt(n) if n > 1 else 0.0,
        "n": n,
    }


def aggregate(rows: list[dict]) -> list[dict]:
    """Collapse the seed axis, keeping model x attack x epsilon."""
    aggregated = []
    for model in MODELS:
        for attack in ATTACKS:
            for epsilon in EPSILONS:
                matching = [
                    r for r in rows
                    if r["model"] == model and r["attack"] == attack and r["epsilon"] == epsilon
                ]
                if not matching:
                    continue
                aggregated.append({
                    "model": model,
                    "attack": attack,
                    "epsilon": epsilon,
                    "flip_rate": _summarise([r["flip_rate"] for r in matching]),
                    "adversarial_accuracy": _summarise(
                        [r["adversarial_accuracy"] for r in matching]
                    ),
                    "clean_accuracy": _summarise([r["clean_accuracy"] for r in matching]),
                })
    return aggregated


def gradient_advantage(rows: list[dict], seeds: list[int]) -> dict:
    """In how many seeds does the gradient attack beat the random control?

    This is the check that decides whether any flip rate reported here means
    anything. A gradient attack that does not beat random noise of the same
    magnitude has not found the decision boundary -- it has only added noise,
    and a low flip rate would say nothing about the model's robustness.
    """
    result = {}
    for model in MODELS:
        for attack in ("fgsm", "pgd"):
            beaten = 0
            for seed in seeds:
                gradient_rates, random_rates = [], []
                for epsilon in EPSILONS:
                    def pick(name, e=epsilon, s=seed, m=model):
                        found = [
                            r for r in rows
                            if r["seed"] == s and r["model"] == m
                            and r["attack"] == name and r["epsilon"] == e
                        ]
                        return found[0]["flip_rate"] if found else None

                    gradient_rate, random_rate = pick(attack), pick("random")
                    if gradient_rate is not None and random_rate is not None:
                        gradient_rates.append(gradient_rate)
                        random_rates.append(random_rate)
                if gradient_rates and np.mean(gradient_rates) > np.mean(random_rates):
                    beaten += 1
            result[f"{model}/{attack}"] = {"beats_random_in": beaten, "of_seeds": len(seeds)}
    return result


def run(
    output_dir: Path,
    seeds: list[int] | None = None,
    workers: int | None = None,
    verbose: bool = True,
) -> dict:
    seeds = list(seeds) if seeds else list(DEFAULT_SEEDS)
    if verbose:
        print(f"seeds:   {seeds}")
        print(f"grid:    {len(MODELS)} models x {len(ATTACKS)} attacks x {len(EPSILONS)} budgets")
        print(f"workers: {workers or 1}\n")

    started = time.perf_counter()
    rows: list[dict] = []
    if workers and workers > 1 and len(seeds) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_seed_worker, s): s for s in seeds}
            for future in as_completed(futures):
                rows += future.result()
                if verbose:
                    print(f"  seed {futures[future]} finished "
                          f"({len({r['seed'] for r in rows})}/{len(seeds)})")
    else:
        for seed in seeds:
            rows += [asdict(r) for r in run_single_seed(seed, verbose=verbose)]

    rows.sort(key=lambda r: (r["seed"], r["model"], r["attack"], r["epsilon"]))

    payload = {
        "config": {
            "seeds": seeds,
            "layers": LAYERS,
            "epsilons": EPSILONS,
            "attacks": list(ATTACKS),
            "models": list(MODELS),
            "eval_shots": EVAL_SHOTS,
            "gradient_shots": GRADIENT_SHOTS,
            "seconds": time.perf_counter() - started,
        },
        "per_seed": rows,
        "aggregate": aggregate(rows),
        "gradient_advantage": gradient_advantage(rows, seeds),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2))

    lines = [
        "model,attack,epsilon,n,flip_rate_mean,flip_rate_std,"
        "adv_accuracy_mean,adv_accuracy_std"
    ]
    for row in payload["aggregate"]:
        flip, adversarial = row["flip_rate"], row["adversarial_accuracy"]
        lines.append(
            f"{row['model']},{row['attack']},{row['epsilon']:g},{flip['n']},"
            f"{flip['mean']:.4f},{flip['std']:.4f},"
            f"{adversarial['mean']:.4f},{adversarial['std']:.4f}"
        )
    (output_dir / "results.csv").write_text("\n".join(lines) + "\n")

    return payload
