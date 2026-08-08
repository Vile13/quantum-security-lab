"""L-infinity adversarial attacks, and the control that makes them interpretable.

Three perturbations of the same budget are applied to every model:

``random``   -- a uniformly random sign pattern of magnitude epsilon
``fgsm``     -- one step along the sign of the input gradient
``pgd``      -- several smaller steps, re-projected into the budget each time

The random attack is not filler. A gradient attack that flips no more labels
than a random perturbation of the same size has demonstrated nothing about the
model, and a gradient attack that flips many more has demonstrated that the
decision boundary is genuinely close in a *findable* direction. Without the
control, a low flip rate reads as robustness when it may only mean the gradient
was too noisy to follow -- a failure mode this module has to rule out, because
its gradients come from a finite number of shots.

The budget is L-infinity on the *encoded* inputs, which live in [0, pi] (see
``qml_lab.data``). An epsilon of 0.1 is therefore about 3% of the encoding
range, not 10% -- §"Setup" in the README states the conversion.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

# Gradient of the loss with respect to the inputs, given (x, y).
GradientFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
# Hard 0/1 predictions, given x.
PredictFn = Callable[[np.ndarray], np.ndarray]

PGD_STEPS = 5
# Each step moves a third of the budget, so the attack can travel roughly
# 1.67x epsilon in total and still be projected back -- enough to round a curved
# boundary rather than only move along the initial gradient direction.
PGD_STEP_FRACTION = 1.0 / 3.0


def _clip_to_ball(perturbed: np.ndarray, original: np.ndarray, epsilon: float) -> np.ndarray:
    """Project back into the L-infinity ball around the original input.

    No clipping to the valid encoding range is applied. Angles outside [0, pi]
    are still legal circuit inputs -- they simply rotate further -- so bounding
    them would be an artificial constraint on the attacker rather than a
    property of the model. The README's limitations section says so explicitly.
    """
    return original + np.clip(perturbed - original, -epsilon, epsilon)


def random_attack(
    x: np.ndarray, epsilon: float, rng: np.random.Generator
) -> np.ndarray:
    """Uniformly random corner of the L-infinity ball -- the control."""
    return x + epsilon * rng.choice([-1.0, 1.0], size=x.shape)


def fgsm_attack(
    x: np.ndarray, y: np.ndarray, epsilon: float, gradient_fn: GradientFn
) -> np.ndarray:
    """One full-budget step along the sign of the loss gradient."""
    return x + epsilon * np.sign(gradient_fn(x, y))


def pgd_attack(
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    gradient_fn: GradientFn,
    steps: int = PGD_STEPS,
    step_fraction: float = PGD_STEP_FRACTION,
) -> np.ndarray:
    """Iterated FGSM with projection back into the budget after every step."""
    step_size = epsilon * step_fraction
    adversarial = x.copy()
    for _ in range(steps):
        adversarial = adversarial + step_size * np.sign(gradient_fn(adversarial, y))
        adversarial = _clip_to_ball(adversarial, x, epsilon)
    return adversarial


def attack_outcome(
    x: np.ndarray,
    y: np.ndarray,
    adversarial: np.ndarray,
    predict_fn: PredictFn,
    clean_prediction: np.ndarray,
) -> dict:
    """Score one attack.

    ``flip_rate`` is computed over the samples the model got *right* before the
    attack. Points already misclassified cannot be flipped into an error, and
    counting them would make a weak model look robust simply by having fewer
    correct predictions left to lose.
    """
    adversarial_prediction = predict_fn(adversarial)
    correct_before = clean_prediction == y
    n_correct = int(correct_before.sum())
    flipped = (adversarial_prediction != clean_prediction) & correct_before
    return {
        "clean_accuracy": float((clean_prediction == y).mean()),
        "adversarial_accuracy": float((adversarial_prediction == y).mean()),
        "flip_rate": float(flipped.sum() / n_correct) if n_correct else 0.0,
        "n_correct_before": n_correct,
        "mean_perturbation": float(np.abs(adversarial - x).mean()),
    }
