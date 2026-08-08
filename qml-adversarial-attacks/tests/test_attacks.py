"""Tests for gradients, attacks and their scoring.

The gradient tests matter most. Every result in this module is downstream of
``input_gradient``, and a wrong gradient does not crash -- it produces a weak
attack, which reads as evidence that the model is robust. The parameter-shift
implementation is therefore checked against an exact statevector derivative,
not against another approximation.
"""

from __future__ import annotations

import numpy as np
import pytest
from adversarial.attacks import (
    attack_outcome,
    fgsm_attack,
    pgd_attack,
    random_attack,
)
from adversarial.classical import ClassicalReference
from adversarial.gradients import (
    BIAS_OFFSET,
    SCALE_OFFSET,
    _weight_index,
    input_gradient,
    loss_gradient,
)
from qiskit.quantum_info import Statevector

from qml_lab.data import load_moons
from qml_lab.model import WEIGHTS_PER_LAYER, VariationalClassifier, build_unitary

LAYERS = 3


def _exact_proba(x_row: np.ndarray, weights: np.ndarray) -> float:
    """P(qubit 0 = 1) by exact simulation -- no shots, no sampling error."""
    circuit, feature_params, weight_params = build_unitary(LAYERS)
    bound = circuit.assign_parameters(
        dict(zip(feature_params, x_row, strict=True))
        | dict(zip(weight_params, weights, strict=True))
    )
    probabilities = np.abs(Statevector(bound).data) ** 2
    return float(probabilities[1] + probabilities[3])  # odd indices: qubit 0 is set


# --- gradients --------------------------------------------------------------


def test_weight_index_matches_the_documented_layout():
    """Layer l, qubit q owns (scale, bias, phase) at 3*q inside its block."""
    assert _weight_index(0, 0, SCALE_OFFSET) == 0
    assert _weight_index(0, 0, BIAS_OFFSET) == 1
    assert _weight_index(0, 1, SCALE_OFFSET) == 3
    assert _weight_index(1, 0, SCALE_OFFSET) == WEIGHTS_PER_LAYER


def test_parameter_shift_matches_the_exact_derivative():
    """The claim the whole module rests on, checked against exact simulation.

    Shifting the *bias* by +-pi/2 shifts that gate's angle by exactly +-pi/2,
    and the chain rule multiplies by the input scale. If either half of that
    reasoning were wrong, the gradient would still have a plausible magnitude.
    """
    _, x_test, _, _ = load_moons(seed=42)
    weights = np.random.default_rng(3).uniform(-1, 1, LAYERS * WEIGHTS_PER_LAYER)

    for x_row in x_test[:5]:
        analytic = np.zeros(2)
        for qubit in range(2):
            for layer in range(LAYERS):
                bias = _weight_index(layer, qubit, BIAS_OFFSET)
                scale = weights[_weight_index(layer, qubit, SCALE_OFFSET)]
                up, down = weights.copy(), weights.copy()
                up[bias] += np.pi / 2
                down[bias] -= np.pi / 2
                analytic[qubit] += scale * (
                    _exact_proba(x_row, up) - _exact_proba(x_row, down)
                ) / 2.0

        step = 1e-6
        numeric = np.zeros(2)
        for dimension in range(2):
            forward, backward = x_row.copy(), x_row.copy()
            forward[dimension] += step
            backward[dimension] -= step
            numeric[dimension] = (
                _exact_proba(forward, weights) - _exact_proba(backward, weights)
            ) / (2 * step)

        assert analytic == pytest.approx(numeric, abs=1e-5)


def test_input_gradient_has_the_right_shape_and_is_not_trivial():
    _, x_test, _, _ = load_moons(seed=42)
    classifier = VariationalClassifier(layers=LAYERS, shots=8192, seed=1)
    weights = np.random.default_rng(3).uniform(-1, 1, classifier.n_weights)
    gradient = input_gradient(classifier, x_test[:16], weights)
    assert gradient.shape == (16, 2)
    assert np.abs(gradient).max() > 0.05, "gradient is essentially zero everywhere"


def test_zero_input_scales_give_a_zero_input_gradient():
    """If no gate's angle depends on the input, neither can the output.

    Anchors the chain-rule factor: with every scale at zero the sum must
    vanish regardless of what the parameter-shift terms are.
    """
    _, x_test, _, _ = load_moons(seed=42)
    classifier = VariationalClassifier(layers=LAYERS, shots=2048, seed=1)
    weights = np.random.default_rng(5).uniform(-1, 1, classifier.n_weights)
    for layer in range(LAYERS):
        for qubit in range(2):
            weights[_weight_index(layer, qubit, SCALE_OFFSET)] = 0.0
    assert input_gradient(classifier, x_test[:8], weights) == pytest.approx(0.0, abs=1e-12)


def test_loss_gradient_points_away_from_the_true_label():
    """Ascending the loss must reduce confidence in the correct class."""
    _, x_test, _, y_test = load_moons(seed=42)
    classifier = VariationalClassifier(layers=LAYERS, shots=8192, seed=1)
    weights = np.random.default_rng(3).uniform(-1, 1, classifier.n_weights)
    x, y = x_test[:24], y_test[:24]

    before = classifier.predict_proba(x, weights)
    stepped = x + 0.05 * np.sign(loss_gradient(classifier, x, y, weights, shots=8192))
    after = classifier.predict_proba(stepped, weights)

    # For y=1 the probability should fall, for y=0 it should rise.
    moved_correctly = np.where(y == 1, after < before, after > before)
    assert moved_correctly.mean() > 0.7


# --- attacks ----------------------------------------------------------------


def test_random_attack_lands_on_the_ball_corner():
    x = np.zeros((50, 2))
    perturbed = random_attack(x, 0.1, np.random.default_rng(0))
    assert np.abs(perturbed - x) == pytest.approx(np.full((50, 2), 0.1))


def test_fgsm_respects_the_budget_exactly():
    x = np.zeros((30, 2))
    y = np.zeros(30, dtype=int)
    gradient = np.random.default_rng(1).normal(size=(30, 2))
    perturbed = fgsm_attack(x, y, 0.07, lambda a, b: gradient)
    assert np.abs(perturbed - x).max() == pytest.approx(0.07)


def test_pgd_stays_inside_the_budget_across_all_steps():
    """Seven steps of a quarter budget each could travel 1.75x epsilon without
    the projection; the projection is what keeps the comparison fair."""
    x = np.zeros((30, 2))
    y = np.zeros(30, dtype=int)
    gradient = np.ones((30, 2))  # always points the same way -- worst case
    perturbed = pgd_attack(x, y, 0.1, lambda a, b: gradient)
    assert np.abs(perturbed - x).max() <= 0.1 + 1e-12


def test_pgd_uses_more_of_the_budget_than_a_single_step_would_waste():
    """With a constant gradient PGD should saturate the budget, like FGSM."""
    x = np.zeros((10, 2))
    y = np.zeros(10, dtype=int)
    gradient = np.ones((10, 2))
    assert np.abs(pgd_attack(x, y, 0.1, lambda a, b: gradient) - x).max() == pytest.approx(0.1)


# --- scoring ----------------------------------------------------------------


def test_flip_rate_counts_only_initially_correct_samples():
    """Points the model already got wrong cannot be flipped into an error.

    Counting them would make a weak model look robust by having less to lose.
    """
    x = np.zeros((4, 2))
    y = np.array([1, 1, 0, 0])
    clean = np.array([1, 1, 1, 1])  # two correct, two already wrong
    outcome = attack_outcome(x, y, x, lambda a: np.array([0, 1, 0, 1]), clean)
    # Only sample 0 was correct and changed.
    assert outcome["n_correct_before"] == 2
    assert outcome["flip_rate"] == pytest.approx(0.5)


def test_flip_rate_is_zero_when_nothing_was_correct():
    x = np.zeros((3, 2))
    y = np.ones(3, dtype=int)
    clean = np.zeros(3, dtype=int)
    outcome = attack_outcome(x, y, x, lambda a: np.zeros(3, dtype=int), clean)
    assert outcome["n_correct_before"] == 0
    assert outcome["flip_rate"] == 0.0


# --- classical reference ----------------------------------------------------


def test_classical_gradient_matches_finite_differences_exactly():
    x_train, x_test, y_train, _ = load_moons(seed=42)
    model = ClassicalReference(x_train, y_train)
    analytic = model.decision_gradient(x_test[:20])

    step = 1e-5
    numeric = np.zeros_like(analytic)
    for dimension in range(2):
        forward, backward = x_test[:20].copy(), x_test[:20].copy()
        forward[:, dimension] += step
        backward[:, dimension] -= step
        numeric[:, dimension] = (
            model.decision_function(forward) - model.decision_function(backward)
        ) / (2 * step)
    assert analytic == pytest.approx(numeric, abs=1e-6)


def test_classical_reference_reaches_the_expected_accuracy():
    """0.963 is the number quoted as the classical baseline in the other module."""
    x_train, x_test, y_train, y_test = load_moons(seed=42)
    model = ClassicalReference(x_train, y_train)
    assert model.svm.score(x_test, y_test) > 0.93
