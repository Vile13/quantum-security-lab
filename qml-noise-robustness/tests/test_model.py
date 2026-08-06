"""Tests for the invariants that would otherwise fail silently.

These target failure modes that still produce plausible numbers: swapped
parameter binding, a noise model that never attaches, a readout bit taken from
the wrong end of the register. A run with any of those bugs completes and
prints accuracies -- which is why they need assertions rather than eyeballing.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data import ANGLE_MAX, ANGLE_MIN, load_moons
from src.model import WEIGHTS_PER_LAYER, VariationalClassifier, build_circuit, log_loss
from src.noise_models import all_conditions, depolarizing, device_like, ideal


def test_training_features_are_scaled_into_angle_range():
    x_train, _, _, _ = load_moons(seed=42)
    assert x_train.min() >= ANGLE_MIN - 1e-9
    assert x_train.max() <= ANGLE_MAX + 1e-9


def test_split_is_stratified_and_sized_as_configured():
    x_train, x_test, y_train, y_test = load_moons(n_samples=160, test_size=0.5, seed=42)
    assert len(x_train) == len(x_test) == 80
    assert y_train.mean() == pytest.approx(0.5, abs=0.02)
    assert y_test.mean() == pytest.approx(0.5, abs=0.02)


@pytest.mark.parametrize("layers", [2, 3, 5])
def test_weight_count_scales_with_layers(layers):
    _, feature_params, weight_params = build_circuit(layers)
    assert len(feature_params) == 2  # one per qubit
    assert len(weight_params) == layers * WEIGHTS_PER_LAYER


def test_entangler_count_scales_with_layers():
    """Layer count is the experiment's expressibility axis; it must also be
    the noise-exposure axis, otherwise the trade-off being plotted is not real."""
    counts = [
        VariationalClassifier(layers=n, shots=64, seed=1).two_qubit_gate_count for n in (2, 4)
    ]
    assert counts[1] > counts[0]


def test_readout_bit_belongs_to_qubit_zero():
    """An endianness slip here silently reads the wrong qubit.

    Anchored on an analytic value: with every weight zero the circuit is the
    identity, so qubit 0 must measure 0 with certainty.
    """
    classifier = VariationalClassifier(layers=2, shots=2048, seed=1)
    zeros = np.zeros(classifier.n_weights)
    assert classifier.predict_proba(np.array([[1.0, 2.0]]), zeros)[0] == 0.0

    # w[1] is the bias on qubit 0 in layer 0: RY(pi) maps |0> to |1>.
    bias = np.zeros(classifier.n_weights)
    bias[1] = np.pi
    assert classifier.predict_proba(np.array([[1.0, 2.0]]), bias)[0] == pytest.approx(1.0, abs=0.01)


def test_output_matches_analytic_rotation_probability():
    """RY(theta) on |0> gives P(1) = sin^2(theta/2); check the encoding path."""
    classifier = VariationalClassifier(layers=2, shots=8192, seed=1)
    weights = np.zeros(classifier.n_weights)
    weights[0] = 1.0  # unit input scale on qubit 0, layer 0
    x0 = np.pi / 2
    observed = classifier.predict_proba(np.array([[x0, 0.0]]), weights)[0]
    assert observed == pytest.approx(np.sin(x0 / 2) ** 2, abs=0.02)


def test_binding_maps_every_circuit_parameter_to_its_own_source():
    """Each bound column must come from the source its parameter name names.

    Checked directly rather than through the output: a swap between a feature
    and a weight changes the numbers but not their plausibility, and at some
    points in weight space the output barely moves with the input at all --
    so an output-based probe would be an unreliable detector.
    """
    classifier = VariationalClassifier(layers=2, shots=64, seed=1)
    x = np.array([[0.11, 0.22]])
    # Disjoint value ranges make a misrouted column unambiguous.
    weights = np.arange(classifier.n_weights, dtype=float) + 100.0
    bound = classifier._bind(x, weights)

    assert bound.shape == (1, len(classifier._circuit.parameters))
    for column, param in enumerate(classifier._circuit.parameters):
        index = int(param.name[param.name.index("[") + 1 : -1])
        expected = x[0, index] if param.name.startswith("x") else weights[index]
        assert bound[0, column] == expected, f"{param.name} was bound to the wrong source"


def test_output_spans_a_useful_range_over_the_weights():
    """Expressibility guard, stated as a range rather than a pairwise difference.

    Two arbitrary weight vectors can coincidentally give near-identical output
    even from a fully expressive model, so a pairwise probe is flaky. The
    architecture this replaced was genuinely near-flat and saturated at 0.70
    test accuracy -- this asserts the property that fix was about.
    """
    classifier = VariationalClassifier(layers=3, shots=4096, seed=1)
    sample = np.array([[0.1, 0.1]])
    rng = np.random.default_rng(0)
    outputs = [
        classifier.predict_proba(sample, rng.uniform(-2, 2, classifier.n_weights))[0]
        for _ in range(8)
    ]
    assert max(outputs) - min(outputs) > 0.3


def test_output_varies_across_the_input_range():
    """Sweeping one feature across its encoding range must move the output."""
    classifier = VariationalClassifier(layers=3, shots=4096, seed=1)
    weights = np.random.default_rng(11).uniform(-2, 2, classifier.n_weights)
    grid = np.stack([np.linspace(ANGLE_MIN, ANGLE_MAX, 9), np.full(9, 0.5)], axis=1)
    proba = classifier.predict_proba(grid, weights)
    assert proba.max() - proba.min() > 0.3


def test_probabilities_are_in_unit_interval():
    classifier = VariationalClassifier(layers=3, shots=512, seed=1)
    x_train, _, _, _ = load_moons(seed=42)
    proba = classifier.predict_proba(x_train, np.full(classifier.n_weights, 0.5))
    assert proba.shape == (len(x_train),)
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_noise_model_actually_changes_the_output():
    """A noise model attached to gate names the transpiler never emits is inert."""
    x_train, _, _, _ = load_moons(seed=42)
    clean = VariationalClassifier(layers=3, shots=4096, seed=1)
    weights = np.full(clean.n_weights, 0.4)
    noisy = VariationalClassifier(
        layers=3, noise_model=depolarizing(0.05).model, shots=4096, seed=1
    )
    difference = noisy.predict_proba(x_train, weights) - clean.predict_proba(x_train, weights)
    shift = np.abs(difference).mean()
    assert shift > 0.02, f"noise model appears inert (mean shift {shift:.4f})"


def test_stronger_noise_shifts_probabilities_further():
    x_train, _, _, _ = load_moons(seed=42)
    clean = VariationalClassifier(layers=3, shots=4096, seed=1)
    weights = np.full(clean.n_weights, 0.4)
    baseline = clean.predict_proba(x_train, weights)

    shifts = []
    for p in (0.002, 0.02):
        noisy = VariationalClassifier(
            layers=3, noise_model=depolarizing(p).model, shots=4096, seed=1
        )
        shifts.append(np.abs(noisy.predict_proba(x_train, weights) - baseline).mean())
    assert shifts[1] > shifts[0]


def test_log_loss_is_finite_at_saturated_probabilities():
    """Finite shot counts routinely produce exactly 0.0 or 1.0."""
    assert np.isfinite(log_loss(np.array([0.0, 1.0]), np.array([1, 0])))


def test_log_loss_rewards_correct_confident_predictions():
    good = log_loss(np.array([0.9, 0.1]), np.array([1, 0]))
    bad = log_loss(np.array([0.1, 0.9]), np.array([1, 0]))
    assert good < bad


def test_condition_set_has_one_ideal_reference():
    conditions = all_conditions()
    assert sum(c.is_ideal for c in conditions) == 1
    assert conditions[0].label == ideal().label
    assert {c.mechanism for c in conditions} >= {
        "none", "depolarizing", "amplitude_damping", "thermal_relaxation", "readout", "composite"
    }


def test_condition_labels_are_unique():
    labels = [c.label for c in all_conditions()]
    assert len(labels) == len(set(labels)), "duplicate labels would collide in the results table"


def test_device_like_model_covers_gates_and_readout():
    model = device_like().model
    assert model is not None
    instructions = set(model.noise_instructions)
    assert {"cx", "u"} <= instructions, "gate errors missing from the composite model"
    assert "measure" in instructions, "readout error missing from the composite model"
