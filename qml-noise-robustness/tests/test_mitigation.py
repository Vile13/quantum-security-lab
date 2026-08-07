"""Tests for circuit folding and the two mitigation strategies.

Folding is the part most likely to break silently: if the transpiler cancels
``U^dag U``, the folded circuit is no noisier than the original, every scale
returns the same value, and zero-noise extrapolation reports a confident zero
improvement that looks like a finding about ZNE rather than a bug.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data import load_moons
from src.mitigation import (
    ZNE_SCALES,
    ReadoutMitigator,
    build_zne_classifiers,
    extrapolate_to_zero,
    mitigation_variants,
    readout_mitigated_proba,
)
from src.model import VariationalClassifier, build_unitary, fold_global
from src.noise_models import depolarizing, readout

# --- folding ----------------------------------------------------------------


@pytest.mark.parametrize("scale", [0, 2, 4, -1])
def test_fold_rejects_non_odd_scales(scale):
    unitary, _, _ = build_unitary(2)
    with pytest.raises(ValueError):
        fold_global(unitary, scale)


@pytest.mark.parametrize(("scale", "factor"), [(1, 1), (3, 3), (5, 5)])
def test_folding_multiplies_entangler_count(scale, factor):
    """The whole point of folding is more gates; if the transpiler optimises the
    repetitions away this is the assertion that notices."""
    base = VariationalClassifier(layers=3, shots=64, seed=1).two_qubit_gate_count
    folded = VariationalClassifier(layers=3, shots=64, seed=1, fold_scale=scale)
    assert folded.two_qubit_gate_count == base * factor


def test_folding_leaves_the_ideal_output_unchanged():
    """U (U^dag U)^k is mathematically U, so on a noiseless backend every scale
    must agree. A folding bug that changed the unitary would show up here."""
    x, _, _, _ = load_moons(seed=42)
    weights = np.random.default_rng(0).uniform(-1, 1, 18)
    outputs = [
        VariationalClassifier(layers=3, shots=16384, seed=1, fold_scale=scale)
        .predict_proba(x[:8], weights)
        for scale in ZNE_SCALES
    ]
    for folded in outputs[1:]:
        assert folded == pytest.approx(outputs[0], abs=0.02)


def test_folding_amplifies_noise():
    x, _, _, _ = load_moons(seed=42)
    weights = np.random.default_rng(0).uniform(-1, 1, 18)
    ideal = VariationalClassifier(layers=3, shots=8192, seed=1).predict_proba(x[:40], weights)
    shifts = []
    for scale in ZNE_SCALES:
        noisy = VariationalClassifier(
            layers=3, noise_model=depolarizing(0.01).model, shots=8192, seed=1, fold_scale=scale
        )
        shifts.append(np.abs(noisy.predict_proba(x[:40], weights) - ideal).mean())
    assert shifts[0] < shifts[1] < shifts[2]


# --- distributions ----------------------------------------------------------


def test_distribution_is_normalised_and_matches_the_marginal():
    classifier = VariationalClassifier(layers=3, shots=4096, seed=1)
    x, _, _, _ = load_moons(seed=42)
    weights = np.random.default_rng(0).uniform(-1, 1, classifier.n_weights)
    distribution = classifier.predict_distribution(x[:20], weights)
    assert distribution.shape == (20, 4)
    assert distribution.sum(axis=1) == pytest.approx(np.ones(20))
    # Derived from the same shots, so these agree up to the sampler's seeded
    # reuse rather than merely up to sampling error.
    marginal = VariationalClassifier.marginal_from_distribution(distribution)
    assert marginal == pytest.approx(classifier.predict_proba(x[:20], weights), abs=0.05)


# --- extrapolation ----------------------------------------------------------


def test_extrapolation_recovers_a_known_intercept():
    """Exact arithmetic check: points on a line must extrapolate to that line."""
    scales = (1, 3, 5)
    observed = np.array([[0.30, 0.60], [0.40, 0.50], [0.50, 0.40]])  # intercept 0.25 / 0.65
    assert extrapolate_to_zero(scales, observed) == pytest.approx([0.25, 0.65])


def test_extrapolation_clips_into_the_unit_interval():
    scales = (1, 3, 5)
    observed = np.array([[0.10], [0.40], [0.70]])  # intercept -0.05
    assert extrapolate_to_zero(scales, observed) == pytest.approx([0.0])


def test_extrapolation_of_a_flat_series_is_the_flat_value():
    """A mechanism folding cannot amplify -- readout error -- produces a flat
    series, and ZNE must then be a no-op rather than an extrapolation artifact."""
    scales = (1, 3, 5)
    observed = np.full((3, 4), 0.42)
    assert extrapolate_to_zero(scales, observed) == pytest.approx(np.full(4, 0.42))


# --- readout mitigation -----------------------------------------------------


def test_calibration_matrix_is_a_stochastic_matrix():
    mitigator = ReadoutMitigator(readout(0.1).model, shots=8192, seed=1)
    assert mitigator.matrix.shape == (4, 4)
    assert mitigator.matrix.sum(axis=0) == pytest.approx(np.ones(4), abs=1e-9)
    # Symmetric 10% flips per qubit: the diagonal should sit near 0.9^2.
    assert np.diag(mitigator.matrix) == pytest.approx(0.81, abs=0.03)


def test_calibration_matrix_is_identity_without_noise():
    mitigator = ReadoutMitigator(None, shots=4096, seed=1)
    assert mitigator.matrix == pytest.approx(np.eye(4), abs=1e-9)


def test_mitigator_output_stays_a_probability_distribution():
    mitigator = ReadoutMitigator(readout(0.1).model, shots=8192, seed=1)
    # Deliberately awkward input: the unconstrained solve returns negatives here.
    corrected = mitigator.apply(np.array([[0.25, 0.25, 0.25, 0.25], [0.9, 0.05, 0.03, 0.02]]))
    assert np.all(corrected >= 0.0)
    assert corrected.sum(axis=1) == pytest.approx(np.ones(2))


def test_readout_mitigation_recovers_a_pure_readout_error():
    x, _, _, _ = load_moons(seed=42)
    weights = np.random.default_rng(0).uniform(-1, 1, 18)
    ideal = VariationalClassifier(layers=3, shots=16384, seed=1).predict_proba(x[:40], weights)
    noisy = VariationalClassifier(
        layers=3, noise_model=readout(0.1).model, shots=16384, seed=1
    )
    mitigator = ReadoutMitigator(readout(0.1).model, shots=16384, seed=1)
    before = np.abs(noisy.predict_proba(x[:40], weights) - ideal).mean()
    after = np.abs(readout_mitigated_proba(noisy, mitigator, x[:40], weights) - ideal).mean()
    assert after < before / 2, f"readout mitigation barely helped ({before:.4f} -> {after:.4f})"


# --- the mechanism-specificity claim ----------------------------------------


def test_zne_does_nothing_to_a_pure_readout_error():
    """Folding repeats the unitary; readout error applies once at measurement
    and is therefore not amplified. ZNE must be a no-op here -- this is the
    control that separates a real mechanism from a number-smoothing effect."""
    x, _, _, _ = load_moons(seed=42)
    weights = np.random.default_rng(0).uniform(-1, 1, 18)
    classifiers = build_zne_classifiers(3, readout(0.1).model, 8192, 1)
    mitigator = ReadoutMitigator(readout(0.1).model, shots=8192, seed=1)
    variants = mitigation_variants(classifiers, mitigator, x[:40], weights)
    assert variants["zne"] == pytest.approx(variants["none"], abs=0.02)


def test_all_strategies_are_returned_and_shaped_correctly():
    x, _, _, _ = load_moons(seed=42)
    weights = np.random.default_rng(0).uniform(-1, 1, 18)
    classifiers = build_zne_classifiers(3, depolarizing(0.01).model, 4096, 1)
    mitigator = ReadoutMitigator(depolarizing(0.01).model, shots=4096, seed=1)
    variants = mitigation_variants(classifiers, mitigator, x[:15], weights)
    assert set(variants) == {"none", "readout", "zne", "readout+zne"}
    for values in variants.values():
        assert values.shape == (15,)
        assert np.all((values >= 0.0) & (values <= 1.0))
