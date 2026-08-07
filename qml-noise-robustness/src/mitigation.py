"""Two error mitigation strategies, chosen because they target different causes.

``ReadoutMitigator`` corrects the measurement step and leaves the circuit
untouched. Zero-noise extrapolation targets accumulated *gate* error and
ignores measurement entirely. Applying both to the same set of noise conditions
turns "does mitigation help" into the sharper question of whether each one
helps specifically where its own mechanism lives -- which is checkable, and
which a strategy that merely smoothed the numbers would fail.

Neither is novel; both are standard. What matters for this module is measuring
what they recover against the unmitigated baseline from v2, using the same
probability-shift metric that turned out to be the sensitive one.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import SamplerV2

from .model import N_QUBITS, VariationalClassifier

# Odd scales only -- global folding produces U (U^dag U)^k, so the gate count
# is multiplied by an odd integer. Three points is the minimum for a linear fit
# that is not simply a line through two points.
ZNE_SCALES = (1, 3, 5)


class ReadoutMitigator:
    """Calibration-matrix correction of measurement assignment error.

    Each basis state is prepared and measured on the *same* backend the model
    runs on. Column ``b`` of the calibration matrix is then the distribution
    observed when ``|b>`` was prepared, so ``measured = M @ true`` and the
    correction is a linear solve.

    Limitation worth stating plainly: the preparation circuits use X gates, so
    on a backend with gate errors the calibration matrix absorbs a little gate
    error too and is not a pure readout characterisation. That is true of this
    technique in general, not just of this implementation.
    """

    def __init__(
        self,
        noise_model: NoiseModel | None,
        shots: int = 8192,
        seed: int = 42,
    ) -> None:
        self.n_states = 2**N_QUBITS
        backend = AerSimulator(noise_model=noise_model)
        sampler = SamplerV2.from_backend(backend, default_shots=shots, seed=seed)

        circuits = []
        for state in range(self.n_states):
            circuit = QuantumCircuit(N_QUBITS)
            for qubit in range(N_QUBITS):
                if state >> qubit & 1:  # little-endian, matching the readout
                    circuit.x(qubit)
            circuit.measure_all()
            circuits.append(transpile(circuit, backend, seed_transpiler=seed))

        matrix = np.zeros((self.n_states, self.n_states))
        for state, circuit in enumerate(circuits):
            raw = sampler.run([(circuit,)]).result()[0].data.meas.array
            outcomes = raw[..., -1].astype(int).ravel()
            counts = np.bincount(outcomes, minlength=self.n_states)[: self.n_states]
            matrix[:, state] = counts / counts.sum()
        self.matrix = matrix

    def apply(self, distributions: np.ndarray) -> np.ndarray:
        """Correct ``(n_samples, n_states)`` measured distributions."""
        # Least squares rather than a direct inverse: the calibration matrix is
        # estimated from finite shots and can be near-singular at high error
        # rates, where an explicit inverse amplifies the estimation noise.
        corrected, *_ = np.linalg.lstsq(self.matrix, distributions.T, rcond=None)
        corrected = corrected.T
        # The solve is unconstrained and routinely returns small negative
        # probabilities. Clipping and renormalising is the standard remedy; it
        # biases the result slightly, which is preferable to a negative
        # probability propagating into the loss.
        corrected = np.clip(corrected, 0.0, None)
        totals = corrected.sum(axis=1, keepdims=True)
        return np.divide(
            corrected, totals, out=np.full_like(corrected, 1.0 / self.n_states),
            where=totals > 0,
        )


def readout_mitigated_proba(
    classifier: VariationalClassifier,
    mitigator: ReadoutMitigator,
    x: np.ndarray,
    weights: np.ndarray,
    shots: int | None = None,
) -> np.ndarray:
    """P(qubit 0 = 1) after correcting the measured joint distribution."""
    measured = classifier.predict_distribution(x, weights, shots=shots)
    return VariationalClassifier.marginal_from_distribution(mitigator.apply(measured))


def extrapolate_to_zero(scales: tuple[int, ...], observed: np.ndarray) -> np.ndarray:
    """Fit a line through ``(scale, value)`` per sample and evaluate it at zero.

    ``observed`` is ``(n_scales, n_samples)``.

    Linear rather than Richardson extrapolation: with three shot-noisy points a
    polynomial forced exactly through all of them mostly extrapolates the
    sampling error, and does so with a large lever arm.
    """
    _slope, intercept = np.polyfit(np.asarray(scales, dtype=float), observed, deg=1)
    # The fit is unconstrained, so the intercept can leave [0, 1] -- most often
    # where the model was already saturated near a boundary. Clipping keeps the
    # output a probability; the raw intercept is not otherwise used.
    return np.clip(intercept, 0.0, 1.0)


def zne_proba(
    classifiers: dict[int, VariationalClassifier],
    x: np.ndarray,
    weights: np.ndarray,
    scales: tuple[int, ...] = ZNE_SCALES,
    shots: int | None = None,
) -> np.ndarray:
    """Extrapolate P(qubit 0 = 1) back to zero noise from amplified runs.

    ``classifiers`` maps each fold scale to a classifier built at that scale on
    the same backend.
    """
    observed = np.stack([
        classifiers[scale].predict_proba(x, weights, shots=shots) for scale in scales
    ])
    return extrapolate_to_zero(scales, observed)


def mitigation_variants(
    classifiers: dict[int, VariationalClassifier],
    mitigator: ReadoutMitigator,
    x: np.ndarray,
    weights: np.ndarray,
    scales: tuple[int, ...] = ZNE_SCALES,
    shots: int | None = None,
) -> dict[str, np.ndarray]:
    """All four strategies derived from one set of measurements.

    Each fold scale is sampled once and both the raw and the readout-corrected
    marginal are derived from those same shots. Beyond saving roughly half the
    runs, this makes the four strategies a paired comparison: any difference
    between them is the mitigation, not a different draw of shot noise.
    """
    raw_by_scale, corrected_by_scale = [], []
    for scale in scales:
        distribution = classifiers[scale].predict_distribution(x, weights, shots=shots)
        raw_by_scale.append(VariationalClassifier.marginal_from_distribution(distribution))
        corrected_by_scale.append(
            VariationalClassifier.marginal_from_distribution(mitigator.apply(distribution))
        )

    unfolded = scales.index(1)
    return {
        "none": raw_by_scale[unfolded],
        "readout": corrected_by_scale[unfolded],
        "zne": extrapolate_to_zero(scales, np.stack(raw_by_scale)),
        "readout+zne": extrapolate_to_zero(scales, np.stack(corrected_by_scale)),
    }


STRATEGIES = ("none", "readout", "zne", "readout+zne")


def build_zne_classifiers(
    layers: int,
    noise_model: NoiseModel | None,
    shots: int,
    seed: int,
    scales: tuple[int, ...] = ZNE_SCALES,
) -> dict[int, VariationalClassifier]:
    """One classifier per fold scale, all on the same backend."""
    return {
        scale: VariationalClassifier(
            layers=layers, noise_model=noise_model, shots=shots, seed=seed, fold_scale=scale
        )
        for scale in scales
    }
