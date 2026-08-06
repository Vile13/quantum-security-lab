"""Variational quantum classifier built directly on Qiskit primitives.

The classifier is implemented here rather than taken from
``qiskit-machine-learning`` because the experiment needs to swap the noise
model underneath a *fixed* set of trained weights, which requires control over
transpilation and parameter binding.

Architecture: data re-uploading
-------------------------------
Each layer re-encodes the input and then applies trainable rotations::

    for each layer:
        for each qubit q:  RY(w_a * x_q + w_b) ; RZ(w_c)
        CX(0, 1)

The readout is the probability that qubit 0 measures 1.

Why re-uploading rather than a single fixed feature map: an encode-once
architecture (``ZZFeatureMap`` followed by ``RealAmplitudes``, read out via
parity) was measured first and saturates at ~0.70 test accuracy on two moons,
against 0.963 for a classical RBF-SVM on the identical split. Doubling the
ansatz parameters there left the training loss unchanged to four decimals --
the added freedom was inert. Interleaving encoding with variation lifts the
model to a truncated Fourier series in the inputs, which reaches the classical
baseline with two qubits. See ``README.md`` for the measured comparison.

This matters for the experiment rather than being an aesthetic preference: a
model stuck at 0.70 cannot show noise-induced degradation, because the drop to
be measured is smaller than the resolution of the test set.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter, ParameterVector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import SamplerV2

N_QUBITS = 2
WEIGHTS_PER_LAYER = 6  # per layer: 2 qubits x (scale, bias, phase)


def build_circuit(layers: int) -> tuple[QuantumCircuit, list[Parameter], list[Parameter]]:
    """Return the re-uploading circuit plus its feature and weight parameters."""
    x = ParameterVector("x", N_QUBITS)
    w = ParameterVector("w", layers * WEIGHTS_PER_LAYER)
    circuit = QuantumCircuit(N_QUBITS)
    k = 0
    for _ in range(layers):
        for qubit in range(N_QUBITS):
            # The trainable scale w[k] lets each layer address a different
            # frequency of the input; without it every layer would re-encode
            # the same rotation and add nothing.
            circuit.ry(w[k] * x[qubit] + w[k + 1], qubit)
            k += 2
            circuit.rz(w[k], qubit)
            k += 1
        circuit.cx(0, 1)
    circuit.measure_all()
    return circuit, list(x), list(w)


class VariationalClassifier:
    """A two-qubit VQC bound to one specific simulator backend.

    One instance corresponds to one noise condition. Weights trained on an
    instance with ``noise_model=None`` can be evaluated on an instance with a
    noise model: the logical circuit -- and therefore the meaning of each
    weight -- is identical, only the backend differs.
    """

    def __init__(
        self,
        layers: int = 4,
        noise_model: NoiseModel | None = None,
        shots: int = 1024,
        seed: int = 42,
    ) -> None:
        self.layers = layers
        self.shots = shots
        self.seed = seed

        circuit, feature_params, weight_params = build_circuit(layers)
        self.n_weights = len(weight_params)

        self._backend = AerSimulator(noise_model=noise_model)
        # Transpile against this specific backend: with a noise model attached
        # Aer restricts the basis gates to those the noise model describes, so
        # a circuit transpiled for the ideal backend would not pick up the
        # intended errors.
        self._circuit = transpile(
            circuit, self._backend, seed_transpiler=seed, optimization_level=1
        )
        self._sampler = SamplerV2.from_backend(self._backend, default_shots=shots, seed=seed)

        # Bind by parameter identity, not by position: transpilation may
        # reorder ``circuit.parameters``, and confusing a feature with a weight
        # would still run and still produce plausible-looking accuracies.
        feature_index = {p: i for i, p in enumerate(feature_params)}
        weight_index = {p: i for i, p in enumerate(weight_params)}
        self._layout: list[tuple[bool, int]] = []
        for param in self._circuit.parameters:
            if param in feature_index:
                self._layout.append((True, feature_index[param]))
            elif param in weight_index:
                self._layout.append((False, weight_index[param]))
            else:  # pragma: no cover - would mean the circuit changed shape
                raise ValueError(f"unrecognised circuit parameter: {param!r}")

    @property
    def depth(self) -> int:
        """Transpiled circuit depth -- the quantity noise actually scales with."""
        return self._circuit.depth()

    @property
    def two_qubit_gate_count(self) -> int:
        """Number of two-qubit gates after transpilation."""
        ops = self._circuit.count_ops()
        return sum(count for name, count in ops.items() if name in {"cx", "cz", "ecr"})

    def _bind(self, x: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Assemble the ``(n_samples, n_circuit_params)`` binding array."""
        values = np.empty((len(x), len(self._layout)))
        for column, (is_feature, index) in enumerate(self._layout):
            values[:, column] = x[:, index] if is_feature else weights[index]
        return values

    def predict_proba(
        self, x: np.ndarray, weights: np.ndarray, shots: int | None = None
    ) -> np.ndarray:
        """Return the estimated probability that qubit 0 measures 1, per sample."""
        values = self._bind(x, weights)
        job = self._sampler.run([(self._circuit, values)], shots=shots or self.shots)
        raw = job.result()[0].data.meas.array  # (n_samples, shots, n_bytes), little-endian
        qubit0 = raw[..., -1] & 1  # least significant bit of the last byte
        return qubit0.mean(axis=1)

    def predict(self, x: np.ndarray, weights: np.ndarray, shots: int | None = None) -> np.ndarray:
        return (self.predict_proba(x, weights, shots) >= 0.5).astype(int)

    def accuracy(
        self, x: np.ndarray, y: np.ndarray, weights: np.ndarray, shots: int | None = None
    ) -> float:
        return float((self.predict(x, weights, shots) == y).mean())


def log_loss(probabilities: np.ndarray, y: np.ndarray, epsilon: float = 1e-7) -> float:
    """Binary cross-entropy.

    Clipping is not cosmetic: a finite shot count regularly yields an estimate
    of exactly 0.0 or 1.0, which would make the loss infinite and stall the
    optimiser at the first confidently-wrong sample.
    """
    p = np.clip(probabilities, epsilon, 1.0 - epsilon)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1.0 - p)))
