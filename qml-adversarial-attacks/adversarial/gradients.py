"""Exact input gradients of the quantum classifier via the parameter-shift rule.

An adversarial attack is only as good as its gradient. Finite differences would
be simpler, but on a shot-noisy circuit the step size trades bias against
variance with no good setting, and a weak gradient makes an attack look like
evidence of robustness when it is only evidence of a bad attack.

The parameter-shift rule gives the exact derivative in expectation. For a gate
``RY(theta)`` the identity is

    d<O>/d(theta) = [ <O>(theta + pi/2) - <O>(theta - pi/2) ] / 2

which is exact for any shot budget, not a small-step approximation.

Getting from there to d/dx takes one observation. The encoding gate angle is
``theta_l = w_scale_l * x_q + w_bias_l``, so a shift of the *bias* by +-pi/2
shifts that gate's angle by exactly +-pi/2 -- and the bias is already a
parameter of the circuit. So the shift can be applied through the existing
weight vector, without rebuilding the circuit with per-gate parameters, and

    d f / d x_q = sum over layers of  w_scale_l * (df / d theta_l)

by the chain rule.
"""

from __future__ import annotations

import numpy as np

from qml_lab.model import N_QUBITS, WEIGHTS_PER_LAYER, VariationalClassifier

# Layout of one layer's six weights, as built in qml_lab.model.build_unitary:
# per qubit, (input scale, bias, phase). Qubit q therefore owns indices
# 3*q + 0 (scale), 3*q + 1 (bias), 3*q + 2 (phase) within its layer block.
SCALE_OFFSET = 0
BIAS_OFFSET = 1


def _weight_index(layer: int, qubit: int, offset: int) -> int:
    return layer * WEIGHTS_PER_LAYER + 3 * qubit + offset


def input_gradient(
    classifier: VariationalClassifier,
    x: np.ndarray,
    weights: np.ndarray,
    shots: int | None = None,
) -> np.ndarray:
    """Return ``d P(qubit 0 = 1) / d x`` with shape ``(n_samples, n_features)``.

    Costs ``2 * layers * n_features`` circuit evaluations, each batched over all
    samples at once.
    """
    layers = classifier.layers
    gradient = np.zeros((len(x), N_QUBITS))

    for qubit in range(N_QUBITS):
        for layer in range(layers):
            bias_index = _weight_index(layer, qubit, BIAS_OFFSET)
            scale = weights[_weight_index(layer, qubit, SCALE_OFFSET)]

            shifted_up = weights.copy()
            shifted_up[bias_index] += np.pi / 2
            shifted_down = weights.copy()
            shifted_down[bias_index] -= np.pi / 2

            partial = (
                classifier.predict_proba(x, shifted_up, shots=shots)
                - classifier.predict_proba(x, shifted_down, shots=shots)
            ) / 2.0
            # Chain rule: this gate's angle moves with the input at rate
            # ``scale``, and every layer contributes to the same input feature.
            gradient[:, qubit] += scale * partial

    return gradient


def loss_gradient(
    classifier: VariationalClassifier,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    shots: int | None = None,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """Gradient of the per-sample binary cross-entropy with respect to ``x``.

    For ``L = -[y log p + (1-y) log(1-p)]`` the derivative is
    ``dL/dp = (p - y) / (p (1 - p))``, and the chain rule carries it to ``x``.
    """
    proba = classifier.predict_proba(x, weights, shots=shots)
    clipped = np.clip(proba, epsilon, 1.0 - epsilon)
    d_loss_d_proba = (clipped - y) / (clipped * (1.0 - clipped))
    return d_loss_d_proba[:, None] * input_gradient(classifier, x, weights, shots=shots)
