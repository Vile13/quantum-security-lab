"""Dataset construction for the noise-robustness experiments.

The classifier operates on two qubits, so the dataset is deliberately
two-dimensional: every feature maps onto exactly one qubit rotation angle and
no dimensionality reduction step sits between the data and the circuit. This
keeps the measured effect attributable to the quantum stack rather than to a
classical preprocessing pipeline.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# Feature values are used directly as rotation angles. Restricting them to
# [0, pi] keeps the encoding injective -- a wider range would wrap around the
# Bloch sphere and alias distinct samples onto identical states.
ANGLE_MIN = 0.0
ANGLE_MAX = np.pi


def _scale_to_angles(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scale features to [0, pi], fitting the range on the training split only."""
    lo = x_train.min(axis=0)
    hi = x_train.max(axis=0)
    span = np.where(hi - lo == 0.0, 1.0, hi - lo)

    def apply(x: np.ndarray) -> np.ndarray:
        unit = (x - lo) / span
        return ANGLE_MIN + unit * (ANGLE_MAX - ANGLE_MIN)

    # The test split is scaled with the training range and may fall slightly
    # outside [0, pi]. That is intended: clipping it would leak test-set
    # information into the preprocessing step.
    return apply(x_train), apply(x_test)


def load_moons(
    n_samples: int = 160,
    noise: float = 0.15,
    test_size: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(x_train, x_test, y_train, y_test)`` for a two-moons problem.

    Two moons is not linearly separable, so a classifier that scores well has
    to use the entangling part of the ansatz rather than a single-qubit
    decision boundary. That matters here: the entangling gates are also the
    gates that carry the largest hardware error rates, which is precisely the
    trade-off the experiment is meant to expose.
    """
    x, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )
    x_train, x_test = _scale_to_angles(x_train, x_test)
    return x_train, x_test, y_train.astype(int), y_test.astype(int)
