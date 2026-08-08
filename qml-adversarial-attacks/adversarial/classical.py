"""The classical reference model, attacked with exactly the same machinery.

An RBF-SVM rather than a neural network, for two reasons. It is already the
classical baseline quoted in ``qml-noise-robustness`` (0.963 on this data), so
the comparison connects to a number the reader has seen; and its decision
function has a closed-form input gradient, so both sides of the comparison use
exact gradients rather than one exact and one approximated.

    f(x)      = sum_i  a_i K(x_i, x) + b,   K(x_i, x) = exp(-gamma ||x - x_i||^2)
    grad f(x) = sum_i  a_i K(x_i, x) * (-2 gamma) (x - x_i)

where ``a_i`` are sklearn's ``dual_coef_`` (already carrying the label sign)
and ``x_i`` the support vectors.
"""

from __future__ import annotations

import numpy as np
from sklearn.svm import SVC


class ClassicalReference:
    """RBF-SVM with an analytic input gradient and a probability-like output."""

    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, gamma: float | str = "scale"):
        self.svm = SVC(kernel="rbf", gamma=gamma).fit(x_train, y_train)
        self.gamma = float(self.svm._gamma)
        self.support_vectors = self.svm.support_vectors_
        self.dual_coef = self.svm.dual_coef_.ravel()

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        return self.svm.decision_function(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.svm.predict(x).astype(int)

    def _kernel(self, x: np.ndarray) -> np.ndarray:
        """``(n_samples, n_support)`` RBF kernel matrix."""
        squared = ((x[:, None, :] - self.support_vectors[None, :, :]) ** 2).sum(axis=2)
        return np.exp(-self.gamma * squared)

    def decision_gradient(self, x: np.ndarray) -> np.ndarray:
        """``d f / d x`` in closed form, shape ``(n_samples, n_features)``."""
        kernel = self._kernel(x)  # (n, s)
        weighted = kernel * self.dual_coef[None, :]  # (n, s)
        differences = x[:, None, :] - self.support_vectors[None, :, :]  # (n, s, d)
        return (-2.0 * self.gamma) * np.einsum("ns,nsd->nd", weighted, differences)

    def loss_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Gradient of a margin loss that increases as the sample is misclassified.

        Using ``L = -signed_margin`` where the sign follows the true label makes
        ascent on ``L`` push a sample across the boundary -- the same direction
        the cross-entropy gradient points for the quantum model, so the two
        attacks are doing the same thing rather than merely sharing a name.
        """
        signs = np.where(y == 1, 1.0, -1.0)
        return -signs[:, None] * self.decision_gradient(x)
