"""Noise models representing distinct physical failure mechanisms.

Each builder isolates one mechanism so that an accuracy drop can be attributed
to a cause rather than to "noise" in the aggregate. The composite model at the
end combines them at magnitudes in the range published for superconducting
NISQ devices, and stands in for what a real backend would do.

All error rates are declared as module-level constants so that a reader can
check the assumptions without reading the code that consumes them.
"""

from __future__ import annotations

from dataclasses import dataclass

from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    amplitude_damping_error,
    depolarizing_error,
    thermal_relaxation_error,
)

# Basis gates that Aer emits for a two-qubit transpilation. Errors have to be
# attached to the gate names that actually survive transpilation -- attaching
# them to, say, "ry" when the transpiler emitted "u" yields a noise model that
# is silently inert.
ONE_QUBIT_GATES = ["u1", "u2", "u3", "u", "p", "rx", "ry", "rz", "h", "x", "y", "z", "sx", "id"]
TWO_QUBIT_GATES = ["cx", "cz", "ecr", "swap"]

# Coherence times and gate durations typical of superconducting hardware.
T1_NS = 50_000.0
T2_NS = 70_000.0  # must satisfy T2 <= 2*T1
ONE_QUBIT_GATE_NS = 50.0
TWO_QUBIT_GATE_NS = 300.0
READOUT_NS = 1_000.0


@dataclass(frozen=True)
class NoiseCondition:
    """A named noise model together with the mechanism and strength it encodes."""

    label: str
    mechanism: str
    strength: float
    model: NoiseModel | None

    @property
    def is_ideal(self) -> bool:
        return self.model is None


def ideal() -> NoiseCondition:
    """Noiseless reference. Residual error is shot noise only."""
    return NoiseCondition("ideal", "none", 0.0, None)


def depolarizing(p_one_qubit: float) -> NoiseCondition:
    """Depolarizing noise: the state decays toward the maximally mixed state.

    Two-qubit gates are given ten times the single-qubit error rate, which is
    the order of magnitude reported for current superconducting devices. This
    is why circuit depth and entangler count matter more than qubit count at
    this scale.
    """
    model = NoiseModel()
    p_two_qubit = min(10.0 * p_one_qubit, 1.0)
    model.add_all_qubit_quantum_error(depolarizing_error(p_one_qubit, 1), ONE_QUBIT_GATES)
    model.add_all_qubit_quantum_error(depolarizing_error(p_two_qubit, 2), TWO_QUBIT_GATES)
    return NoiseCondition(f"depolarizing p={p_one_qubit:g}", "depolarizing", p_one_qubit, model)


def amplitude_damping(gamma: float) -> NoiseCondition:
    """Energy relaxation toward |0>.

    Unlike depolarizing noise this is directional: it biases measurement
    outcomes toward the all-zero bitstring, which for parity decoding means a
    systematic pull toward class 0 rather than toward a coin flip.
    """
    model = NoiseModel()
    one_qubit = amplitude_damping_error(gamma)
    model.add_all_qubit_quantum_error(one_qubit, ONE_QUBIT_GATES)
    model.add_all_qubit_quantum_error(one_qubit.tensor(one_qubit), TWO_QUBIT_GATES)
    return NoiseCondition(f"amplitude damping g={gamma:g}", "amplitude_damping", gamma, model)


def thermal_relaxation(scale: float) -> NoiseCondition:
    """T1/T2 relaxation, with coherence times divided by ``scale``.

    ``scale=1`` corresponds to the constants declared above; larger values
    model a worse device. Expressing the sweep as a divisor rather than as
    absolute times keeps the T2 <= 2*T1 relation intact at every point.
    """
    t1 = T1_NS / scale
    t2 = T2_NS / scale
    model = NoiseModel()
    one_qubit = thermal_relaxation_error(t1, t2, ONE_QUBIT_GATE_NS)
    two_qubit = thermal_relaxation_error(t1, t2, TWO_QUBIT_GATE_NS).tensor(
        thermal_relaxation_error(t1, t2, TWO_QUBIT_GATE_NS)
    )
    model.add_all_qubit_quantum_error(one_qubit, ONE_QUBIT_GATES)
    model.add_all_qubit_quantum_error(two_qubit, TWO_QUBIT_GATES)
    return NoiseCondition(f"thermal T1={t1/1000:.0f}us", "thermal_relaxation", scale, model)


def readout(p_flip: float) -> NoiseCondition:
    """Symmetric measurement assignment error.

    This mechanism is singled out because it is the one an attacker or a
    miscalibrated device can influence *after* the computation is finished, and
    because it is classically correctable -- unlike the gate errors above.
    """
    model = NoiseModel()
    model.add_all_qubit_readout_error(
        ReadoutError([[1.0 - p_flip, p_flip], [p_flip, 1.0 - p_flip]])
    )
    return NoiseCondition(f"readout p={p_flip:g}", "readout", p_flip, model)


def device_like() -> NoiseCondition:
    """All mechanisms combined at plausible present-day magnitudes."""
    model = NoiseModel()
    depol_one = depolarizing_error(0.001, 1)
    depol_two = depolarizing_error(0.01, 2)
    thermal_one = thermal_relaxation_error(T1_NS, T2_NS, ONE_QUBIT_GATE_NS)
    thermal_two = thermal_relaxation_error(T1_NS, T2_NS, TWO_QUBIT_GATE_NS).tensor(
        thermal_relaxation_error(T1_NS, T2_NS, TWO_QUBIT_GATE_NS)
    )
    model.add_all_qubit_quantum_error(depol_one.compose(thermal_one), ONE_QUBIT_GATES)
    model.add_all_qubit_quantum_error(depol_two.compose(thermal_two), TWO_QUBIT_GATES)
    model.add_all_qubit_readout_error(ReadoutError([[0.98, 0.02], [0.03, 0.97]]))
    return NoiseCondition("device-like composite", "composite", 1.0, model)


# Sweep points. Kept as explicit lists rather than generated ranges so that a
# result table can be traced back to the exact parameters that produced it.
DEPOLARIZING_SWEEP = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]
AMPLITUDE_DAMPING_SWEEP = [0.001, 0.005, 0.01, 0.02, 0.05]
THERMAL_SWEEP = [1.0, 2.0, 5.0, 10.0, 25.0]
READOUT_SWEEP = [0.005, 0.01, 0.02, 0.05, 0.1]


def all_conditions() -> list[NoiseCondition]:
    """Every noise condition evaluated by the experiment, ideal reference first."""
    conditions = [ideal()]
    conditions += [depolarizing(p) for p in DEPOLARIZING_SWEEP]
    conditions += [amplitude_damping(g) for g in AMPLITUDE_DAMPING_SWEEP]
    conditions += [thermal_relaxation(s) for s in THERMAL_SWEEP]
    conditions += [readout(p) for p in READOUT_SWEEP]
    conditions.append(device_like())
    return conditions
