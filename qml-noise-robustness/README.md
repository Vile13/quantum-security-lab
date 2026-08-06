# qml-noise-robustness

How much classification accuracy does a variational quantum classifier lose
when it meets a noisy device, which noise mechanism costs the most, and does
adding circuit depth buy more than it costs?

## 1. Research question

A variational quantum classifier is trained on a noiseless simulator and then
deployed to hardware whose error profile it never saw. Two questions follow:

1. **Per mechanism** — how does test accuracy degrade under depolarizing noise,
   amplitude damping, thermal relaxation (T1/T2), and readout assignment error,
   each varied in isolation?
2. **Per depth** — a deeper circuit is more expressive but contains more
   entangling gates, and entangling gates carry the largest error rates. Where
   does the trade-off turn?

## 2. Threat and failure model

The concern is not an attacker manipulating a quantum computer. It is that a
QML model's accuracy is a property of the *model together with the device it
runs on*, while it is usually reported as a property of the model alone.

| Failure | Operational consequence |
|---|---|
| Model validated on a simulator, deployed to hardware | Reported accuracy does not transfer; the gap is unquantified |
| Backend heterogeneity (scheduler picks a different device) | Accuracy varies run to run with no code change and no error |
| Device calibration drifts between runs | Silent, gradual degradation with no failure signal |
| Readout miscalibration | Systematic label bias -- and unlike gate error, correctable after the fact |

The common thread is silence: none of these raise an exception. The circuit
executes, returns a bitstring distribution, and the classifier returns a label.
The failure mode is a **wrong answer delivered with full confidence**, which is
why the experiment tracks probability shift alongside accuracy — see §6.

## 3. Setup

**Data.** `make_moons` (160 samples, `noise=0.15`, seed 42), split 80/80
stratified. Two features, so each maps to exactly one qubit and no classical
dimensionality reduction sits between the data and the circuit. Features are
scaled to `[0, pi]` with the range fitted on the training split only.

**Model.** Two qubits, data re-uploading. Each layer re-encodes the input and
then applies trainable rotations:

```
for each layer:
    for each qubit q:  RY(w_a * x_q + w_b) ; RZ(w_c)
    CX(0, 1)
```

Readout is `P(qubit 0 = 1)`. Six weights per layer; layer counts 2, 3, 4, 5.

**Protocol.** Train per layer count on the **noiseless** simulator (COBYLA, 512
shots, up to 600 evaluations, 3 random restarts keeping the best), then freeze
the weights and evaluate them under every noise condition at 4096 shots.

The restarts are not decoration. With a single start, 2 layers trained to 0.713
and looked like a capacity limit in the results table; a third restart reached
0.887 from the same architecture. Every restart's final loss is recorded in
`results.json` so the run-to-run spread stays visible rather than being hidden
behind the best result.

Training noiseless and evaluating noisy is deliberate. It isolates the question
being asked — what does a model lose when it meets a noisy device — from the
separate question of whether training can compensate for noise it can observe.
It also matches the realistic deployment case.

## 4. Why not a fixed feature map

The first version of this module used the textbook architecture: `ZZFeatureMap`
for encoding, `RealAmplitudes` as the ansatz, parity readout. It was replaced
after measurement, and the reason is worth recording:

| Architecture | Weights | Test accuracy |
|---|---|---|
| `ZZFeatureMap(reps=2)` + `RealAmplitudes(reps=2)`, parity | 6 | 0.675 |
| `ZZFeatureMap(reps=2)` + `RealAmplitudes(reps=4)`, parity | 10 | 0.675 |
| `ZZFeatureMap(reps=1)` + `RealAmplitudes(reps=2)`, parity | 6 | 0.787 |
| Classical RBF-SVM, identical split | — | **0.963** |

Trained with **exact statevector simulation and no shot noise at all**, so this
is an architectural ceiling, not an optimisation artifact. Doubling the ansatz
parameters left the training loss unchanged to four decimal places — the added
freedom was inert.

This mattered for the experiment, not just for the score. With a noiseless
baseline near 0.70, the accuracy changes induced by noise were around ±0.012 —
exactly one test sample out of 80 — and several were *negative*, meaning the
model appeared to improve under noise. The measurement had no resolution.

Data re-uploading reaches the classical baseline on two qubits, which restores
the headroom the degradation measurement needs.

## 5. Reproducing

```bash
pip install -r requirements.txt
python run_experiment.py
```

Roughly ten minutes on a laptop CPU. Writes `results/results.json`,
`results/results.csv`, and two figures. Everything is seeded — same seed, same
numbers.

```bash
pytest -q          # from the repository root; paths come from pyproject.toml
```

The tests target failure modes that still produce plausible output: swapped
parameter binding, a noise model attached to gate names the transpiler never
emits (and which is therefore silently inert), and the readout bit taken from
the wrong end of the register. Two tests are anchored on analytic values —
identity weights must give `P(q0=1) = 0`, and a unit input scale must give
`sin^2(x/2)`. `test_experiment.py` runs the whole sweep at throwaway fidelity
so a break in the orchestration or figure code cannot reach a commit unnoticed.

Both run in CI on every push, together with `ruff check .`.

## 6. Metrics

**Test accuracy**, and **mean absolute probability shift** relative to the
noiseless run on the same inputs.

The second metric exists because accuracy is thresholded and therefore blunt: a
prediction moving from 0.95 to 0.55 is a large loss of confidence and no change
in accuracy at all. Probability shift registers the erosion while the labels
still look correct — which is the regime a deployed model spends most of its
time in before accuracy visibly breaks.

## 7. Results

Seed 42, 80 test samples, 4096 evaluation shots. Full data in
[`results/results.csv`](./results/results.csv).

### 7.1 Baseline

| Layers | 2q gates | Noiseless accuracy | Under composite device noise | Accuracy drop |
|---|---|---|---|---|
| 2 | 2 | 0.812 | 0.812 | 0.000 |
| 3 | 3 | **0.950** | **0.950** | 0.000 |
| 4 | 4 | 0.887 | 0.887 | 0.000 |
| 5 | 5 | 0.875 | 0.875 | 0.000 |

Three layers essentially match the classical RBF-SVM (0.963) on the same split.
Beyond three, accuracy declines — with 80 training samples the extra parameters
cost more in optimisation difficulty than they return in expressibility.

### 7.2 The main result: accuracy does not see it

At device-like error rates the accuracy drop is **0.000 at every depth**. Taken
alone, that reads as "noise is not a problem here." The probability shift says
otherwise, and it grows with every entangling gate added:

![accuracy vs. probability shift by depth](./results/depth_tradeoff.png)

Pushed to the strongest setting of each mechanism, the split widens further.

Mean |probability shift| versus the noiseless run — monotone in depth for every
gate-based mechanism:

| Mechanism (strongest setting) | L2 | L3 | L4 | L5 |
|---|---|---|---|---|
| Depolarizing, p=0.02 | 0.143 | 0.186 | 0.240 | 0.250 |
| Thermal, T1/T2 ÷ 25 | 0.091 | 0.130 | 0.154 | 0.196 |
| Amplitude damping, γ=0.05 | 0.071 | 0.100 | 0.116 | 0.155 |
| Readout, p=0.1 | 0.067 | 0.068 | 0.073 | 0.067 |

Accuracy drop at the very same settings — near zero, and not monotone in
anything:

| Mechanism (strongest setting) | L2 | L3 | L4 | L5 |
|---|---|---|---|---|
| Depolarizing, p=0.02 | +0.000 | +0.000 | −0.013 | −0.025 |
| Thermal, T1/T2 ÷ 25 | +0.050 | +0.050 | +0.037 | +0.075 |
| Amplitude damping, γ=0.05 | +0.025 | +0.037 | +0.012 | +0.050 |
| Readout, p=0.1 | +0.000 | +0.000 | +0.000 | +0.000 |

Depolarizing noise at p=0.02 moves the output probabilities by 0.25 on average
— a quarter of the available range — while accuracy does not move at all, and
at 4 and 5 layers it nominally *improves*. Those negative drops are one to two
test samples; they are sampling artifacts, not noise acting as a regulariser,
and they illustrate the resolution problem: on 80 samples, accuracy cannot
resolve anything finer than 0.0125.

**A QML acceptance test built on accuracy alone would pass a model whose
decision confidence has already been substantially eroded.**

### 7.3 Readout error behaves differently, and mechanistically it should

Readout is the only mechanism whose probability shift is flat in circuit depth
(0.067 / 0.068 / 0.073 / 0.067). It applies once, at measurement, rather than
accumulating per gate — so depth is irrelevant to it. Every gate-based
mechanism is monotone in depth; readout is not. That the distinction falls out
of the data unprompted is a reasonable check that the noise models attach where
they are supposed to.

Readout also costs the least accuracy — zero, at every depth and every strength
tested up to a 10% flip probability. Symmetric assignment error pulls estimates
toward 0.5 without reordering them relative to the decision threshold.

### 7.4 Per-mechanism sweeps

![accuracy under isolated noise mechanisms](./results/noise_sweeps.png)

Ranked by probability shift at the strongest setting: depolarizing > thermal
relaxation > amplitude damping > readout. The ordering reflects the error
magnitudes chosen in `src/noise_models.py` as much as the mechanisms
themselves, so it should be read as "at these rates", not as an intrinsic
ranking.

Accuracy only breaks visibly once noise is pushed well past realistic levels —
thermal relaxation at a T1/T2 divisor of 25 (T1 = 2 µs) costs 5 to 7.5
percentage points. That is the point at which a metric based on accuracy would
finally raise an alarm, long after the model's outputs stopped being
trustworthy.

## 8. Limitations

- **Two qubits, one dataset.** Nothing here extrapolates to circuit widths
  where noise accumulates across many entangling gates. The direction of the
  depth effect should hold; the magnitudes should not be transferred.
- **Simulated noise, not measured hardware.** Aer noise models are an
  idealisation. Real devices add crosstalk, non-Markovian effects, and
  qubit-to-qubit variation that the all-qubit models used here do not capture.
- **One seed.** Three restarts guard against a single unlucky initialisation,
  but every number here comes from seed 42. Error bars across seeds — for the
  data split as well as the weight initialisation — are the honest next step
  and are not in this version. Differences below roughly 0.03 in accuracy
  should not be read as real.
- **Test-set resolution.** 80 samples means accuracy moves in steps of 0.0125.
  Several reported drops are one or two samples wide.
- **No mitigation applied.** Readout error is classically correctable and gate
  errors respond to zero-noise extrapolation. This module measures the
  unmitigated baseline only.
- **Accuracy on a balanced two-class problem** is a weak metric in general; it
  is adequate here because the split is stratified and the classes are equal.

## 9. What a mitigation would look like

The §7 results reorder the priorities that intuition would suggest.

**Readout correction is the obvious mitigation and the least useful one here.**
A calibration matrix measured once per device and inverted at inference is
cheap and model-agnostic — but readout error was already costing zero accuracy
and the smallest probability shift of any mechanism tested. It would be effort
spent on the mechanism that hurts least.

**Depth reduction is where the leverage is.** Every gate-based mechanism scales
with entangler count, and at 3 layers this model is both the most accurate and
among the least perturbed. Shortening the circuit mitigates all gate-based
mechanisms simultaneously and costs nothing at inference.

**The measurement itself needs to change.** The strongest finding is that
accuracy did not detect degradation that was plainly present. Any monitoring
for a deployed QML model should track output distributions against a noiseless
reference, not just label agreement — otherwise the first visible signal
arrives long after the outputs stopped being trustworthy.

None of these are implemented here. Measuring the unmitigated baseline comes
first, or there is nothing to compare a mitigation against.
