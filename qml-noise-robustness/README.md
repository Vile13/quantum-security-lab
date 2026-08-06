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
the weights and evaluate them under every noise condition at 4096 shots. The
whole sweep is repeated over **8 seeds** (42–49) and every reported number is a
mean with a spread across them.

The restarts are not decoration. With a single start, 2 layers trained to 0.713
and looked like a capacity limit in the results table; a third restart reached
0.887 from the same architecture. Every restart's final loss is recorded in
`results.json` so the run-to-run spread stays visible rather than being hidden
behind the best result.

**What a seed controls.** The dataset and its split, the weight initialisation
of every restart, and the sampler's shot noise — all three at once. That is
deliberate: the resulting spread is what a reader would see on re-running the
experiment, which is what an error bar should mean. It does mean the error bars
on *absolute* accuracy are dominated by which 80 points landed in the test
split, and are correspondingly wide.

**Why the degradation resolves more sharply than the accuracy.** Each noise
condition is compared against the noiseless run **of its own seed**, so
`accuracy_drop` is a paired difference and the split variance cancels. A drop
can therefore be resolved to a precision the absolute accuracies never reach.
Wide error bars on accuracy next to tight ones on drop are not inconsistent —
they are the point of pairing.

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

Roughly 15 minutes on a 4-core laptop; `--workers 1` runs it serially in about
an hour, `--seeds 42 43` cuts it down for a quick look. Writes
`results/results.json`, `results/results.csv`, and two figures. Everything is
seeded and the output is sorted before writing, so repeated runs with the same
seeds produce byte-identical files regardless of worker scheduling.

The entry point pins every numeric library to a single thread before importing
them. That is a speedup, not a restriction: on two-qubit circuits Aer's internal
threading costs more than it saves (measured 118 ms vs 85 ms per objective
evaluation), and single-threaded workers then parallelise cleanly across seeds
rather than competing for the same cores.

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

8 seeds (42–49), 80 test samples, 4096 evaluation shots. All values are mean ±
1 SD across seeds unless stated otherwise. Full per-seed data in
[`results/results.json`](./results/results.json), aggregates in
[`results/results.csv`](./results/results.csv).

### 7.1 Baseline, and a claim v1 got wrong

| Layers | 2q gates | Noiseless accuracy | SEM | Range across seeds |
|---|---|---|---|---|
| 2 | 2 | 0.853 ± 0.042 | 0.015 | 0.787 – 0.900 |
| 3 | 3 | 0.906 ± 0.050 | 0.018 | 0.800 – 0.963 |
| 4 | 4 | 0.889 ± 0.031 | 0.011 | 0.850 – 0.938 |
| 5 | 5 | 0.886 ± 0.069 | 0.024 | 0.787 – 0.988 |

**The single-seed version of this module reported that three layers was the
optimum. That claim does not survive.** The gap between 3 and 4 layers is 0.017
against standard errors of 0.018 and 0.011, and the per-seed ranges overlap
almost completely — five layers produced both the worst run (0.787) and the
best (0.988) in the whole sweep. Beyond two layers, this experiment cannot
resolve a depth ranking in accuracy at all, and the v1 number was one draw from
a wide distribution being read as a result.

Two layers is the one distinction that survives: it is consistently the weakest,
which is a capacity statement rather than a tuning artifact.

### 7.2 The main result, now with error bars

At device-like error rates the accuracy drop is indistinguishable from zero at
every depth — and the paired construction (§3) makes that a tight statement,
not a vague one:

| Layers | Accuracy drop under composite noise | Mean absolute probability shift |
|---|---|---|
| 2 | +0.0016 ± 0.0080 | 0.027 ± 0.001 |
| 3 | +0.0000 ± 0.0000 | 0.037 ± 0.002 |
| 4 | +0.0016 ± 0.0124 | 0.043 ± 0.002 |
| 5 | +0.0031 ± 0.0111 | 0.049 ± 0.003 |

![accuracy vs. probability shift by depth](./results/depth_tradeoff.png)

The figure states the finding on its own: overlapping error bars on accuracy,
non-overlapping ones on probability shift. Note how much tighter the shift
error bars are (SD ≈ 0.002) than those on absolute accuracy (SD ≈ 0.05) —
probability shift is measured against a per-seed reference and is not exposed
to the split variance that dominates the accuracy column.

Pushed to the strongest setting of each mechanism, the gap widens:

| Mechanism (strongest setting) | L2 | L3 | L4 | L5 |
|---|---|---|---|---|
| Depolarizing, p=0.02 | 0.135 ± 0.004 | 0.197 ± 0.008 | 0.230 ± 0.011 | 0.263 ± 0.012 |
| Thermal, T1/T2 ÷ 25 | 0.115 ± 0.015 | 0.152 ± 0.011 | 0.194 ± 0.027 | 0.210 ± 0.024 |
| Amplitude damping, γ=0.05 | 0.090 ± 0.012 | 0.114 ± 0.009 | 0.150 ± 0.024 | 0.162 ± 0.021 |
| Readout, p=0.1 | 0.064 ± 0.002 | 0.071 ± 0.002 | 0.069 ± 0.003 | 0.070 ± 0.003 |

Depolarizing noise at p=0.02 moves output probabilities by 0.26 at five layers
— a quarter of the available range — for an accuracy change of −0.003 ± 0.015.

**An acceptance test built on accuracy alone would pass a model whose decision
confidence has already eroded substantially.** That was v1's claim from one
seed; across eight it holds with error bars that do not come close to zero.

### 7.3 Depth-scaling, tested rather than observed

v1 observed that probability shift grows with circuit depth. With 8 seeds that
can be checked instead of asserted: in how many seeds does the shift increase
strictly with *every* added layer?

| Mechanism | Strictly increasing in |
|---|---|
| Depolarizing | **8 / 8** |
| Composite device model | **8 / 8** |
| Thermal relaxation | 6 / 8 |
| Amplitude damping | 6 / 8 |
| Readout | **1 / 8** |

Readout is the control, and it behaves like one. It applies once at
measurement rather than accumulating per gate, so it has no reason to scale
with depth — and 1 of 8 is about what four values in random order would give
(chance is 1/4! ≈ 4%). Every gate-based mechanism separates clearly from it.

That contrast is the strongest evidence here that the noise models attach where
they are meant to: nothing in the setup forces readout to behave differently,
and it does, in the direction the mechanism predicts.

### 7.4 Where accuracy finally does break

One mechanism costs real accuracy once pushed past realistic levels — thermal
relaxation at a T1/T2 divisor of 25 (T1 = 2 µs):

| Layers | Accuracy drop |
|---|---|
| 2 | +0.031 ± 0.037 |
| 3 | +0.008 ± 0.031 |
| 4 | +0.089 ± 0.084 |
| 5 | +0.158 ± 0.132 |

Five layers lose nearly 16 percentage points. The standard deviation is
enormous (0.132) — some seeds collapse to chance while others barely move — so
the honest reading is that deep circuits under severe decoherence become
*unreliable* rather than uniformly worse. For a deployed model, high variance
across otherwise identical runs is its own failure mode.

![accuracy under isolated noise mechanisms](./results/noise_sweeps.png)

Ranked by probability shift at the strongest setting: depolarizing > thermal
relaxation > amplitude damping > readout. The ordering reflects the error
magnitudes chosen in `src/noise_models.py` as much as the mechanisms
themselves, so read it as "at these rates", not as an intrinsic ranking.

## 8. Limitations

- **Two qubits, one dataset.** Nothing here extrapolates to circuit widths
  where noise accumulates across many entangling gates. The direction of the
  depth effect should hold; the magnitudes should not be transferred.
- **Simulated noise, not measured hardware.** Aer noise models are an
  idealisation. Real devices add crosstalk, non-Markovian effects, and
  qubit-to-qubit variation that the all-qubit models used here do not capture.
- **Eight seeds is a small sample.** Enough for a standard deviation that is
  informative rather than misleading, not enough for a confident claim about
  the shape of the distribution. The error bars should be read as "roughly this
  wide", and no significance test is claimed anywhere in §7.
- **Test-set resolution.** 80 samples means accuracy moves in steps of 0.0125,
  so a single-sample difference is not a finding. Pairing within seeds is what
  makes the drops resolvable at all.
- **Seeds are not independent across the layer axis.** Within one seed, all
  four layer counts share the same train/test split, which is what makes the
  depth comparison paired and sensitive — but it also means the four curves in
  a figure are correlated, not independent samples.
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
with entangler count in 8 of 8 seeds, and the accuracy cost of using fewer
layers is — beyond two — too small for this experiment to resolve at all
(§7.1). Shortening the circuit therefore mitigates every gate-based mechanism
at once, for a price the measurement cannot even detect. That asymmetry, not a
specific optimal depth, is the actionable result.

**The measurement itself needs to change.** The strongest finding is that
accuracy did not detect degradation that was plainly present. Any monitoring
for a deployed QML model should track output distributions against a noiseless
reference, not just label agreement — otherwise the first visible signal
arrives long after the outputs stopped being trustworthy.

None of these are implemented here. Measuring the unmitigated baseline comes
first, or there is nothing to compare a mitigation against.
