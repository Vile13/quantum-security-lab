# quantum-security-lab

[![CI](https://github.com/Vile13/quantum-security-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Vile13/quantum-security-lab/actions/workflows/ci.yml)

Research and engineering lab investigating the **security and robustness properties of quantum and hybrid quantum-classical machine learning systems**.

Part of a broader portfolio at the intersection of AI security, agent/tool security, and post-quantum cryptography.

## Motivation

Quantum machine learning (QML) models are increasingly proposed for real-world deployment on noisy intermediate-scale quantum (NISQ) hardware. Unlike classical ML security, QML introduces attack surfaces that are specific to the quantum stack: device noise, transpilation, backend heterogeneity, and the encoding/measurement boundary between classical and quantum data. This lab treats these as **security and reliability properties to be measured, not assumed.**

Each module follows the same structure:
1. **Research question** — what property are we measuring?
2. **Threat / failure model** — what could go wrong, and why does it matter operationally?
3. **Experiment** — reproducible code, fixed seeds, documented parameters
4. **Results** — metrics, tables, plots
5. **Discussion** — limitations, what a mitigation would look like

## Modules

| Module | Status | Question |
|---|---|---|
| [`qml-noise-robustness`](./qml-noise-robustness) | ✅ v1 results | How does classification accuracy degrade under realistic Qiskit Aer noise models (depolarizing, amplitude damping, thermal relaxation, readout error), and does circuit depth trade off expressibility against noise resilience? |
| `qml-adversarial-attacks` | 📋 planned | Can small, targeted perturbations to classical input data or encoded quantum states flip QML classifier decisions, and how does this compare to classical adversarial robustness? |
| `circuit-parameter-tampering` | 📋 planned (roadmap) | What happens if variational parameters are tampered with post-training (supply-chain analogy for quantum models)? |
| `quantum-artificial-life` | 💡 idea (roadmap) | Emergent behavior in evolutionary/quantum artificial life systems — exploratory. |

## Tooling

- **Qiskit + Qiskit Aer** for circuit construction, simulation, and noise modeling
- **scikit-learn / NumPy / SciPy** for classical data handling and optimization
- **matplotlib** for result visualization

## Responsible research statement

All experiments run against local simulators or explicitly self-owned test setups. No experiments target third-party systems, production infrastructure, or real quantum hardware without authorization. This lab is for research, benchmarking, and portfolio demonstration purposes.

## Repository layout

```
quantum-security-lab/
├── qml-noise-robustness/     # module 1 — see its own README
│   ├── src/                  #   data, model, noise models, sweep, plots
│   ├── tests/                #   invariants that would otherwise fail silently
│   ├── results/              #   committed results and figures
│   └── run_experiment.py     #   entry point
└── README.md                 # this file
```

## Findings so far

**`qml-noise-robustness`** (8 seeds, error bars throughout) — at device-like
error rates, test accuracy showed **no degradation at any circuit depth**
(+0.003 ± 0.011 at worst), while the model's output probabilities shifted
measurably, and that shift grew with every entangling gate added — strictly, in
8 of 8 seeds. Readout error was the control and behaved like one: flat in depth
(1 of 8 seeds, about what chance gives), because it applies once at measurement
rather than accumulating per gate. The practical consequence is that an
acceptance test built on accuracy alone would pass a model whose decision
confidence has already eroded substantially.

Running it across seeds also **retracted a v1 claim**: the apparent optimum at
three layers was one draw from a wide distribution, and depth differences in
accuracy turned out not to be resolvable at this sample size at all.
[Details, method and limitations →](./qml-noise-robustness)

## Roadmap

- [x] `qml-noise-robustness` — baseline model, noise sweep, results writeup
- [x] CI (ruff lint + pytest on Python 3.10 and 3.12) via GitHub Actions
- [x] `qml-noise-robustness` v2 — 8-seed sweep with error bars, depth-scaling tested across seeds
- [ ] `qml-noise-robustness` v3 — mitigation comparison (readout calibration, zero-noise extrapolation)
- [ ] `qml-adversarial-attacks` — perturbation attacks on encoded inputs, comparison to classical adversarial robustness
- [ ] Architecture diagram + short demo
- [ ] CI (linting, unit tests) via GitHub Actions
- [ ] `circuit-parameter-tampering`
- [ ] `quantum-artificial-life`

## License

Licensed under [Apache License 2.0](./LICENSE).

## Citation

See [`CITATION.cff`](./CITATION.cff). GitHub renders a "Cite this repository" button from it automatically.

## About

Maintained as part of an ongoing research and security engineering portfolio at the intersection of AI security, quantum software security, and post-quantum cryptography.
