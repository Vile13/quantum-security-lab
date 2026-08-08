"""End-to-end smoke test for the sweep pipeline, plus the aggregation logic.

The unit tests in ``test_model.py`` never touch ``experiment.py`` or
``plots.py``, so a break in the orchestration or figure code would reach a
commit unnoticed. The sweep here runs at throwaway fidelity -- the accuracies
are meaningless, only the wiring is under test. The aggregation tests are the
opposite: they use hand-built inputs with known statistics, because an error
bar computed wrongly is invisible in a plot.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from noise_robustness import experiment
from noise_robustness.plots import plot_depth_tradeoff, plot_mitigation, plot_noise_sweeps

from qml_lab.noise_models import depolarizing, device_like, ideal, readout


@pytest.fixture
def fast_sweep(monkeypatch):
    """Shrink every cost knob so the pipeline runs in seconds, not minutes."""
    monkeypatch.setattr(experiment, "TRAIN_SHOTS", 64)
    monkeypatch.setattr(experiment, "EVAL_SHOTS", 128)
    monkeypatch.setattr(experiment, "MAX_ITERATIONS", 6)
    monkeypatch.setattr(experiment, "N_RESTARTS", 2)
    monkeypatch.setattr(experiment, "LAYER_COUNTS", [2, 3])
    # A representative condition per code path: the ideal reference, one swept
    # mechanism (so the sweep plots have something to draw), and the composite.
    monkeypatch.setattr(
        experiment, "all_conditions",
        lambda: [ideal(), depolarizing(0.001), depolarizing(0.01), device_like()],
    )
    # The mitigation stage runs three fold scales per condition, so the full
    # four-condition set dominates the suite's runtime for no extra coverage.
    monkeypatch.setattr(experiment, "MITIGATION_CONDITIONS", [device_like(), readout(0.1)])
    monkeypatch.setattr(experiment, "CALIBRATION_SHOTS", 256)


SEEDS = [1, 2]


def test_full_sweep_runs_and_writes_expected_artifacts(tmp_path, fast_sweep):
    payload = experiment.run(tmp_path, seeds=SEEDS, workers=1, verbose=False)

    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "results.csv").exists()

    on_disk = json.loads((tmp_path / "results.json").read_text())
    assert on_disk["config"]["seeds"] == SEEDS
    assert len(on_disk["per_seed"]["training"]) == len(SEEDS) * 2
    assert len(on_disk["per_seed"]["evaluation"]) == len(SEEDS) * 2 * 4
    # Aggregation collapses the seed axis and nothing else.
    assert len(on_disk["aggregate"]["evaluation"]) == 2 * 4

    lines = (tmp_path / "results.csv").read_text().strip().splitlines()
    assert len(lines) == 1 + len(on_disk["aggregate"]["evaluation"])

    for row in payload["aggregate"]["evaluation"]:
        assert row["test_accuracy"]["n"] == len(SEEDS)
        assert 0.0 <= row["test_accuracy"]["mean"] <= 1.0
        assert row["test_accuracy"]["std"] >= 0.0


def test_written_results_are_ordered_independently_of_seed_order(tmp_path, fast_sweep):
    """Workers finish in nondeterministic order; the row ordering must not.

    Only the ordering is asserted, not the values. Training is not bit-exact
    between runs (see README §5.1), so comparing the file contents here would
    produce a test that fails a few times a year for reasons unrelated to the
    property being checked.
    """
    first = experiment.run(tmp_path / "a", seeds=SEEDS, workers=1, verbose=False)
    second = experiment.run(tmp_path / "b", seeds=list(reversed(SEEDS)), workers=1, verbose=False)
    for section, keys in (
        ("training", ("seed", "layers")),
        ("evaluation", ("seed", "layers", "mechanism", "strength")),
        ("mitigation", ("seed", "layers", "label", "strategy")),
    ):
        assert [tuple(r[k] for k in keys) for r in first["per_seed"][section]] == \
               [tuple(r[k] for k in keys) for r in second["per_seed"][section]]


def test_ideal_condition_is_its_own_baseline(tmp_path, fast_sweep):
    """The noiseless row is the reference the drops are measured against, so it
    must report a zero drop and a zero shift in every seed."""
    payload = experiment.run(tmp_path, seeds=SEEDS, workers=1, verbose=False)
    for row in payload["per_seed"]["evaluation"]:
        if row["mechanism"] == "none":
            assert row["accuracy_drop"] == 0.0
            assert row["mean_abs_proba_shift"] == 0.0


def test_every_restart_loss_is_recorded(tmp_path, fast_sweep):
    payload = experiment.run(tmp_path, seeds=SEEDS, workers=1, verbose=False)
    for training in payload["per_seed"]["training"]:
        assert len(training["restart_losses"]) == 2
        assert training["final_loss"] == pytest.approx(min(training["restart_losses"]))


def test_figures_are_produced_from_the_payload(tmp_path, fast_sweep):
    payload = experiment.run(tmp_path, seeds=SEEDS, workers=1, verbose=False)
    figures = [
        plot_noise_sweeps(payload, tmp_path / "noise_sweeps.png"),
        plot_depth_tradeoff(payload, tmp_path / "depth_tradeoff.png"),
        plot_mitigation(payload, tmp_path / "mitigation.png"),
    ]
    for figure in figures:
        assert figure.exists() and figure.stat().st_size > 0


def test_mitigation_stage_is_recorded_and_aggregated(tmp_path, fast_sweep):
    payload = experiment.run(tmp_path, seeds=SEEDS, workers=1, verbose=False)
    per_seed = payload["per_seed"]["mitigation"]
    # seeds x layer counts x conditions x strategies
    assert len(per_seed) == len(SEEDS) * 2 * 2 * 4
    # Aggregation pools the seed and layer axes, leaving conditions x strategies.
    assert len(payload["mitigation_aggregate"]) == 2 * 4
    assert (tmp_path / "mitigation.csv").exists()

    for row in payload["mitigation_aggregate"]:
        if row["strategy"] == "none":
            # "none" is the reference the reduction is measured against.
            assert row["shift_reduction"]["mean"] == 0.0
        assert row["mean_abs_proba_shift"]["mean"] >= 0.0


def test_mitigation_reduction_is_consistent_with_the_recorded_shifts(tmp_path, fast_sweep):
    """shift_reduction must be derivable from the shifts it claims to summarise."""
    payload = experiment.run(tmp_path, seeds=SEEDS, workers=1, verbose=False)
    rows = payload["per_seed"]["mitigation"]
    for row in rows:
        reference = next(
            r for r in rows
            if r["seed"] == row["seed"] and r["layers"] == row["layers"]
            and r["label"] == row["label"] and r["strategy"] == "none"
        )
        base = reference["mean_abs_proba_shift"]
        expected = (base - row["mean_abs_proba_shift"]) / base if base > 0 else 0.0
        assert row["shift_reduction"] == pytest.approx(expected)


# --- aggregation statistics, checked against known values -------------------


def test_summarise_matches_numpy_sample_statistics():
    values = [0.1, 0.5, 0.3, 0.9]
    summary = experiment._summarise(values)
    assert summary["mean"] == pytest.approx(np.mean(values))
    # ddof=1: these seeds are a sample, not the population.
    assert summary["std"] == pytest.approx(np.std(values, ddof=1))
    assert summary["sem"] == pytest.approx(np.std(values, ddof=1) / np.sqrt(len(values)))
    assert summary["min"] == 0.1
    assert summary["max"] == 0.9
    assert summary["n"] == 4


def test_summarise_reports_zero_spread_for_a_single_value():
    """One seed must not produce a NaN error bar -- ddof=1 on n=1 divides by zero."""
    summary = experiment._summarise([0.7])
    assert summary["mean"] == 0.7
    assert summary["std"] == 0.0
    assert summary["sem"] == 0.0
    assert summary["n"] == 1


def test_depth_monotonicity_counts_strictly_increasing_seeds(monkeypatch):
    monkeypatch.setattr(experiment, "LAYER_COUNTS", [2, 3, 4])
    rows = []
    # Seed 1 increases with depth, seed 2 does not.
    for seed, shifts in ((1, [0.10, 0.20, 0.30]), (2, [0.10, 0.30, 0.20])):
        for layers, shift in zip([2, 3, 4], shifts, strict=True):
            rows.append({
                "seed": seed, "layers": layers, "mechanism": "depolarizing",
                "strength": 0.02, "mean_abs_proba_shift": shift,
            })
    result = experiment.depth_monotonicity(rows, [1, 2])
    assert result["depolarizing"] == {"strictly_increasing_in": 1, "of_seeds": 2}
