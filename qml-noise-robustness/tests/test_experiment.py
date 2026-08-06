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

from src import experiment
from src.noise_models import depolarizing, device_like, ideal
from src.plots import plot_depth_tradeoff, plot_noise_sweeps


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


def test_written_results_are_order_independent(tmp_path, fast_sweep):
    """Workers finish in nondeterministic order; the files must not."""
    first = experiment.run(tmp_path / "a", seeds=SEEDS, workers=1, verbose=False)
    second = experiment.run(tmp_path / "b", seeds=list(reversed(SEEDS)), workers=1, verbose=False)
    assert [r["seed"] for r in first["per_seed"]["training"]] == \
           [r["seed"] for r in second["per_seed"]["training"]]
    assert (tmp_path / "a" / "results.csv").read_text() == \
           (tmp_path / "b" / "results.csv").read_text()


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
    sweeps = plot_noise_sweeps(payload, tmp_path / "noise_sweeps.png")
    tradeoff = plot_depth_tradeoff(payload, tmp_path / "depth_tradeoff.png")
    assert sweeps.exists() and sweeps.stat().st_size > 0
    assert tradeoff.exists() and tradeoff.stat().st_size > 0


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
