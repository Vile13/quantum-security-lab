"""End-to-end smoke test for the sweep pipeline.

The unit tests in ``test_model.py`` never touch ``experiment.py`` or
``plots.py``, so a break in the orchestration or figure code would reach a
commit unnoticed. This runs the whole pipeline at throwaway fidelity -- the
numbers are meaningless, only the wiring is under test.
"""

from __future__ import annotations

import json

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


def test_full_sweep_runs_and_writes_expected_artifacts(tmp_path, fast_sweep):
    payload = experiment.run(tmp_path, seed=1, verbose=False)

    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "results.csv").exists()

    on_disk = json.loads((tmp_path / "results.json").read_text())
    assert on_disk["config"]["layer_counts"] == [2, 3]
    assert len(on_disk["training"]) == 2
    assert len(on_disk["evaluation"]) == 2 * 4  # layer counts x conditions

    # The CSV must carry a header plus one row per evaluation, or the committed
    # results file silently loses rows.
    lines = (tmp_path / "results.csv").read_text().strip().splitlines()
    assert len(lines) == 1 + len(on_disk["evaluation"])

    for row in payload["evaluation"]:
        assert 0.0 <= row["test_accuracy"] <= 1.0
        assert 0.0 <= row["mean_abs_proba_shift"] <= 1.0


def test_ideal_condition_is_its_own_baseline(tmp_path, fast_sweep):
    """The noiseless row is the reference the drops are measured against, so it
    must report a zero drop and a zero shift by construction."""
    payload = experiment.run(tmp_path, seed=1, verbose=False)
    for row in payload["evaluation"]:
        if row["mechanism"] == "none":
            assert row["accuracy_drop"] == 0.0
            assert row["mean_abs_proba_shift"] == 0.0


def test_every_restart_loss_is_recorded(tmp_path, fast_sweep):
    payload = experiment.run(tmp_path, seed=1, verbose=False)
    for training in payload["training"]:
        assert len(training["restart_losses"]) == 2
        assert training["final_loss"] == pytest.approx(min(training["restart_losses"]))


def test_figures_are_produced_from_the_payload(tmp_path, fast_sweep):
    payload = experiment.run(tmp_path, seed=1, verbose=False)
    sweeps = plot_noise_sweeps(payload, tmp_path / "noise_sweeps.png")
    tradeoff = plot_depth_tradeoff(payload, tmp_path / "depth_tradeoff.png")
    assert sweeps.exists() and sweeps.stat().st_size > 0
    assert tradeoff.exists() and tradeoff.stat().st_size > 0
