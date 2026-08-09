"""Smoke test for the demo script.

The demo is the first thing a reader runs, so it breaking is the most visible
possible failure -- and it imports across both modules and the shared package,
which is exactly the wiring a refactor tends to disturb. Running it in
``--quick`` mode takes about ten seconds and touches every one of those paths.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_demo_runs_and_reports_all_three_findings():
    result = subprocess.run(
        [sys.executable, str(ROOT / "demo.py"), "--quick"],
        capture_output=True, text=True, timeout=600, cwd=ROOT,
    )
    assert result.returncode == 0, f"demo exited {result.returncode}\n{result.stderr[-2000:]}"

    output = result.stdout
    for heading in (
        "Accuracy does not see what noise does to the outputs",
        "Each mitigation corrects its own mechanism",
        "A gradient attack beats random noise of the same size",
    ):
        assert heading in output, f"missing section: {heading}"

    # Each section must print numbers, not just prose.
    for label in ("mean |probability shift|", "readout+zne", "epsilon"):
        assert label in output, f"missing output: {label}"


def test_demo_states_that_its_numbers_are_reduced_fidelity():
    """The demo trades accuracy for speed, and must not be mistakable for the
    8-seed measurement it illustrates."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "demo.py"), "--quick"],
        capture_output=True, text=True, timeout=600, cwd=ROOT,
    )
    assert "reduced fidelity" in result.stdout.lower()
    assert "README" in result.stdout
