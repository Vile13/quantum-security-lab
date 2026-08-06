"""Result plots.

Matplotlib defaults are overridden only where the default actively obscures the
result -- notably the y-axis, which is pinned to include the 0.5 chance line so
that a drop cannot be made to look dramatic or negligible by autoscaling.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display in CI or over SSH
import matplotlib.pyplot as plt  # noqa: E402

CHANCE_LEVEL = 0.5
MECHANISM_TITLES = {
    "depolarizing": "Depolarizing (1q error rate)",
    "amplitude_damping": "Amplitude damping (gamma)",
    "thermal_relaxation": "Thermal relaxation (T1/T2 divisor)",
    "readout": "Readout assignment error",
}


def _rows_for(payload: dict, mechanism: str, layers: int) -> tuple[list[float], list[float]]:
    rows = [
        r for r in payload["evaluation"]
        if r["mechanism"] == mechanism and r["layers"] == layers
    ]
    rows.sort(key=lambda r: r["strength"])
    return [r["strength"] for r in rows], [r["test_accuracy"] for r in rows]


def plot_noise_sweeps(payload: dict, output_path: Path) -> Path:
    """One panel per mechanism, one line per ansatz depth."""
    mechanisms = list(MECHANISM_TITLES)
    depths = payload["config"]["layer_counts"]
    ideal_by_depth = {
        r["layers"]: r["test_accuracy"]
        for r in payload["evaluation"] if r["mechanism"] == "none"
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    for ax, mechanism in zip(axes.flat, mechanisms):
        for depth in depths:
            strengths, accuracies = _rows_for(payload, mechanism, depth)
            ax.plot(strengths, accuracies, marker="o", label=f"{depth} re-uploading layers")
        ax.axhline(CHANCE_LEVEL, color="grey", linestyle=":", linewidth=1)
        ax.set_title(MECHANISM_TITLES[mechanism], fontsize=10)
        ax.set_xscale("log")
        ax.set_xlabel("noise strength")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("test accuracy")
    axes[1][0].set_ylabel("test accuracy")
    axes[0][0].set_ylim(0.35, 1.02)
    axes[0][0].legend(fontsize=8)

    subtitle = ", ".join(f"{d}L: {ideal_by_depth.get(d, float('nan')):.3f}" for d in depths)
    fig.suptitle(
        "VQC accuracy under isolated noise mechanisms\n"
        f"noiseless reference -- {subtitle};  dotted line = chance level",
        fontsize=11,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_depth_tradeoff(payload: dict, output_path: Path) -> Path:
    """Accuracy and confidence erosion under the composite model, by depth.

    Both are plotted because they disagree, and the disagreement is the result:
    at device-like error rates accuracy reports no loss at any depth, while the
    probability shift grows with every added entangling gate.
    """
    depths = payload["config"]["layer_counts"]
    ideal, composite, shift = [], [], []
    for depth in depths:
        rows = [r for r in payload["evaluation"] if r["layers"] == depth]
        ideal.append(next(r["test_accuracy"] for r in rows if r["mechanism"] == "none"))
        composite_row = next(r for r in rows if r["mechanism"] == "composite")
        composite.append(composite_row["test_accuracy"])
        shift.append(composite_row["mean_abs_proba_shift"])

    two_qubit_gates = {t["layers"]: t["two_qubit_gates"] for t in payload["training"]}
    labels = [f"{d} layers\n({two_qubit_gates.get(d, '?')} 2q gates)" for d in depths]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = list(range(len(depths)))
    # Drawn wide-and-translucent underneath a thin dashed overlay: the two are
    # identical at every depth, and a reader has to be able to see that rather
    # than assume one series failed to plot.
    ax.plot(x, ideal, marker="o", color="tab:blue", linewidth=5, alpha=0.35,
            label="accuracy, noiseless")
    ax.plot(x, composite, marker="s", color="tab:green", linestyle="--", linewidth=1.5,
            markersize=5, label="accuracy, device-like noise")
    ax.axhline(CHANCE_LEVEL, color="grey", linestyle=":", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0.35, 1.02)
    ax.grid(alpha=0.3)

    twin = ax.twinx()
    twin.bar(x, shift, width=0.35, color="tab:red", alpha=0.25, zorder=0,
             label="mean |probability shift|")
    twin.set_ylabel("mean |probability shift| vs. noiseless", color="tab:red")
    twin.tick_params(axis="y", labelcolor="tab:red")
    twin.set_ylim(0, max(shift) * 2.2 if shift else 1)

    handles, texts = ax.get_legend_handles_labels()
    twin_handles, twin_texts = twin.get_legend_handles_labels()
    ax.legend(handles + twin_handles, texts + twin_texts, fontsize=8, loc="lower right")
    ax.set_title("Accuracy hides what the probabilities are already doing")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
