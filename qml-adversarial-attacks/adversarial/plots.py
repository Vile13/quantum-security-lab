"""Result plots for the attack sweep."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # no display in CI or over SSH
import matplotlib.pyplot as plt

MODEL_STYLE = {
    "quantum": ("tab:blue", "VQC (noiseless)"),
    "quantum_noisy": ("tab:purple", "VQC (device noise)"),
    "classical": ("tab:green", "RBF-SVM"),
}
ATTACK_STYLE = {
    "random": (":", "o", "random (control)"),
    "fgsm": ("--", "s", "FGSM"),
    "pgd": ("-", "^", "PGD"),
}


def _series(payload: dict, model: str, attack: str, field: str = "flip_rate"):
    rows = [r for r in payload["aggregate"] if r["model"] == model and r["attack"] == attack]
    rows.sort(key=lambda r: r["epsilon"])
    return (
        [r["epsilon"] for r in rows],
        [r[field]["mean"] for r in rows],
        [r[field]["std"] for r in rows],
    )


def plot_flip_rates(payload: dict, output_path: Path) -> Path:
    """One panel per model; the random control is on every panel deliberately.

    Keeping the control in the same panel as the attacks it validates means a
    reader cannot look at an attack curve without also seeing what noise of the
    same magnitude does.
    """
    models = [m for m in MODEL_STYLE if any(r["model"] == m for r in payload["aggregate"])]
    n_seeds = len(payload["config"]["seeds"])

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, model in zip(axes, models, strict=True):
        colour, title = MODEL_STYLE[model]
        for attack, (linestyle, marker, legend) in ATTACK_STYLE.items():
            epsilons, means, errors = _series(payload, model, attack)
            if not epsilons:
                continue
            ax.errorbar(epsilons, means, yerr=errors, linestyle=linestyle, marker=marker,
                        color=colour, alpha=1.0 if attack == "pgd" else 0.65,
                        capsize=3, markersize=5, label=legend)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("epsilon (radians, L-inf)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("flip rate among initially correct samples")
    axes[0].set_ylim(-0.02, 1.02)

    fig.suptitle(
        "Adversarial flip rate vs. perturbation budget\n"
        f"error bars = 1 SD over {n_seeds} seeds;  inputs are encoded into [0, pi]",
        fontsize=11,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_model_comparison(payload: dict, output_path: Path) -> Path:
    """The strongest attack (PGD) on every model, in one panel."""
    n_seeds = len(payload["config"]["seeds"])
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, (colour, legend) in MODEL_STYLE.items():
        epsilons, means, errors = _series(payload, model, "pgd")
        if not epsilons:
            continue
        ax.errorbar(epsilons, means, yerr=errors, marker="^", color=colour,
                    capsize=3, linewidth=2, label=f"{legend} — PGD")
        epsilons, means, _ = _series(payload, model, "random")
        ax.plot(epsilons, means, linestyle=":", color=colour, alpha=0.5,
                label=f"{legend} — random")

    ax.set_xlabel("epsilon (radians, L-inf)")
    ax.set_ylabel("flip rate among initially correct samples")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.set_title(
        "Strongest attack per model, against its own random control\n"
        f"error bars = 1 SD over {n_seeds} seeds", fontsize=11,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
