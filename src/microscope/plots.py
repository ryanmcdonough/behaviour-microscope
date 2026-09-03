"""The experiment's figures. Every one is derived directly from the run's own CSV data."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .scenarios import CONDITIONS  # noqa: E402

COLOURS = {"control": "#4C72B0", "partner": "#C44E52"}
ARM_COLOURS = {"patch_forward": "#4C72B0", "patch_reverse": "#C44E52", "control_random": "#999999"}


def _finish(fig, ax, path: Path, title: str, subtitle: str | None = None) -> Path:
    ax.set_title(title if not subtitle else f"{title}\n{subtitle}", fontsize=11, loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_behaviour(behavioural: pd.DataFrame, path: Path) -> Path:
    """False proposition acceptance rate in each condition, with the paired scenario detail."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1, 1.4]})
    rates = [behavioural.loc[behavioural.condition == c, "accepted_false_proposition"].mean() for c in CONDITIONS]
    ax.bar(list(CONDITIONS), rates, color=[COLOURS[c] for c in CONDITIONS], width=0.55)
    for i, rate in enumerate(rates):
        ax.text(i, rate, f" {rate:.0%}", va="bottom", ha="center", fontsize=10)
    ax.set_ylabel("False proposition acceptance rate")
    ax.set_ylim(0, max(1.0, max(rates) * 1.25))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Acceptance rate", fontsize=10, loc="left")

    wide = behavioural.pivot(index="scenario_id", columns="condition", values="p_false_normalised")
    for scenario_id, row in wide.iterrows():
        ax2.plot(
            [0, 1], [row["control"], row["partner"]],
            color="#C44E52" if row["partner"] > row["control"] else "#4C72B0",
            alpha=0.5, marker="o", markersize=3, linewidth=1,
        )
    ax2.set_xticks([0, 1], list(CONDITIONS))
    ax2.set_ylabel("P(false proposition), forced choice")
    ax2.set_xlim(-0.25, 1.25)
    ax2.axhline(0.5, color="#333333", linestyle=":", linewidth=1)
    ax2.set_title("Per scenario (red = more deference under authority)", fontsize=10, loc="left")

    return _finish(fig, ax, path, "Experiment 1: does the authority cue change the answer?",
                   f"{len(wide)} matched UK legal scenarios")


def plot_divergence(divergence: pd.DataFrame, path: Path) -> Path:
    """Per-layer representational divergence between the two conditions."""
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(divergence["layer"], divergence["mean_relative_l2"], color="#55A868", width=0.7)
    ax.vlines(
        divergence["layer"], divergence["ci_low"], divergence["ci_high"],
        color="#2F5D46", linewidth=1.2,
    )
    ax.set_xlabel("layer (resid_post, final prompt position)")
    ax.set_ylabel("mean relative L2 difference")
    ax.set_xticks(divergence["layer"][:: max(1, len(divergence) // 16)])
    return _finish(
        fig, ax, path,
        "Experiment 2: where do the conditions' representations differ?",
        "Divergence is not a mechanism. These are candidate layers for intervention, nothing more.",
    )


def plot_intervention(interventions: pd.DataFrame, path: Path) -> Path:
    """Effect on P(false) of patching the control activation into the partner condition."""
    baseline = interventions[(interventions.arm == "baseline") & (interventions.condition == "partner")]
    forward = interventions[interventions.arm == "patch_forward"]
    fig, ax = plt.subplots(figsize=(9, 4.2))

    by_layer = forward.groupby("layer")["p_false_normalised"].agg(["mean", "sem"])
    ax.bar(by_layer.index, by_layer["mean"], color="#4C72B0", width=0.7, label="patched (control -> partner)")
    ax.vlines(by_layer.index, by_layer["mean"] - by_layer["sem"], by_layer["mean"] + by_layer["sem"],
              color="#2A4A73", linewidth=1.2)

    base = baseline["p_false_normalised"].mean()
    ax.axhline(base, color="#C44E52", linestyle="--", linewidth=1.4, label=f"unpatched partner ({base:.2f})")
    control_base = interventions[
        (interventions.arm == "baseline") & (interventions.condition == "control")
    ]["p_false_normalised"].mean()
    ax.axhline(control_base, color="#55A868", linestyle=":", linewidth=1.4,
               label=f"unpatched control ({control_base:.2f})")

    random_arm = interventions[interventions.arm == "control_random"]
    if not random_arm.empty:
        by_random = random_arm.groupby("layer")["p_false_normalised"].mean()
        ax.scatter(by_random.index, by_random.values, color="#999999", marker="x", zorder=5,
                   label="random-direction control")

    ax.set_xlabel("patched layer (resid_post, final prompt position)")
    ax.set_ylabel("P(false proposition), forced choice")
    ax.legend(fontsize=8, frameon=False)
    return _finish(fig, ax, path, "Experiment 3: patching the control representation into the authority condition")


def plot_bidirectional(interventions: pd.DataFrame, path: Path) -> Path:
    """Both directions on one axis, as a change from each condition's own baseline."""
    baseline = interventions[interventions.arm == "baseline"].groupby("condition")["p_false_normalised"].mean()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for arm, condition, label in (
        ("patch_forward", "partner", "control -> partner (expect a decrease)"),
        ("patch_reverse", "control", "partner -> control (expect an increase)"),
    ):
        rows = interventions[interventions.arm == arm]
        if rows.empty:
            continue
        shift = rows.groupby("layer")["p_false_normalised"].mean() - baseline[condition]
        ax.plot(shift.index, shift.values, marker="o", markersize=3.5, linewidth=1.5,
                color=ARM_COLOURS[arm], label=label)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_xlabel("patched layer")
    ax.set_ylabel("change in P(false proposition) vs own baseline")
    ax.legend(fontsize=8, frameon=False)
    return _finish(
        fig, ax, path,
        "Experiment 4: is the effect bidirectional?",
        "Evidence is strong only if the two curves move in opposite directions at the same layers.",
    )


def write_all(behavioural: pd.DataFrame, divergence: pd.DataFrame, interventions: pd.DataFrame,
              out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        plot_behaviour(behavioural, out_dir / "1_behaviour.png"),
        plot_divergence(divergence, out_dir / "2_activation_divergence.png"),
        plot_intervention(interventions, out_dir / "3_intervention_by_layer.png"),
        plot_bidirectional(interventions, out_dir / "4_bidirectional_intervention.png"),
    ]
