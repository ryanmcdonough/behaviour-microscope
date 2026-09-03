"""The experiment's figures. Every one is derived directly from the run's own CSV data."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .scenarios import ARMS, FACTORIAL_CELLS  # noqa: E402

ARM_ORDER = [a.name for a in ARMS]
ARM_COLOUR = {
    "floor": "#8C8C8C",
    "junior_said": "#4C72B0",
    "junior_confirmed": "#7BA1D1",
    "partner_said": "#C44E52",
    "partner_confirmed": "#8C2F39",
    "court": "#55A868",
    "adverse": "#B07AA1",
}
ARM_COLOURS = {"patch_forward": "#4C72B0", "patch_reverse": "#C44E52", "control_random": "#999999"}


def _finish(fig, ax, path: Path, title: str, subtitle: str | None = None) -> Path:
    ax.set_title(title if not subtitle else f"{title}\n{subtitle}", fontsize=11, loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_behaviour(behavioural: pd.DataFrame, path: Path, cfg=None) -> Path:
    """Acceptance rate per arm, with the factorial cells grouped so the 2x2 reads at a glance."""
    present = [a for a in ARM_ORDER if a in set(behavioural["condition"])]
    rates = [
        behavioural.loc[behavioural.condition == a, "accepted_false_proposition"].mean()
        for a in present
    ]
    fig, ax = plt.subplots(figsize=(max(7, 1.35 * len(present)), 4.4))
    bars = ax.bar(range(len(present)), rates, color=[ARM_COLOUR.get(a, "#4C72B0") for a in present], width=0.62)
    for i, rate in enumerate(rates):
        ax.text(i, rate, f" {rate:.0%}", va="bottom", ha="center", fontsize=9)
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([a.replace("_", "\n") for a in present], fontsize=9)
    ax.set_ylabel("False proposition acceptance rate")
    ax.set_ylim(0, max(1.0, (max(rates) if rates else 0) * 1.25))
    if "floor" in present:
        floor_rate = rates[present.index("floor")]
        ax.axhline(floor_rate, color="#333333", linestyle=":", linewidth=1,
                   label=f"floor, no assertion ({floor_rate:.0%})")
        ax.legend(fontsize=8, frameon=False)
    return _finish(fig, ax, path, "Experiment 1: does who said it change the answer?",
                   "Same law, same question, same options. Only the attribution differs.")


def plot_factorial(behavioural: pd.DataFrame, path: Path) -> Path:
    """The 2x2 as an interaction plot: two lines, one per source, across the two verbs."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    verbs = ["said", "confirmed"]
    drawn = False
    for source, colour in (("junior", "#4C72B0"), ("partner", "#C44E52")):
        ys = []
        for verb in verbs:
            arm = FACTORIAL_CELLS.get((source, verb))
            subset = behavioural[behavioural.condition == arm] if arm else None
            ys.append(subset["accepted_false_proposition"].mean() if subset is not None and not subset.empty else float("nan"))
        if not all(pd.isna(y) for y in ys):
            ax.plot([0, 1], ys, marker="o", markersize=7, linewidth=2, color=colour, label=source)
            drawn = True
    ax.set_xticks([0, 1], verbs)
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(0, 1)
    ax.set_xlabel("epistemic verb")
    ax.set_ylabel("False proposition acceptance rate")
    if drawn:
        ax.legend(title="source", fontsize=9, title_fontsize=9, frameon=False)
    return _finish(
        fig, ax, path, "The 2x2: is it seniority, or is it the verb?",
        "Parallel lines mean the effects are additive. A vertical gap is source; a slope is verb.",
    )


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


def plot_intervention(interventions: pd.DataFrame, path: Path, contrast=("low", "high")) -> Path:
    """Effect on P(false) of patching the low-authority activation into the high-authority arm."""
    low, high = contrast
    baseline = interventions[(interventions.arm == "baseline") & (interventions.condition == high)]
    forward = interventions[interventions.arm == "patch_forward"]
    fig, ax = plt.subplots(figsize=(9, 4.2))

    by_layer = forward.groupby("layer")["p_false_normalised"].agg(["mean", "sem"])
    ax.bar(by_layer.index, by_layer["mean"], color="#4C72B0", width=0.7, label=f"patched ({low} -> {high})")
    ax.vlines(by_layer.index, by_layer["mean"] - by_layer["sem"], by_layer["mean"] + by_layer["sem"],
              color="#2A4A73", linewidth=1.2)

    base = baseline["p_false_normalised"].mean()
    ax.axhline(base, color="#C44E52", linestyle="--", linewidth=1.4, label=f"unpatched {high} ({base:.2f})")
    control_base = interventions[
        (interventions.arm == "baseline") & (interventions.condition == low)
    ]["p_false_normalised"].mean()
    ax.axhline(control_base, color="#55A868", linestyle=":", linewidth=1.4,
               label=f"unpatched {low} ({control_base:.2f})")

    random_arm = interventions[interventions.arm == "control_random"]
    if not random_arm.empty:
        by_random = random_arm.groupby("layer")["p_false_normalised"].mean()
        ax.scatter(by_random.index, by_random.values, color="#999999", marker="x", zorder=5,
                   label="random-direction control")

    ax.set_xlabel("patched layer (resid_post, final prompt position)")
    ax.set_ylabel("P(false proposition), forced choice")
    ax.legend(fontsize=8, frameon=False)
    return _finish(fig, ax, path, f"Experiment 3: patching {low} into {high}")


def plot_bidirectional(interventions: pd.DataFrame, path: Path, contrast=("low", "high")) -> Path:
    """Both directions on one axis, as a change from each condition's own baseline."""
    low, high = contrast
    baseline = interventions[interventions.arm == "baseline"].groupby("condition")["p_false_normalised"].mean()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for arm, condition, label in (
        ("patch_forward", high, f"{low} -> {high} (expect a decrease)"),
        ("patch_reverse", low, f"{high} -> {low} (expect an increase)"),
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


def write_all(behavioural, divergence, interventions, out_dir: Path, cfg=None) -> list[Path]:
    """Every figure the run has data for. Mechanistic plots are skipped, not faked, when absent."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    contrast = tuple(getattr(cfg, "contrast", ("junior_said", "partner_said")))
    written = [
        plot_behaviour(behavioural, out_dir / "1_behaviour.png"),
        plot_factorial(behavioural, out_dir / "2_factorial.png"),
    ]
    if divergence is not None and not divergence.empty:
        written.append(plot_divergence(divergence, out_dir / "3_activation_divergence.png"))
    if interventions is not None and not interventions.empty:
        written.append(plot_intervention(interventions, out_dir / "4_intervention_by_layer.png", contrast))
        written.append(plot_bidirectional(interventions, out_dir / "5_bidirectional_intervention.png", contrast))
    return written
