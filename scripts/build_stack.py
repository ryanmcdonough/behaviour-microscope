#!/usr/bin/env python
"""Build the paper's cross-model artefacts from whatever runs are in `results/` right now.

The per-run figures and `summary.json` answer "is this run any good". This answers "what does
the study say", which is a different question and spans runs. Everything it writes is
*derived* -- nothing here is hand-maintained, so a re-run landing in `results/` changes the
paper's numbers by being re-run, not by anyone editing a table.

    python scripts/build_stack.py                 # -> paper/
    python scripts/build_stack.py --out somewhere

Run selection is `microscope.runs.select_current`: newest commit per configuration, quality
`fail` excluded, replicates kept so a median and range can be computed. Drop a new Qwen run in
and re-run this; it supersedes its predecessor with no edit anywhere.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from microscope import metrics, runs as runlib  # noqa: E402
from microscope.plots import ARM_COLOUR  # noqa: E402
from microscope.scenarios import PLANNED_CONTRASTS  # noqa: E402

ARMS = runlib.ARMS
ARM_SHORT = {
    "floor": "floor", "junior_said": "jnr said", "junior_confirmed": "jnr conf",
    "partner_said": "ptr said", "partner_confirmed": "ptr conf", "court": "court",
    "adverse": "adverse",
}
# Display order: open weights, then the legal model, then the frontier tier. The tier argument
# is the paper's spine, so the tables should read down it.
TIER_ORDER = ["gemma", "Qwen", "Thomson", "gpt-5.1", "gpt-5.6", "claude"]


def tier_sort(name: str) -> tuple[int, str]:
    for i, prefix in enumerate(TIER_ORDER):
        if name.startswith(prefix):
            return (i, name)
    return (len(TIER_ORDER), name)


def pct(value) -> str:
    return "--" if value is None else f"{value * 100:.1f}"


def median_and_range(values: list[float]) -> tuple[float | None, str]:
    """Median with its observed range. Never a pooled denominator: replicates are the same items."""
    if not values:
        return None, ""
    med = statistics.median(values)
    if len(values) == 1 or min(values) == max(values):
        return med, ""
    return med, f"{min(values) * 100:.1f}-{max(values) * 100:.1f}"


# --------------------------------------------------------------------------- tables


def table_fpar(selected) -> pd.DataFrame:
    rows = []
    for config, group in sorted(selected.items(), key=lambda kv: tier_sort(kv[0])):
        row = {"configuration": config, "n_runs": len(group), "quality": group[0].quality}
        for arm in ARMS:
            values = [v for v, _ in (r.rate(arm) for r in group) if v is not None]
            denominators = {n for _, n in (r.rate(arm) for r in group)}
            med, spread = median_and_range(values)
            row[arm] = None if med is None else round(med * 100, 1)
            row[f"{arm}_n"] = min(denominators) if denominators else 0
            if spread:
                row[f"{arm}_range"] = spread
        accs = [a for a in (r.floor_accuracy() for r in group) if a is not None]
        row["floor_accuracy"] = round(statistics.median(accs) * 100, 1) if accs else None
        worst_n = min((min(n for _, n in (r.rate(a) for r in group)) for a in ARMS), default=0)
        row["upper_bound_on_zero"] = round(runlib.clopper_pearson_upper(0, worst_n) * 100, 1)
        rows.append(row)
    return pd.DataFrame(rows)


def table_contrasts(selected) -> pd.DataFrame:
    """Planned contrasts, from the first run of each configuration.

    Deliberately not averaged across replicates: a p-value is a property of a measurement, and
    an average of three is not a test of anything. The FPAR table carries the replicate spread.
    """
    rows = []
    for config, group in sorted(selected.items(), key=lambda kv: tier_sort(kv[0])):
        summary = json.loads((group[0].dir / "summary.json").read_text())
        for contrast in summary.get("planned_contrasts_binary", []):
            rows.append({
                "configuration": config, "measure": "binary (FPAR)",
                **{k: contrast.get(k) for k in
                   ("arm_a", "arm_b", "isolates", "n", "rate_a", "rate_b", "difference",
                    "p_value", "p_holm", "significant_holm_05")},
            })
        for contrast in summary.get("planned_contrasts_continuous", []):
            rows.append({
                "configuration": config, "measure": "continuous (p_false_normalised)",
                **{k: contrast.get(k) for k in
                   ("arm_a", "arm_b", "isolates", "n", "rate_a", "rate_b", "difference",
                    "p_value", "p_holm", "significant_holm_05")},
            })
    return pd.DataFrame(rows)


def table_factorial(selected) -> pd.DataFrame:
    rows = []
    for config, group in sorted(selected.items(), key=lambda kv: tier_sort(kv[0])):
        factorial = json.loads((group[0].dir / "summary.json").read_text()).get("factorial", {})
        if not factorial.get("available"):
            continue
        row = {"configuration": config, "n": factorial["n"], "measure": factorial.get("measure")}
        for key, label in (("main_effect_source", "source"), ("main_effect_verb", "verb"),
                           ("interaction", "interaction")):
            effect = factorial[key]
            row[f"{label}_mean"] = round(effect["mean"], 4)
            row[f"{label}_dz"] = round(effect["cohens_dz"], 2)
            row[f"{label}_p"] = effect["p_value"]
        row.update({f"cell_{k}": round(v, 4) for k, v in factorial["cell_means"].items()})
        rows.append(row)
    return pd.DataFrame(rows)


def table_mechanism(selected) -> pd.DataFrame:
    rows = []
    for config, group in sorted(selected.items(), key=lambda kv: tier_sort(kv[0])):
        run = group[0]
        if not run.mechanistic:
            continue
        summary = json.loads((run.dir / "summary.json").read_text())
        rep, causal = summary.get("representational", {}), summary.get("causal", {})
        controls = summary.get("intervention_controls", {})
        row = {
            "configuration": config,
            "n_layers": run.manifest.get("n_layers"),
            "peak_relative_l2": round(rep.get("max_mean_relative_l2", float("nan")), 4),
            "candidate_layers": ", ".join(str(x) for x in rep.get("candidate_layers", [])),
        }
        for key in ("patch_forward", "patch_reverse"):
            layers = causal.get(key, [])
            if not layers:
                continue
            best = max(layers, key=lambda x: abs(x["mean_difference"]))
            onset = [x["layer"] for x in layers
                     if abs(x["mean_difference"]) > 0.1 * abs(best["mean_difference"])]
            row[f"{key}_from"] = round(best["mean_a"], 4)
            row[f"{key}_to"] = round(best["mean_b"], 4)
            row[f"{key}_peak_layer"] = best["layer"]
            row[f"{key}_onset_layer"] = min(onset) if onset else None
        row["zero_control_max"] = controls.get("control_zero", {}).get("max_abs_change_in_p_false")
        row["random_control_mean"] = controls.get("control_random", {}).get("mean_abs_change_in_p_false")
        # The maximum matters as much as the mean: on gemma it exceeds the real effect, so the
        # control licenses the claim on average across layers and items, not item by item.
        row["random_control_max"] = controls.get("control_random", {}).get("max_abs_change_in_p_false")
        rows.append(row)
    return pd.DataFrame(rows)


def table_elaboration(selected) -> pd.DataFrame:
    rows = []
    for config, group in sorted(selected.items(), key=lambda kv: tier_sort(kv[0])):
        per_arm: dict[str, list[float]] = {a: [] for a in ARMS}
        reason = None
        for run in group:
            frame = pd.read_csv(run.dir / "behavioural.csv")
            result = metrics.elaboration_by_arm(
                frame, PLANNED_CONTRASTS, reasoning_expected=run.thinking
            )
            if not result.get("available"):
                reason = result.get("reason")
                continue
            for arm, rate in result["rate_by_arm"].items():
                if arm in per_arm:
                    per_arm[arm].append(rate)
        if not any(per_arm.values()):
            rows.append({"configuration": config, "available": False, "reason": reason})
            continue
        row = {"configuration": config, "available": True, "n_runs": len(group)}
        for arm in ARMS:
            med, _ = median_and_range(per_arm[arm])
            row[arm] = None if med is None else round(med * 100, 1)
        rows.append(row)
    return pd.DataFrame(rows)


def table_stability(selected) -> pd.DataFrame:
    rows = []
    for config, group in sorted(selected.items(), key=lambda kv: tier_sort(kv[0])):
        if len(group) < 2:
            continue
        maps = [{(r["scenario_id"], r["condition"]): r["chosen_letter"] for r in run.rows}
                for run in group]
        keys = set(maps[0])
        unstable = [k for k in keys if len({m.get(k) for m in maps}) > 1]
        by_arm: dict[str, int] = {}
        for _, arm in unstable:
            by_arm[arm] = by_arm.get(arm, 0) + 1
        rows.append({
            "configuration": config, "n_runs": len(group), "n_items": len(keys),
            "n_unstable": len(unstable),
            "where": ", ".join(f"{a}:{c}" for a, c in sorted(by_arm.items())) or "none",
        })
    return pd.DataFrame(rows)


def table_provenance(selected) -> pd.DataFrame:
    rows = []
    for config, group in sorted(selected.items(), key=lambda kv: tier_sort(kv[0])):
        for run in group:
            rows.append({
                "configuration": config, "directory": run.label, "model": run.model,
                "provider": run.provider, "reasoning": run.thinking, "effort": run.effort,
                "quality": run.quality, "commit": run.commit, "interp_engine": run.engine,
                "mechanistic": run.mechanistic, "timestamp_utc": run.timestamp,
                "n_measurements": len(run.rows),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- figures


def figure_fpar(fpar: pd.DataFrame, path: Path) -> Path:
    """The headline: acceptance by arm, every configuration on one axis."""
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.8 / max(1, len(fpar))
    for i, (_, row) in enumerate(fpar.iterrows()):
        xs = [j + i * width for j in range(len(ARMS))]
        ys = [row[a] if pd.notna(row[a]) else 0 for a in ARMS]
        ax.bar(xs, ys, width=width, label=row["configuration"].split(" @")[0])
    ax.set_xticks([j + 0.4 - width / 2 for j in range(len(ARMS))])
    ax.set_xticklabels([ARM_SHORT[a] for a in ARMS])
    ax.set_ylabel("false proposition accepted (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, ncol=2, frameon=False)
    ax.set_title("Acceptance of a false legal proposition, by who is credited with it",
                 fontsize=11, loc="left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def figure_tier(fpar: pd.DataFrame, path: Path) -> Path:
    """Partner against court: the ordering claim, one point per configuration."""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 100], [0, 100], color="#BBBBBB", linestyle="--", linewidth=1, zorder=0)
    # Several configurations sit on top of each other at the origin, so labels are stacked
    # deterministically rather than left to collide.
    placed: list[tuple[float, float]] = []
    for _, row in fpar.iterrows():
        x, y = row["partner_confirmed"], row["court"]
        if pd.isna(x) or pd.isna(y):
            continue
        ax.scatter(x, y, s=70, zorder=2)
        dx, dy = 8.0, 4.0
        while any(abs(x - px) < 6 and abs(y + dy / 2.2 - py) < 4 for px, py in placed):
            dy += 11.0
        placed.append((x, y + dy / 2.2))
        ax.annotate(row["configuration"].split(" @")[0], (x, y),
                    textcoords="offset points", xytext=(dx, dy), fontsize=8)
    ax.set_xlabel("accepted from 'the supervising partner has confirmed' (%)")
    ax.set_ylabel("accepted from 'the court has held' (%)")
    ax.set_xlim(-5, 100)
    ax.set_ylim(-5, 100)
    ax.set_title("On the diagonal, a partner is weighted like a court",
                 fontsize=11, loc="left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def figure_reasoning(fpar: pd.DataFrame, path: Path) -> Path:
    """Reasoning off against on, for every model measured both ways."""
    pairs = []
    for _, row in fpar.iterrows():
        name = row["configuration"].split(" @")[0]
        if " think" in name:
            continue
        partner = fpar[fpar["configuration"].str.startswith(f"{name} think")]
        if not partner.empty:
            pairs.append((name, row, partner.iloc[0]))
    if not pairs:
        return path
    fig, axes = plt.subplots(1, len(pairs), figsize=(5.5 * len(pairs), 4.4), squeeze=False)
    for ax, (name, off, on) in zip(axes[0], pairs):
        for arm in ARMS:
            ax.plot([0, 1], [off[arm], on[arm]], marker="o",
                    color=ARM_COLOUR.get(arm, "#666666"), label=ARM_SHORT[arm])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["reasoning off", "reasoning on"])
        ax.set_ylim(-3, 100)
        ax.set_ylabel("accepted (%)")
        ax.set_title(name, fontsize=10, loc="left")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0][-1].legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle("Deliberation dissolves deference to a partner and leaves deference to a court",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def figure_mechanism(selected, path: Path) -> Path:
    """Bidirectional patch effect against relative depth, every mechanistic run overlaid."""
    mechanistic = [(c, g[0]) for c, g in selected.items() if g[0].mechanistic]
    if not mechanistic:
        return path
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for config, run in sorted(mechanistic, key=lambda kv: tier_sort(kv[0])):
        summary = json.loads((run.dir / "summary.json").read_text())
        layers = summary.get("causal", {}).get("patch_reverse", [])
        if not layers:
            continue
        n_layers = run.manifest.get("n_layers") or max(x["layer"] for x in layers) + 1
        peak = max(abs(x["mean_difference"]) for x in layers) or 1.0
        xs = [x["layer"] / (n_layers - 1) for x in layers]
        # Normalised by each model's own peak: the question is *where* the effect switches on,
        # and the absolute swings differ by three orders of magnitude across these models.
        ys = [abs(x["mean_difference"]) / peak for x in layers]
        ax.plot(xs, ys, marker="", linewidth=1.8, label=config.split(" @")[0])
    ax.set_xlabel("relative depth (layer / final layer)")
    ax.set_ylabel("patch effect, normalised to each model's peak")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Where the cue's effect becomes causal: nothing before roughly mid-network",
                 fontsize=11, loc="left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def figure_elaboration(elaboration: pd.DataFrame, path: Path) -> Path:
    """The sub-threshold signal, for configurations where it exists."""
    available = elaboration[elaboration["available"] == True]  # noqa: E712
    if available.empty:
        return path
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for _, row in available.iterrows():
        ax.plot(range(len(ARMS)), [row[a] for a in ARMS], marker="o",
                label=row["configuration"].split(" @")[0])
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([ARM_SHORT[a] for a in ARMS])
    ax.set_ylabel("response exceeded the single letter asked for (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Unmoved answers, moved behaviour: format compliance tracks the authority cue",
                 fontsize=11, loc="left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- digest


def markdown_table(frame: pd.DataFrame, columns: list[str], headers: list[str]) -> str:
    present = [c for c in columns if c in frame.columns]
    labels = [h for c, h in zip(columns, headers) if c in frame.columns]
    lines = ["| " + " | ".join(labels) + " |",
             "| " + " | ".join("---" for _ in labels) + " |"]
    for _, row in frame.iterrows():
        cells = []
        for column in present:
            cells.append(_format(row[column]))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _format(value) -> str:
    """Readable cells. Full precision lives in the CSVs; a digest that prints
    `1.862645149230957e-09` is harder to read, not more rigorous."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "--"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) < 0.001:
            return f"{value:.1e}"
        return f"{value:g}"
    return str(value)


def write_digest(out: Path, tables: dict[str, pd.DataFrame], selected, left_out) -> Path:
    fpar = tables["fpar"]
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts = [
        "# Results digest",
        "",
        "**Generated** by `scripts/build_stack.py` — do not edit. Re-run it after dropping a new",
        f"run into `results/` and every number here moves with the evidence. Built {generated}.",
        "",
        "Narrative, theory and the limits on what may be claimed: `PAPER_BRIEF.md`.",
        "",
        f"{len(selected)} configurations, "
        f"{sum(len(g) for g in selected.values())} runs.",
        "",
        "## 1. Acceptance of the false proposition (%)",
        "",
        markdown_table(
            fpar,
            ["configuration", *ARMS, "floor_accuracy"],
            ["configuration", *[ARM_SHORT[a] for a in ARMS], "floor acc"],
        ),
        "",
        "Median across replicates where a configuration ran more than once; `tables/fpar.csv`",
        "carries the per-arm denominators and observed ranges. A rate of 0/30 has an exact 95%",
        "upper bound of 9.5% — configurations at zero are not distinguishable from one another,",
        "nor from gpt-5.1.",
        "",
        "## 2. Replicate stability",
        "",
        markdown_table(
            tables["stability"],
            ["configuration", "n_runs", "n_unstable", "n_items", "where"],
            ["configuration", "runs", "unstable", "items", "which arms"],
        ) if not tables["stability"].empty else "_No configuration ran more than once._",
        "",
        "## 3. The 2×2",
        "",
        markdown_table(
            tables["factorial"],
            ["configuration", "measure", "n", "source_mean", "source_dz", "verb_mean", "verb_dz",
             "interaction_mean", "interaction_p"],
            ["configuration", "measure", "n", "source", "dz", "verb", "dz", "interaction", "p"],
        ),
        "",
        "**Cell means are not comparable across measures.** Check the `measure` column before",
        "reading two rows against each other.",
        "",
        "## 4. Mechanism",
        "",
        markdown_table(
            tables["mechanism"],
            ["configuration", "n_layers", "peak_relative_l2", "patch_forward_from",
             "patch_forward_to", "patch_reverse_to", "patch_reverse_onset_layer",
             "zero_control_max", "random_control_mean", "random_control_max"],
            ["configuration", "layers", "peak rel L2", "fwd from", "fwd to", "rev to",
             "onset", "zero ctrl max", "rand ctrl mean", "rand ctrl max"],
        ) if not tables["mechanism"].empty else "_No mechanistic run selected._",
        "",
        "Read `random_control_max` beside `random_control_mean`. The mean is what the quality",
        "gate compares and it passes cleanly; the maximum can exceed the real effect, so the",
        "control licenses the claim on average across layers and items, not item by item.",
        "",
        "## 5. Elaboration (exploratory)",
        "",
        markdown_table(
            tables["elaboration"][tables["elaboration"]["available"] == True],  # noqa: E712
            ["configuration", *ARMS],
            ["configuration", *[ARM_SHORT[a] for a in ARMS]],
        ),
        "",
        "Not a deference measure. Configurations absent from this table either scored off the",
        "logit path, where there is no format instruction being obeyed, or complied every time;",
        "`tables/elaboration.csv` records which.",
        "",
        "## 6. Provenance",
        "",
        markdown_table(
            tables["provenance"],
            ["directory", "configuration", "quality", "commit", "interp_engine", "mechanistic",
             "n_measurements"],
            ["directory", "configuration", "quality", "commit", "engine", "mech", "rows"],
        ),
        "",
    ]
    if left_out:
        parts += [
            "### Discovered but not selected",
            "",
            "Superseded by a newer run of the same configuration, or graded `fail`.",
            "",
            *[f"- `{r.label}` — {r.config}, quality {r.quality}" for r in left_out],
            "",
        ]
    path = out / "RESULTS.md"
    path.write_text("\n".join(parts))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(REPO_ROOT / "results"))
    parser.add_argument("--out", default=str(REPO_ROOT / "paper"))
    args = parser.parse_args()

    out = Path(args.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    discovered = runlib.discover([args.results])
    selected = runlib.select_current(discovered)
    left_out = runlib.superseded(discovered, selected)
    print(f"{len(discovered)} runs discovered, {len(selected)} configurations selected")

    tables = {
        "fpar": table_fpar(selected),
        "contrasts": table_contrasts(selected),
        "factorial": table_factorial(selected),
        "mechanism": table_mechanism(selected),
        "elaboration": table_elaboration(selected),
        "stability": table_stability(selected),
        "provenance": table_provenance(selected),
    }
    for name, frame in tables.items():
        target = out / "tables" / f"{name}.csv"
        frame.to_csv(target, index=False)
        print(f"  tables/{name}.csv  ({len(frame)} rows)")

    figures = [
        ("fig1_fpar_by_arm.png", lambda p: figure_fpar(tables["fpar"], p)),
        ("fig2_tier_gradient.png", lambda p: figure_tier(tables["fpar"], p)),
        ("fig3_reasoning.png", lambda p: figure_reasoning(tables["fpar"], p)),
        ("fig4_mechanism.png", lambda p: figure_mechanism(selected, p)),
        ("fig5_elaboration.png", lambda p: figure_elaboration(tables["elaboration"], p)),
    ]
    for name, build in figures:
        build(out / "figures" / name)
        print(f"  figures/{name}")

    digest = write_digest(out, tables, selected, left_out)
    print(f"  {digest.relative_to(out.parent)}")


if __name__ == "__main__":
    main()
