"""Paired statistics for the behavioural, representational and causal measurements.

Everything here is paired, because both conditions of every measurement derive from the same
scenario. The unit of analysis is the scenario, not the prompt.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

RNG_SEED = 20260902


@dataclass
class PairedBinary:
    """A paired comparison of two binary outcomes over the same scenarios."""

    n: int
    rate_control: float
    rate_partner: float
    delta: float
    discordant_partner_only: int
    discordant_control_only: int
    mcnemar_exact_p: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class PairedContinuous:
    """A paired comparison of two continuous outcomes over the same scenarios.

    Deliberately not named ``control``/``partner``: the same statistic compares the two
    behavioural conditions in experiment 1 and a baseline against a patched forward in
    experiments 3 and 4. ``mean_difference`` is always ``b - a``, and the caller records what
    the two arms were.
    """

    n: int
    mean_a: float
    mean_b: float
    mean_difference: float
    ci_low: float
    ci_high: float
    wilcoxon_p: float
    cohens_dz: float

    def as_dict(self) -> dict:
        return asdict(self)


def false_proposition_acceptance(accepted_control, accepted_partner) -> PairedBinary:
    """FPAR in each condition, the Authority Deference Delta, and an exact paired test.

    McNemar's test conditions on the discordant pairs -- the scenarios where the two conditions
    disagreed -- which is the paired question. Its exact binomial form is used rather than the
    chi-square approximation because 30 scenarios will usually leave only a handful of
    discordant pairs.
    """
    control = np.asarray(accepted_control, dtype=bool)
    partner = np.asarray(accepted_partner, dtype=bool)
    if control.shape != partner.shape:
        raise ValueError("Conditions are not paired: different numbers of scenarios")
    partner_only = int(np.sum(~control & partner))
    control_only = int(np.sum(control & ~partner))
    discordant = partner_only + control_only
    p = 1.0 if discordant == 0 else float(stats.binomtest(partner_only, discordant, 0.5).pvalue)
    return PairedBinary(
        n=int(control.size),
        rate_control=float(control.mean()),
        rate_partner=float(partner.mean()),
        delta=float(partner.mean() - control.mean()),
        discordant_partner_only=partner_only,
        discordant_control_only=control_only,
        mcnemar_exact_p=p,
    )


def paired_difference(a, b, *, n_boot: int = 10_000, seed: int = RNG_SEED) -> PairedContinuous:
    """Mean paired difference with a bootstrap CI and a distribution-free test.

    The bootstrap resamples scenarios, not observations, so the interval carries the scenario
    sampling uncertainty that is the real limit on 30 items.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Arms are not paired: different numbers of scenarios")
    diff = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    boot = diff[idx].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    if np.allclose(diff, 0):
        p, dz = 1.0, 0.0
    else:
        p = float(stats.wilcoxon(diff, zero_method="wilcox").pvalue)
        sd = diff.std(ddof=1)
        dz = float(diff.mean() / sd) if sd > 0 else 0.0
    return PairedContinuous(
        n=int(diff.size),
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        mean_difference=float(diff.mean()),
        ci_low=float(low),
        ci_high=float(high),
        wilcoxon_p=p,
        cohens_dz=dz,
    )


def activation_divergence(control_acts: dict[int, "np.ndarray"], partner_acts: dict[int, "np.ndarray"]) -> pd.DataFrame:
    """Per-layer divergence between two matched activations at the same position.

    Three quantities, because they answer different questions and can disagree:

    * ``cosine_distance`` -- how much the direction moved, scale-free.
    * ``l2_distance`` -- how much the vector moved in absolute terms.
    * ``relative_l2`` -- that distance as a fraction of the control activation's norm, which is
      the one to compare across layers: residual norms grow with depth, so a raw L2 rises with
      layer index whether or not anything interesting happened.
    """
    rows = []
    for layer in sorted(control_acts):
        a = np.asarray(control_acts[layer], dtype=np.float64)
        b = np.asarray(partner_acts[layer], dtype=np.float64)
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        cosine = float(a @ b / (norm_a * norm_b)) if norm_a and norm_b else np.nan
        l2 = float(np.linalg.norm(a - b))
        rows.append(
            {
                "layer": layer,
                "cosine_distance": 1.0 - cosine,
                "l2_distance": l2,
                "relative_l2": l2 / norm_a if norm_a else np.nan,
                "control_norm": float(norm_a),
                "partner_norm": float(norm_b),
            }
        )
    return pd.DataFrame(rows)


def summarise_divergence(per_scenario: pd.DataFrame) -> pd.DataFrame:
    """Mean divergence per layer across scenarios, with a bootstrap CI on the mean.

    A layer is only interesting if the divergence is *consistent* across scenarios, so the
    spread matters as much as the mean. ``consistency`` is the fraction of scenarios above the
    across-layer median, a crude but assumption-free stand-in for that.
    """
    rng = np.random.default_rng(RNG_SEED)
    out = []
    for layer, group in per_scenario.groupby("layer"):
        values = group["relative_l2"].to_numpy(dtype=float)
        boot = values[rng.integers(0, values.size, size=(2000, values.size))].mean(axis=1)
        low, high = np.percentile(boot, [2.5, 97.5])
        out.append(
            {
                "layer": int(layer),
                "mean_relative_l2": float(values.mean()),
                "ci_low": float(low),
                "ci_high": float(high),
                "mean_cosine_distance": float(group["cosine_distance"].mean()),
                "sd_relative_l2": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "n_scenarios": int(values.size),
            }
        )
    frame = pd.DataFrame(out).sort_values("layer").reset_index(drop=True)
    frame["rank"] = frame["mean_relative_l2"].rank(ascending=False).astype(int)
    return frame


def candidate_layers(divergence: pd.DataFrame, k: int = 4) -> list[int]:
    """The k layers with the largest mean relative divergence.

    These are *candidates for intervention*, not located mechanisms. A layer earns that name
    only from the intervention experiments, and only if the controls in those experiments come
    out clean.
    """
    return [int(layer) for layer in divergence.nlargest(k, "mean_relative_l2")["layer"]]


# --------------------------------------------------------------------------- factorial


def holm_correction(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order.

    The planned contrasts are decided before the data is seen, but there are seven of them, and
    reporting seven uncorrected tests is how a null result is made to look like a finding. Holm
    is used rather than Bonferroni because it is uniformly more powerful at the same family-wise
    error rate, and rather than FDR because these are confirmatory tests of specific hypotheses.
    """
    indexed = sorted(enumerate(pvalues), key=lambda pair: pair[1])
    n = len(pvalues)
    adjusted = [0.0] * n
    running = 0.0
    for rank, (original_index, p) in enumerate(indexed):
        running = max(running, (n - rank) * p)
        adjusted[original_index] = min(1.0, running)
    return adjusted


def planned_contrasts(
    per_arm: dict[str, "pd.Series"],
    contrasts,
    *,
    binary: bool = False,
) -> pd.DataFrame:
    """Run each planned contrast as a paired test and Holm-correct the family.

    ``per_arm`` maps arm name to a Series indexed by scenario id. Arms absent from the run are
    skipped rather than faked, so a behavioural-only or partial run still reports what it has.
    """
    rows = []
    for arm_a, arm_b, isolates in contrasts:
        if arm_a not in per_arm or arm_b not in per_arm:
            continue
        a, b = per_arm[arm_a], per_arm[arm_b]
        common = a.index.intersection(b.index)
        if len(common) == 0:
            continue
        a, b = a.loc[common], b.loc[common]
        if binary:
            result = false_proposition_acceptance(a.astype(bool), b.astype(bool))
            rows.append({
                "arm_a": arm_a, "arm_b": arm_b, "isolates": isolates, "n": result.n,
                "rate_a": result.rate_control, "rate_b": result.rate_partner,
                "difference": result.delta, "p_value": result.mcnemar_exact_p,
                "test": "mcnemar_exact",
            })
        else:
            result = paired_difference(a.astype(float), b.astype(float), n_boot=4000)
            rows.append({
                "arm_a": arm_a, "arm_b": arm_b, "isolates": isolates, "n": result.n,
                "rate_a": result.mean_a, "rate_b": result.mean_b,
                "difference": result.mean_difference, "ci_low": result.ci_low,
                "ci_high": result.ci_high, "p_value": result.wilcoxon_p,
                "cohens_dz": result.cohens_dz, "test": "wilcoxon",
            })
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["p_holm"] = holm_correction(frame["p_value"].tolist())
    frame["significant_holm_05"] = frame["p_holm"] < 0.05
    return frame


def factorial_effects(cells: dict[tuple[str, str], "pd.Series"]) -> dict:
    """Main effects and interaction for the 2x2, computed within scenario.

    ``cells`` maps ``(source, verb)`` to a per-scenario Series. Every quantity is a paired
    contrast over the same scenarios, so this is a within-items analysis rather than a
    between-groups ANOVA -- the design is fully crossed within each scenario, which is what
    makes n=30 workable.

    The interaction is the one to read first when the two main effects disagree: a large
    interaction means source and verb are not additive, and the headline "authority effect"
    depends on which verb carries it.
    """
    required = [("junior", "said"), ("junior", "confirmed"), ("partner", "said"), ("partner", "confirmed")]
    if not all(cell in cells for cell in required):
        return {"available": False, "reason": "not all four factorial cells are present in this run"}

    js, jc = cells[("junior", "said")].astype(float), cells[("junior", "confirmed")].astype(float)
    ps, pc = cells[("partner", "said")].astype(float), cells[("partner", "confirmed")].astype(float)
    index = js.index.intersection(jc.index).intersection(ps.index).intersection(pc.index)
    js, jc, ps, pc = js.loc[index], jc.loc[index], ps.loc[index], pc.loc[index]

    # Main effect of source: partner mean minus junior mean, averaging over verb.
    source = ((ps + pc) / 2) - ((js + jc) / 2)
    # Main effect of verb: confirmed minus said, averaging over source.
    verb = ((jc + pc) / 2) - ((js + ps) / 2)
    # Interaction: does the source effect differ by verb?
    interaction = (pc - jc) - (ps - js)

    def summarise(diff, label):
        zero = np.zeros_like(diff.to_numpy(dtype=float))
        result = paired_difference(zero, diff.to_numpy(dtype=float), n_boot=4000)
        return {
            "effect": label,
            "mean": result.mean_difference,
            "ci_low": result.ci_low,
            "ci_high": result.ci_high,
            "p_value": result.wilcoxon_p,
            "cohens_dz": result.cohens_dz,
        }

    return {
        "available": True,
        "n": int(len(index)),
        "cell_means": {
            "junior_said": float(js.mean()), "junior_confirmed": float(jc.mean()),
            "partner_said": float(ps.mean()), "partner_confirmed": float(pc.mean()),
        },
        "main_effect_source": summarise(source, "source (partner - junior)"),
        "main_effect_verb": summarise(verb, "verb (confirmed - said)"),
        "interaction": summarise(interaction, "source x verb"),
    }
