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
