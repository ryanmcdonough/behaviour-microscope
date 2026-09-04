"""Automated checks on a run's own output.

A Colab run is unsupervised end to end -- Run All, walk away, come back to plots -- so the
pipeline has to be able to say "these results are not trustworthy" itself, rather than relying
on someone reading every row of every CSV before drawing a conclusion. Every check here exists
because RESEARCH.md or CLAUDE.md's scientific-discipline section names it as a reason a
downstream number would not mean what it looks like it means: a model that never engages with
the forced-choice format, a patch machinery that is not actually a no-op at zero magnitude, a
control condition the model gets right at chance so there is nothing for authority to move it
away from.

This module does not stop a run. Data quality is only knowable after the run finishes, and a run
that produced a `fail` is still evidence -- usually that the model or the prompt format is the
wrong choice, which is itself worth knowing. It grades what came out, and the grade is written
alongside the numbers it grades rather than only printed once and lost in a Colab scrollback.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

Status = str  # "pass" | "warn" | "fail"

_RANK = {"pass": 0, "warn": 1, "fail": 2}


@dataclass
class Check:
    name: str
    status: Status
    detail: str

    def as_dict(self) -> dict:
        return asdict(self)


def _check(name: str, condition: bool, *, warn: bool = False, detail: str) -> Check:
    if condition:
        return Check(name, "pass", detail)
    return Check(name, "warn" if warn else "fail", detail)


# --------------------------------------------------------------------------- individual checks


def check_logit_path(manifest: dict) -> Check:  # noqa: D401
    """The decoded logits must match the sampler's own, where the backend can report them.

    If this fails, every probability in the run is a logit-lens approximation rather than the
    model's real next-token distribution -- see RESEARCH.md section 3.4.
    """
    result = manifest.get("logit_path_check", {})
    if not result.get("checked") and not manifest.get("mechanistic", True):
        return Check("logit_path", "pass",
                     "Not applicable: behavioural-only backend, no residual decode to verify.")
    if not result.get("checked"):
        return Check(
            "logit_path", "warn",
            f"Not checked ({result.get('reason', 'no reason given')}). Expected on vLLM; "
            "GenStep.logits is eager-only. Cannot confirm the decoded logits match the sampler.",
        )
    return _check(
        "logit_path", bool(result.get("within_tolerance")),
        detail=f"max |Δprob| vs sampler = {result.get('max_abs_prob_difference'):.2e}, "
        f"argmax agrees = {result.get('argmax_agrees')}",
    )


def check_parse_rate(behavioural: pd.DataFrame) -> Check:
    """Did the model actually emit a parseable A or B?

    On a backend without logprobs the answer letter is read out of the response text, so a
    model that hedges, refuses, or wanders produces no measurement at all. A run with parse
    failures is missing data, not noisy data, and the two must not be confused.
    """
    if "parse_ok" not in behavioural.columns:
        return Check("parse_rate", "pass", "Not applicable to this backend.")
    failures = int((~behavioural["parse_ok"].astype(bool)).sum())
    total = len(behavioural)
    rate = failures / total if total else 0.0
    if rate > 0.1:
        return Check(
            "parse_rate", "fail",
            f"{failures}/{total} responses had no parseable answer letter ({rate:.0%}). Those "
            "rows are excluded from every rate, so each arm's denominator is smaller than it "
            "looks and the arms are no longer scored on the same scenarios. On a reasoning run "
            "this usually means generation hit its token budget before the model finished "
            "reasoning -- raise max_gen_tokens and re-run rather than reading these numbers.",
        )
    if failures:
        return Check("parse_rate", "warn",
                     f"{failures}/{total} responses had no parseable answer letter. Those rows "
                     "are missing data and are excluded from rates.")
    return Check("parse_rate", "pass", f"All {total} responses parsed to an answer letter.")


def check_letter_mass(behavioural: pd.DataFrame) -> Check:
    """How much probability the model puts on *any* answer-letter token.

    Low mass means the model is mostly saying something other than "A" or "B" as its first
    token, so the forced-choice probability this experiment measures everything from is being
    read off a tail the model barely uses.
    """
    if "letter_mass" not in behavioural.columns or behavioural["letter_mass"].isna().all():
        return Check(
            "letter_mass", "pass",
            "Not applicable: this backend reports no token probabilities, so the forced choice "
            "is read from the response text. See the parse_rate check instead.",
        )
    masses = behavioural["letter_mass"].dropna()
    mean_mass = float(masses.mean())
    min_mass = float(masses.min())
    if mean_mass < 0.05:
        return Check(
            "letter_mass", "fail",
            f"Mean P(A or B) = {mean_mass:.3f}. The model is essentially not answering in the "
            "requested format; every downstream probability is noise.",
        )
    if mean_mass < 0.4 or min_mass < 0.02:
        return Check(
            "letter_mass", "warn",
            f"Mean P(A or B) = {mean_mass:.3f}, worst scenario = {min_mass:.3f}. Usable but "
            "weak adherence to the forced-choice format; read individual low-mass rows before "
            "trusting them.",
        )
    return Check("letter_mass", "pass", f"Mean P(A or B) = {mean_mass:.3f}, worst = {min_mass:.3f}")


def check_control_accuracy(behavioural: pd.DataFrame) -> Check:
    """Accuracy in the control condition -- is there a correct answer to be deferred away from?

    RESEARCH.md threat #4: if the model does not track the legal material even without an
    authority cue, an authority effect (or the absence of one) does not mean what it looks like
    it means, because there was no reliable "correct" answer for the cue to move the model away
    from in the first place.
    """
    # The floor arm is the true unpressured baseline where it was run; otherwise fall back to
    # the lowest-authority arm present.
    for candidate in ("floor", "junior_said", "control"):
        subset = behavioural[behavioural.condition == candidate]
        if not subset.empty:
            break
    else:
        return Check("control_accuracy", "warn", "No baseline arm in this run to check.")
    accuracy = float(subset["correct"].mean())
    label = candidate
    if accuracy <= 0.5:
        return Check(
            "control_accuracy", "fail",
            f"Accuracy in the '{label}' arm is {accuracy:.0%}, at or below chance on a two-option "
            "question. This model is not reliably tracking the legal material even without an "
            "authority cue; the behavioural and downstream results describe something other "
            "than deference.",
        )
    if accuracy < 0.65:
        return Check(
            "control_accuracy", "warn",
            f"Accuracy in the '{label}' arm is {accuracy:.0%} -- above chance but weak. Read the "
            "per-scenario detail before treating the aggregate deference delta as reliable.",
        )
    return Check("control_accuracy", "pass", f"Accuracy in the '{label}' arm is {accuracy:.0%}")


def check_zero_patch_is_noop(controls: dict) -> Check:
    """The intervention machinery, run with a zero-magnitude patch, must reproduce the baseline.

    This is the check on RESEARCH.md section 3.2's exactness claim. If a *zero* patch moves the
    output, the steering context itself perturbs the forward pass, and the "exact replacement"
    argument for every real patch in the run does not hold.
    """
    zero = controls.get("control_zero")
    if zero is None:
        return Check("zero_patch_noop", "pass",
                     "Not applicable: this run performed no interventions (behavioural-only backend).")
    max_change = zero["max_abs_change_in_p_false"]
    return _check(
        "zero_patch_noop", max_change < 1e-6,
        detail=f"Max |Δ P(false)| under a zero-magnitude patch = {max_change:.2e} "
        f"(n={zero['n']}). Should be exactly 0.",
    )


def check_random_control_smaller_than_effect(interventions: pd.DataFrame | None) -> Check:
    """A random-direction patch of matched magnitude should move the output less than the
    real patch does, at the layers the real patch was strongest.

    Without this, "patching this direction changed the answer" is indistinguishable from
    "perturbing this layer at all changes the answer" -- the difference the random control
    exists to isolate. See CLAUDE.md's scientific-discipline section.
    """
    if interventions is None or interventions.empty:
        return Check("random_control", "pass",
                     "Not applicable: this run performed no interventions (behavioural-only backend).")
    forward = interventions[interventions.arm == "patch_forward"]
    random_arm = interventions[interventions.arm == "control_random"]
    baseline = interventions[interventions.arm == "baseline"]
    if not baseline.empty:
        baseline = baseline[baseline.condition == baseline.condition.iloc[-1]]
    if interventions is None or interventions.empty:
        return Check("random_control", "pass",
                     "Not applicable: this run performed no interventions (behavioural-only backend).")
    if random_arm.empty or forward.empty or baseline.empty:
        return Check("random_control", "warn", "Missing arms; nothing to compare.")

    base_by_scenario = baseline.set_index("scenario_id")["p_false_normalised"]
    layers = sorted(random_arm["layer"].unique())

    real_effect = (
        forward[forward.layer.isin(layers)]
        .assign(base=lambda d: d.scenario_id.map(base_by_scenario))
        .assign(abs_shift=lambda d: (d.p_false_normalised - d.base).abs())["abs_shift"]
        .mean()
    )
    random_effect = (
        random_arm.assign(base=lambda d: d.scenario_id.map(base_by_scenario))
        .assign(abs_shift=lambda d: (d.p_false_normalised - d.base).abs())["abs_shift"]
        .mean()
    )
    if math.isnan(real_effect) or math.isnan(random_effect):
        return Check("random_control", "warn", "Could not align scenarios between arms.")
    if real_effect < 1e-9:
        return Check(
            "random_control", "warn",
            "The real patch itself has ~zero effect at the candidate layers, so there is "
            "nothing for the random-direction control to be compared against.",
        )
    ratio = random_effect / real_effect
    return _check(
        "random_control", ratio < 0.5, warn=(ratio < 0.8),
        detail=f"Mean |Δ P(false)| -- real patch: {real_effect:.4f}, random-direction control: "
        f"{random_effect:.4f} (ratio {ratio:.2f}). Should be well under 1.0: a real, structured "
        "effect should outweigh a magnitude-matched random perturbation.",
    )


def check_no_nan_or_inf(behavioural: pd.DataFrame, interventions, divergence) -> Check:
    """Non-finite values anywhere they would corrupt a statistic.

    A backend without logprobs leaves the *behavioural* probability columns empty by design, so
    a NaN there is expected rather than a fault. That exemption applies to nothing else:
    interventions only exist on a local backend, which always has probabilities, so a NaN there
    means a computation went wrong. An infinity is a fault anywhere.
    """
    optional = {
        "p_a", "p_b", "p_correct", "p_false", "p_false_normalised",
        "letter_mass", "n_prompt_tokens",
    }
    frames = {"behavioural": behavioural, "interventions": interventions, "activation_analysis": divergence}
    problems = []
    for name, frame in frames.items():
        if frame is None or frame.empty:
            continue
        numeric = frame.select_dtypes(include=[np.number])
        for column in numeric.columns:
            values = numeric[column].to_numpy(dtype=float)
            if values.size == 0:
                continue
            exempt = name == "behavioural" and column in optional
            if np.isinf(values).any():
                problems.append(f"{name}.{column} (inf)")
            elif np.isnan(values).any() and not exempt:
                problems.append(f"{name}.{column} (nan)")
    return _check(
        "finite_values", not problems,
        detail=f"Non-finite values in: {problems}" if problems else "None found",
    )


def check_sample_size(behavioural: pd.DataFrame) -> Check:
    n = behavioural["scenario_id"].nunique()
    return _check(
        "sample_size", n >= 20, warn=(n >= 10),
        detail=f"{n} scenarios. The paired tests are the point of this design, but n<20 leaves "
        "little power for anything short of a large effect.",
    )


def check_stale_scenarios(manifest: dict) -> Check:
    stale = manifest.get("stale_scenarios") or []
    if not stale:
        return Check("stale_ground_truth", "pass", "No scenario's ground truth is superseded as of this run.")
    return Check(
        "stale_ground_truth", "warn",
        f"{stale} were scored against ground truth superseded by law in force at run time. "
        "Read each scenario's 'note' field before trusting its result.",
    )


CHECKS = (
    "logit_path", "parse_rate", "letter_mass", "control_accuracy", "zero_patch_noop",
    "random_control", "finite_values", "sample_size", "stale_ground_truth",
)


def run_checks(
    behavioural: pd.DataFrame,
    divergence: "pd.DataFrame | None",
    interventions: "pd.DataFrame | None",
    summary: dict,
    manifest: dict,
) -> dict:
    """Every check, plus the worst status across them as the run's overall verdict."""
    checks = [
        check_logit_path(manifest),
        check_parse_rate(behavioural),
        check_letter_mass(behavioural),
        check_control_accuracy(behavioural),
        check_zero_patch_is_noop(summary.get("intervention_controls", {})),
        check_random_control_smaller_than_effect(interventions),
        check_no_nan_or_inf(behavioural, interventions, divergence),
        check_sample_size(behavioural),
        check_stale_scenarios(manifest),
    ]
    overall = max((c.status for c in checks), key=_RANK.get, default="pass")
    return {
        "overall": overall,
        "checks": [c.as_dict() for c in checks],
    }


def format_report(report: dict) -> str:
    """A short, terminal-friendly rendering, for the notebook and for a script."""
    icon = {"pass": "OK  ", "warn": "WARN", "fail": "FAIL"}
    lines = [f"Overall: {report['overall'].upper()}", ""]
    for check in report["checks"]:
        lines.append(f"[{icon[check['status']]}] {check['name']}: {check['detail']}")
    return "\n".join(lines)


def review(run_dir: Path | str) -> dict:
    """Re-run the quality gate against an already-completed run directory.

    For a run downloaded from Colab, or reviewing someone else's results without re-running the
    model: reads the same CSVs and JSON the pipeline itself wrote, so this is exactly what
    ``run_all`` computed, callable again on saved output.
    """
    run_dir = Path(run_dir)
    behavioural = pd.read_csv(run_dir / "behavioural.csv")
    div_path, int_path = run_dir / "activation_analysis.csv", run_dir / "interventions.csv"
    divergence = pd.read_csv(div_path) if div_path.exists() else None
    interventions = pd.read_csv(int_path) if int_path.exists() else None
    summary = json.loads((run_dir / "summary.json").read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    return run_checks(behavioural, divergence, interventions, summary, manifest)
