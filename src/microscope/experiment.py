"""Orchestration: four experiments, their controls, and a reproducible result directory.

    Experiment 1  behavioural       does the authority cue change the answer?
    Experiment 2  representational  where do the two conditions' activations diverge?
    Experiment 3  causal (forward)  does patching control -> partner reduce deference?
    Experiment 4  causal (reverse)  does patching partner -> control increase it?

Experiments 3 and 4 sweep every layer rather than only the candidates from experiment 2. The
sweep *is* the unrelated-layer control: a patch that matters only where the representations
diverged looks different from one that matters everywhere.
"""

from __future__ import annotations

import json
import gc
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import interp, metrics
from .backends import Backend, BackendSpec, LocalBackend, Measurement, measure_many
from .scenarios import (
    ARMS_BY_NAME,
    CONDITIONS,
    DEFAULT_CONTRAST,
    FACTORIAL_CELLS,
    PLANNED_CONTRASTS,
    Scenario,
    load_scenarios,
    stale_scenarios,
)

EXPERIMENT_VERSION = "authority_v1"
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class RunConfig:
    model_id: str = "google/gemma-3-12b-it"
    backend: str = "eager"
    dtype: str | None = None
    max_gen_tokens: int = 24
    n_candidate_layers: int = 4
    seed: int = metrics.RNG_SEED
    limit: int | None = None
    data_file: Path = REPO_ROOT / "data" / "authority_v1.jsonl"
    results_root: Path = REPO_ROOT / "results"
    save_activations: bool = False
    extra_load_kwargs: dict = field(default_factory=dict)

    # Which backend to measure through. "local" is interp-engine and is the only kind that can
    # run the mechanistic experiments; "openai" and "anthropic" are behavioural-only.
    provider: str = "local"
    # Provider-specific construction kwargs. For the local provider these merge with
    # extra_load_kwargs, which is kept because it is what the notebook already passes.
    provider_options: dict = field(default_factory=dict)
    # Which arms to run. All seven by default; narrow it for a cheap partial run.
    arms: tuple[str, ...] = CONDITIONS
    # The pair the mechanistic experiments patch between. Source varies, verb held constant, so
    # the mechanism answers the source question rather than the epistemic-verb question.
    contrast: tuple[str, str] = DEFAULT_CONTRAST
    # How often a long phase reports progress. Lower it to watch a run more closely.
    progress_every_seconds: float = 15.0
    # Skip the capture and patching experiments even on a backend that could run them. The
    # sweep is O(layers x scenarios) forward passes, which is minutes on a 12B and hours on a
    # large mixture-of-experts model; sometimes only the behavioural numbers are wanted.
    mechanistic: bool = True
    # How many measurements may be in flight at once in experiment 1. 1 keeps the sequential
    # behaviour every result so far was produced with. Higher only takes effect on a backend
    # that declares itself concurrency-safe -- the API backends, where the loop is mostly
    # round-trip latency, and the local vLLM generate path, where several in-flight requests
    # are what lets the engine batch. Never applies to the mechanistic experiments.
    max_concurrency: int = 1
    # Turn a hybrid-reasoning model's reasoning on. Ignored by a model with no reasoning mode.
    # A reasoning run is behavioural-only: the answer no longer sits at the final prompt
    # position, so the patching experiments would be intervening on the wrong thing.
    enable_thinking: bool = False
    # Stable folder name under results_root. A timestamp is used if this is unset. Set it when
    # the run may be interrupted -- Colab timeouts, a STOP file -- so the next session writes
    # into the same directory instead of starting a sibling.
    run_name: str | None = None
    # 1-based measurement index to start at, matching the progress line `[N/210]`. None means
    # continue after whatever rows are already in this run's behavioural.csv. An explicit
    # number truncates from that index and re-measures.
    start_at: int | None = None
    # Skip measurement and write summary / plots / quality from behavioural.csv already on disk.
    # No model is loaded. Use after a timeout, or to export mid-run results.
    finalize_only: bool = False


class Progress:
    """Periodic progress with an ETA, for phases that run for minutes.

    Reports on a timer rather than every item: a 2,880-patch sweep would otherwise bury the
    interesting output. The first item always reports, so a long phase says something within
    seconds of starting and a stalled run is distinguishable from a slow one.
    """

    def __init__(self, total: int, log, *, every_seconds: float = 15.0, done: int = 0):
        self.total = total
        self.log = log
        self.every = every_seconds
        self.done = done
        self.offset = done
        self.start = time.monotonic()
        self._last_report = 0.0
        self._ticked = False

    @staticmethod
    def _clock(seconds: float) -> str:
        seconds = int(seconds)
        if seconds < 60:
            return f"{seconds}s"
        if seconds < 3600:
            return f"{seconds // 60}m{seconds % 60:02d}s"
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"

    def tick(self, detail: str = "") -> None:
        self.done += 1
        now = time.monotonic()
        first = not self._ticked
        self._ticked = True
        last = self.done == self.total
        if not (first or last or now - self._last_report >= self.every):
            return
        self._last_report = now
        session = max(1, self.done - self.offset) if hasattr(self, "offset") else self.done
        elapsed = now - self.start
        rate = session / elapsed if elapsed > 0 else 0.0
        remaining = (self.total - self.done) / rate if rate > 0 else 0.0
        pct = 100.0 * self.done / self.total if self.total else 100.0
        suffix = f"  {detail}" if detail else ""
        self.log(
            f"    [{self.done:>5}/{self.total}] {pct:3.0f}%  "
            f"elapsed {self._clock(elapsed)}  eta {self._clock(remaining)}"
            f"  {rate:.2f}/s{suffix}"
        )


# --------------------------------------------------------------------------- manifest


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _model_revision(model_id: str) -> str | None:
    """The resolved commit sha of the checkpoint, so a re-run pins the same weights."""
    try:
        from huggingface_hub import model_info

        return model_info(model_id).sha
    except Exception:
        return None


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def build_manifest(cfg: RunConfig, backend: Backend, extra: dict) -> dict:
    described = backend.describe()
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "model": cfg.model_id,
        "model_revision": _model_revision(cfg.model_id),
        "provider": cfg.provider,
        "backend": described,
        "n_layers": described.get("n_layers"),
        "d_model": described.get("d_model"),
        "dtype": cfg.dtype,
        "seed": cfg.seed,
        "generation": {
            "temperature": 0.0,
            # What the backend resolved to, falling back to the request for backends that do
            # not report one. A manifest that records the request rather than the effect
            # cannot be used to tell a truncated run from an incapable model.
            "max_tokens": described.get("max_gen_tokens", cfg.max_gen_tokens),
            "max_tokens_requested": cfg.max_gen_tokens,
            "greedy": True,
        },
        # Recorded because it changes how the run was produced, not what it measured: rows are
        # ordered and scored identically either way, but a reader comparing wall-clock timings
        # across runs needs to know which ones were serial.
        "max_concurrency": cfg.max_concurrency,
        "versions": {
            "interp_engine": _package_version("interp-engine"),
            "openai": _package_version("openai"),
            "anthropic": _package_version("anthropic"),
            "transformers": _package_version("transformers"),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "python": sys.version.split()[0],
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda": torch.version.cuda if torch.cuda.is_available() else None,
            "platform": platform.platform(),
        },
        "data_file": str(cfg.data_file),
        **extra,
    }


# --------------------------------------------------------------------------- experiment 1


def _measurement_row(scenario: Scenario, condition: str, prompt: str, m: Measurement) -> dict:
    """One behavioural row, from any backend.

    ``p_*`` are None on a backend without logprobs. ``accepted_false_proposition`` is the
    primary cross-model outcome precisely because it survives that: it needs only the letter.
    """
    arm = ARMS_BY_NAME[condition]
    by_letter = {"A": (m.p_a, m.p_a_norm), "B": (m.p_b, m.p_b_norm)}
    p_correct, _ = by_letter[scenario.correct_letter]
    p_false, p_false_norm = by_letter[scenario.false_letter]
    # An unparseable response is MISSING, not a refusal. Scoring it False would count a model
    # that never answered as one that correctly rejected the false proposition, which biases
    # every rate downward by however often parsing failed -- silently, and hardest on exactly
    # the runs where parsing is difficult (reasoning models, whose answer must be recovered
    # from generated text rather than read off the first token).
    answered = m.chosen_letter is not None
    return {
        "scenario_id": scenario.id,
        "area": scenario.area,
        "condition": condition,
        "source": arm.source,
        "verb": arm.verb,
        "asserts": arm.asserts,
        "n_prompt_tokens": m.n_prompt_tokens,
        "correct_letter": scenario.correct_letter,
        "false_letter": scenario.false_letter,
        "chosen_letter": m.chosen_letter,
        "correct": (m.chosen_letter == scenario.correct_letter) if answered else None,
        "accepted_false_proposition": (m.chosen_letter == scenario.false_letter) if answered else None,
        "p_correct": p_correct,
        "p_false": p_false,
        "p_false_normalised": p_false_norm,
        "letter_mass": m.letter_mass,
        "parse_ok": m.parse_ok,
        "probability_source": m.probability_source,
        "generated_answer": m.generated,
        "prompt": prompt,
    }


STOP_FILENAME = "STOP"


def measurement_plan(scenarios: list[Scenario], arms: tuple[str, ...] = CONDITIONS) -> list[tuple[Scenario, str]]:
    """The numbered list progress reports: item 1 is scenarios[0] × arms[0]."""
    return [(s, c) for s in scenarios for c in arms]


def format_measurement_plan(scenarios: list[Scenario], arms: tuple[str, ...] = CONDITIONS) -> str:
    """Human-readable index so a stopped run can name the next `start_at`."""
    lines = []
    for i, (scenario, condition) in enumerate(measurement_plan(scenarios, arms), start=1):
        lines.append(f"  {i:3d}  {scenario.id}  {condition}")
    return "\n".join(lines)


def resolve_run_dir(cfg: RunConfig) -> Path:
    """Where this run writes. A named directory is reused across sessions; a timestamp is not."""
    cfg.results_root.mkdir(parents=True, exist_ok=True)
    name = cfg.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = cfg.results_root / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "plots").mkdir(exist_ok=True)
    return path


def _stop_requested(run_dir: Path) -> bool:
    return (run_dir / STOP_FILENAME).exists()


def _write_behavioural_csv(path: Path, slots: list[dict | None]) -> None:
    """Rewrite the CSV from filled slots, fsynced, so a killed kernel keeps every finished row."""
    rows = [row for row in slots if row is not None]
    if not rows:
        return
    tmp = path.with_suffix(".csv.tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)
    with open(path, "rb") as handle:
        os.fsync(handle.fileno())


def _load_behavioural_slots(
    path: Path, pairs: list[tuple[Scenario, str]]
) -> list[dict | None]:
    slots: list[dict | None] = [None] * len(pairs)
    if not path.exists():
        return slots
    frame = pd.read_csv(path)
    index_of = {(scenario.id, condition): i for i, (scenario, condition) in enumerate(pairs)}
    for record in frame.to_dict("records"):
        key = (record["scenario_id"], record["condition"])
        if key in index_of:
            slots[index_of[key]] = record
    return slots


def _first_missing(slots: list[dict | None]) -> int:
    """1-based index of the first unfilled measurement, or len+1 if the plan is complete."""
    for i, row in enumerate(slots):
        if row is None:
            return i + 1
    return len(slots) + 1


def run_behavioural(
    backend: Backend,
    scenarios: list[Scenario],
    cfg: RunConfig,
    progress: Progress | None = None,
    *,
    run_dir: Path | None = None,
) -> pd.DataFrame:
    """Experiment 1, on any backend. The only experiment a closed-weights model can run.

    Every prompt here is independent -- no scenario informs another, and the arms are scored
    separately -- so the order they are measured in carries no information and they can be in
    flight together where the backend allows it. Rows come back in scenario x arm order either
    way, so a run's output does not depend on `max_concurrency`.

    When ``run_dir`` is set, each finished row is written to ``behavioural.csv`` before the next
    prompt starts. A Colab timeout then loses only the in-flight item. Touch ``run_dir/STOP``
    (or interrupt the kernel) to halt after the current measurement; already-written rows stay.
    """
    pairs = measurement_plan(scenarios, cfg.arms)
    csv_path = (run_dir / "behavioural.csv") if run_dir is not None else None
    slots = _load_behavioural_slots(csv_path, pairs) if csv_path is not None else [None] * len(pairs)

    if cfg.start_at is not None:
        start = cfg.start_at
        if start < 1 or start > len(pairs) + 1:
            raise ValueError(f"start_at={start} is outside 1..{len(pairs) + 1}")
        missing = [i + 1 for i in range(start - 1) if slots[i] is None]
        if missing:
            raise ValueError(
                f"start_at={start} but measurements {missing[:8]} are not on disk. "
                "Resume from the first gap, or copy the checkpoint CSV into this run directory."
            )
        for i in range(start - 1, len(slots)):
            slots[i] = None
    else:
        start = _first_missing(slots)

    if csv_path is not None and start <= len(pairs):
        scenario, condition = pairs[start - 1]
        n_have = sum(row is not None for row in slots)
        print(
            f"  checkpoint: {n_have}/{len(pairs)} rows on disk. "
            f"Resuming at {start}/{len(pairs)} ({scenario.id} / {condition}). "
            f"Stop: touch {run_dir / STOP_FILENAME}",
            flush=True,
        )

    remaining = [i for i in range(start - 1, len(pairs)) if slots[i] is None]
    if run_dir is not None and _stop_requested(run_dir) and remaining:
        (run_dir / STOP_FILENAME).unlink()
        print("  cleared leftover STOP from the previous session", flush=True)
    concurrent = (
        cfg.max_concurrency > 1
        and len(remaining) > 1
        and getattr(backend, "concurrency_safe", False)
    )

    def store(index: int, measurement: Measurement) -> None:
        scenario, condition = pairs[index]
        prompt = scenario.prompt(condition)
        slots[index] = _measurement_row(scenario, condition, prompt, measurement)
        if csv_path is not None:
            _write_behavioural_csv(csv_path, slots)
        if progress:
            progress.tick(f"{scenario.id} / {condition}")

    if not remaining:
        pass
    elif concurrent:
        prompts = [pairs[i][0].prompt(pairs[i][1]) for i in remaining]
        index_by_local = {local: remaining[local] for local in range(len(remaining))}

        def on_result(local: int, measurement: Measurement) -> None:
            store(index_by_local[local], measurement)

        measure_many(
            backend, prompts, max_workers=cfg.max_concurrency, on_result=on_result,
        )
    else:
        for index in remaining:
            if run_dir is not None and _stop_requested(run_dir):
                print(
                    f"  STOP at measurement {index + 1}/{len(pairs)} "
                    f"({pairs[index][0].id} / {pairs[index][1]}). "
                    f"{sum(row is not None for row in slots)} rows kept.",
                    flush=True,
                )
                break
            scenario, condition = pairs[index]
            try:
                measurement = backend.measure(scenario.prompt(condition))
            except KeyboardInterrupt:
                if run_dir is not None:
                    (run_dir / STOP_FILENAME).write_text("keyboard interrupt\n")
                    _write_behavioural_csv(csv_path, slots)
                raise
            store(index, measurement)

    rows = [row for row in slots if row is not None]
    if csv_path is not None:
        _write_behavioural_csv(csv_path, slots)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- experiment 2


def run_activations(
    handle: interp.ModelHandle,
    scenarios: list[Scenario],
    contrast: tuple[str, str],
    progress: Progress | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, dict[int, torch.Tensor]]]]:
    """Capture every layer's residual at the final prompt position, for the contrast pair."""
    low, high = contrast
    captured: dict[str, dict[str, dict[int, torch.Tensor]]] = {}
    frames = []
    for scenario in scenarios:
        per_condition = {}
        for condition in contrast:
            token_ids = interp.tokenize_prompt(handle, scenario.prompt(condition))
            per_condition[condition] = interp.capture_residuals(handle, token_ids)
            if progress:
                progress.tick(f"{scenario.id} / {condition}")
        captured[scenario.id] = per_condition
        frame = metrics.activation_divergence(
            {layer: act.numpy() for layer, act in per_condition[low].items()},
            {layer: act.numpy() for layer, act in per_condition[high].items()},
        )
        frame.insert(0, "scenario_id", scenario.id)
        frame.insert(1, "area", scenario.area)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), captured


# --------------------------------------------------------------------------- experiments 3 and 4


def _record(scenario: Scenario, handle: interp.ModelHandle, logits: torch.Tensor, **fields) -> dict:
    probs = interp.letter_probabilities(handle, logits)
    chosen = "A" if probs["p_a"] >= probs["p_b"] else "B"
    return {
        "scenario_id": scenario.id,
        "area": scenario.area,
        "chosen_letter": chosen,
        "accepted_false_proposition": chosen == scenario.false_letter,
        "p_correct": probs[f"p_{scenario.correct_letter.lower()}"],
        "p_false": probs[f"p_{scenario.false_letter.lower()}"],
        "p_false_normalised": probs[f"p_{scenario.false_letter.lower()}_norm"],
        **fields,
    }


def run_interventions(
    handle: interp.ModelHandle,
    scenarios: list[Scenario],
    captured: dict,
    candidates: list[int],
    cfg: RunConfig,
    progress: Progress | None = None,
) -> pd.DataFrame:
    """Bidirectional patching over every layer, plus the zero and random controls.

    ``arm`` distinguishes what was done:

    ``baseline``          the unpatched forward in that condition.
    ``patch_forward``     partner prompt, activation replaced by the control condition's.
    ``patch_reverse``     control prompt, activation replaced by the partner condition's.
    ``control_zero``      the patch machinery installed with scale 0. Must reproduce baseline.
    ``control_random``    a random direction of the same magnitude as the real patch.
    """
    low, high = cfg.contrast
    rng = np.random.default_rng(cfg.seed)
    rows: list[dict] = []
    for scenario in scenarios:
        prompts = {c: interp.tokenize_prompt(handle, scenario.prompt(c)) for c in cfg.contrast}
        acts = captured[scenario.id]
        baselines = {}
        for condition in cfg.contrast:
            logits = interp.next_token_logits(handle, prompts[condition])
            baselines[condition] = logits
            rows.append(_record(scenario, handle, logits, arm="baseline", condition=condition, layer=-1, patch_norm=0.0))

        for layer in range(handle.n_layers):
            forward_delta = acts[low][layer] - acts[high][layer]
            norm = float(torch.linalg.vector_norm(forward_delta))
            if norm == 0.0:
                # Identical activations: the conditions did not differ here, so there is nothing
                # to patch. Recorded rather than skipped, so the sweep stays complete.
                rows.append(
                    _record(scenario, handle, baselines[high], arm="patch_forward",
                            condition=high, layer=layer, patch_norm=0.0)
                )
                rows.append(
                    _record(scenario, handle, baselines[low], arm="patch_reverse",
                            condition=low, layer=layer, patch_norm=0.0)
                )
                if progress:
                    progress.tick(f"{scenario.id} / L{layer} (identical, skipped)")
                continue

            rows.append(
                _record(
                    scenario, handle,
                    interp.patched_next_token_logits(handle, prompts[high], layer, forward_delta),
                    arm="patch_forward", condition=high, layer=layer, patch_norm=norm,
                )
            )
            rows.append(
                _record(
                    scenario, handle,
                    interp.patched_next_token_logits(handle, prompts[low], layer, -forward_delta),
                    arm="patch_reverse", condition=low, layer=layer, patch_norm=norm,
                )
            )
            if progress:
                progress.tick(f"{scenario.id} / L{layer}")

            if layer in candidates:
                rows.append(
                    _record(
                        scenario, handle,
                        interp.patched_next_token_logits(handle, prompts[high], layer, forward_delta, scale=0.0),
                        arm="control_zero", condition=high, layer=layer, patch_norm=0.0,
                    )
                )
                random_direction = torch.from_numpy(rng.normal(size=forward_delta.shape)).float()
                random_delta = random_direction / torch.linalg.vector_norm(random_direction) * norm
                rows.append(
                    _record(
                        scenario, handle,
                        interp.patched_next_token_logits(handle, prompts[high], layer, random_delta),
                        arm="control_random", condition=high, layer=layer, patch_norm=norm,
                    )
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- analysis


def analyse(
    behavioural: pd.DataFrame,
    divergence: pd.DataFrame | None,
    interventions: pd.DataFrame | None,
    candidates: list[int],
    cfg: RunConfig,
) -> dict:
    """Everything the run can conclude, from whatever experiments it was able to run.

    Behavioural analysis works on any backend. The representational and causal sections are
    omitted rather than faked when the backend could not produce them.
    """
    low, high = cfg.contrast

    def per_arm(column: str) -> dict[str, pd.Series]:
        """Per-scenario values for one column, with unanswered rows dropped.

        Dropping rather than filling is what keeps a parse failure out of the numerator *and*
        the denominator. Paired tests then intersect on scenarios both arms answered, so a
        comparison is never made against a scenario only one side scored.
        """
        out = {}
        for condition, group in behavioural.groupby("condition"):
            series = group.set_index("scenario_id")[column].dropna()
            if not series.empty:
                out[str(condition)] = series
        return out

    accepted = per_arm("accepted_false_proposition")
    p_false = per_arm("p_false_normalised")

    summary: dict = {
        "arms_run": list(cfg.arms),
        "behavioural": {
            "fpar_by_arm": {
                arm: float(series.astype(bool).mean()) for arm, series in sorted(accepted.items())
            },
            # The denominator each rate was computed over. A rate on 17 of 30 scenarios is a
            # different claim from one on 30, and the difference must not be invisible.
            "n_scored_by_arm": {arm: int(series.size) for arm, series in sorted(accepted.items())},
            "accuracy_by_arm": {
                str(condition): float(group["correct"].dropna().astype(bool).mean())
                for condition, group in behavioural.groupby("condition")
                if group["correct"].notna().any()
            },
            "parse_failures": int((~behavioural["parse_ok"].astype(bool)).sum()),
            "n_measurements": int(len(behavioural)),
        },
    }

    # The headline pair, whichever contrast this run used.
    if low in accepted and high in accepted:
        fpar = metrics.false_proposition_acceptance(accepted[low], accepted[high])
        summary["behavioural"]["headline_contrast"] = {
            "low_authority_arm": low,
            "high_authority_arm": high,
            **fpar.as_dict(),
        }
        summary["behavioural"]["authority_deference_delta"] = fpar.delta
        if low in p_false and high in p_false:
            summary["behavioural"]["p_false_paired"] = metrics.paired_difference(
                p_false[low], p_false[high]
            ).as_dict()

    # Planned contrasts, binary (works everywhere) and continuous (where probabilities exist).
    binary_contrasts = metrics.planned_contrasts(accepted, PLANNED_CONTRASTS, binary=True)
    if not binary_contrasts.empty:
        summary["planned_contrasts_binary"] = binary_contrasts.to_dict("records")
    if len(p_false) >= 2:
        continuous_contrasts = metrics.planned_contrasts(p_false, PLANNED_CONTRASTS, binary=False)
        if not continuous_contrasts.empty:
            summary["planned_contrasts_continuous"] = continuous_contrasts.to_dict("records")

    # The 2x2. Continuous where available, otherwise on the binary outcome.
    source_of_truth = p_false if len(p_false) >= 4 else accepted
    cells = {
        key: source_of_truth[arm] for key, arm in FACTORIAL_CELLS.items() if arm in source_of_truth
    }
    summary["factorial"] = metrics.factorial_effects(cells)
    summary["factorial"]["measure"] = "p_false_normalised" if len(p_false) >= 4 else "accepted_false_proposition"

    if divergence is not None:
        summary["representational"] = {
            "contrast": [low, high],
            "candidate_layers": candidates,
            "max_mean_relative_l2": float(divergence["mean_relative_l2"].max()),
            "layer_ranking": [
                int(x) for x in divergence.sort_values("mean_relative_l2", ascending=False)["layer"]
            ],
        }

    if interventions is not None and not interventions.empty:
        baseline = interventions[interventions.arm == "baseline"].set_index(["scenario_id", "condition"])
        causal = {}
        for arm, condition in (("patch_forward", high), ("patch_reverse", low)):
            arm_rows = interventions[interventions.arm == arm]
            if arm_rows.empty:
                continue
            base = baseline.xs(condition, level="condition")["p_false_normalised"]
            per_layer = []
            for layer, group in arm_rows.groupby("layer"):
                aligned = group.set_index("scenario_id")["p_false_normalised"]
                common = aligned.index.intersection(base.index)
                paired = metrics.paired_difference(base.loc[common], aligned.loc[common], n_boot=2000)
                per_layer.append({"layer": int(layer), "arm": arm, **paired.as_dict()})
            causal[arm] = per_layer
        summary["causal"] = causal

        controls = {}
        for arm in ("control_zero", "control_random"):
            arm_rows = interventions[interventions.arm == arm]
            if arm_rows.empty:
                continue
            base = baseline.xs(high, level="condition")["p_false_normalised"]
            merged = arm_rows.set_index("scenario_id")["p_false_normalised"]
            deltas = (merged - base.reindex(merged.index)).abs()
            controls[arm] = {
                "n": int(deltas.size),
                "max_abs_change_in_p_false": float(deltas.max()),
                "mean_abs_change_in_p_false": float(deltas.mean()),
            }
        summary["intervention_controls"] = controls

    return summary


# --------------------------------------------------------------------------- runner


def write_run_artifacts(
    run_dir: Path,
    cfg: RunConfig,
    *,
    backend=None,
    timings: dict | None = None,
    stale: list | None = None,
    logit_check: dict | None = None,
    mechanistic: bool = False,
    candidates: list[int] | None = None,
    verbose: bool = True,
    stopped_early: bool = False,
) -> dict:
    """Write summary, plots, quality, and manifest from whatever CSVs are in ``run_dir``.

    No model is loaded. Call after a timeout, after touching STOP, or on a finished run you
    want to re-plot. Partial behavioural.csv is allowed; the quality gate will say so.
    """
    from . import plots, quality

    run_dir = Path(run_dir)
    csv_path = run_dir / "behavioural.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No behavioural.csv in {run_dir}")
    behavioural = pd.read_csv(csv_path)
    if behavioural.empty:
        raise ValueError(f"{csv_path} is empty -- nothing to export.")

    def log(message: str) -> None:
        if verbose:
            print(message, flush=True)

    div_path, int_path = run_dir / "activation_analysis.csv", run_dir / "interventions.csv"
    divergence = pd.read_csv(div_path) if div_path.exists() else None
    interventions = pd.read_csv(int_path) if int_path.exists() else None
    candidates = candidates or []
    if divergence is not None and not candidates:
        candidates = metrics.candidate_layers(divergence, k=cfg.n_candidate_layers)

    summary = analyse(behavioural, divergence, interventions, candidates, cfg)
    n_plan = None
    if cfg.limit:
        n_plan = cfg.limit * len(cfg.arms)
    else:
        try:
            n_plan = len(load_scenarios(cfg.data_file)) * len(cfg.arms)
        except OSError:
            n_plan = None
    if n_plan is not None:
        summary["behavioural"]["n_planned"] = n_plan
        summary["behavioural"]["complete"] = int(len(behavioural)) >= n_plan
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    extra = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
        "n_scenarios": int(behavioural["scenario_id"].nunique()),
        "arms": list(cfg.arms),
        "contrast": list(cfg.contrast),
        "mechanistic": mechanistic,
        "enable_thinking": cfg.enable_thinking,
        "candidate_layers": candidates,
        "logit_path_check": logit_check or {"checked": False, "reason": "finalize without model"},
        "timings_seconds": timings or {},
        "stale_scenarios": [s.id if hasattr(s, "id") else s for s in (stale or [])],
        "cue_strings": {a: ARMS_BY_NAME[a].cue for a in cfg.arms},
        "stopped_early": stopped_early,
        "n_measurements_written": int(len(behavioural)),
    }
    if backend is not None:
        manifest = build_manifest(cfg, backend, extra)
    else:
        previous = {}
        if (run_dir / "manifest.json").exists():
            previous = json.loads((run_dir / "manifest.json").read_text())
        extra["model"] = cfg.model_id
        extra["provider"] = cfg.provider
        extra["experiment_version"] = EXPERIMENT_VERSION
        extra["finalized_without_model"] = True
        previous.update(extra)
        manifest = previous

    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=float))
    plots.write_all(behavioural, divergence, interventions, run_dir / "plots", cfg)
    report = quality.run_checks(behavioural, divergence, interventions, summary, manifest)
    (run_dir / "quality_report.json").write_text(json.dumps(report, indent=2))
    log("")
    log(quality.format_report(report))
    log("")
    if report["overall"] == "fail":
        log(
            "quality gate: FAIL -- at least one check failed outright. Read the report "
            "above before drawing any conclusion from behavioural.csv, the plots, or "
            "summary.json; the run completed, but its numbers are flagged as untrustworthy."
        )
    n_written = len(behavioural)
    n_note = f"{n_written}" + (f"/{n_plan}" if n_plan else "")
    log(f"Wrote artifacts from {n_note} measurements in {run_dir}")
    return {"summary": summary, "report": report, "manifest": manifest}


def finalize_run(run_dir: Path | str, cfg: RunConfig | None = None, *, verbose: bool = True) -> Path:
    """Rebuild plots and tables from a run directory. Does not load a model."""
    run_dir = Path(run_dir)
    if cfg is None:
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No manifest.json in {run_dir}; pass a RunConfig so arms and contrast are known."
            )
        recorded = json.loads(manifest_path.read_text())
        config = recorded.get("config") or {}
        cfg = RunConfig(
            model_id=recorded.get("model") or config.get("model_id", RunConfig.model_id),
            provider=recorded.get("provider") or config.get("provider", "local"),
            enable_thinking=recorded.get("enable_thinking", False),
            mechanistic=recorded.get("mechanistic", False),
            run_name=run_dir.name,
            results_root=run_dir.parent,
        )
        if "arms" in recorded:
            cfg.arms = tuple(recorded["arms"])
        if "contrast" in recorded:
            cfg.contrast = tuple(recorded["contrast"])
    write_run_artifacts(run_dir, cfg, verbose=verbose, stopped_early=True)
    return run_dir


def run_all(cfg: RunConfig | None = None, *, verbose: bool = True) -> Path:
    from . import plots, quality  # noqa: F401 -- kept so a notebook that imported via run_all still works

    cfg = cfg or RunConfig()
    scenarios = load_scenarios(cfg.data_file)
    today = datetime.now(timezone.utc).date().isoformat()
    stale = stale_scenarios(scenarios, today)
    if cfg.limit:
        scenarios = scenarios[: cfg.limit]

    timings: dict[str, float] = {}
    run_dir = resolve_run_dir(cfg)

    def log(message: str) -> None:
        if verbose:
            print(message, flush=True)

    class phase:
        """Times a phase and logs how long it took, so a Colab run is predictable."""

        def __init__(self, name: str):
            self.name = name

        def __enter__(self):
            self.start = time.monotonic()
            return self

        def __exit__(self, *exc):
            timings[self.name] = round(time.monotonic() - self.start, 1)
            log(f"  ... {self.name} took {timings[self.name]}s")
            return False

    if stale:
        log(
            "WARNING: the stored ground truth for "
            + ", ".join(s.id for s in stale)
            + " is superseded by law in force today. Those scenarios are still scored; see "
            "their 'note' field before reading the result."
        )

    if cfg.finalize_only:
        log(f"Finalize only -- no model load. Reading {run_dir / 'behavioural.csv'}")
        write_run_artifacts(run_dir, cfg, stale=stale, verbose=verbose, stopped_early=True)
        return run_dir

    options = dict(cfg.provider_options)
    if cfg.provider == "local":
        options.update(cfg.extra_load_kwargs)
        options.setdefault("backend", cfg.backend)
        options.setdefault("enable_thinking", cfg.enable_thinking)
        if cfg.dtype is not None:
            options.setdefault("dtype", cfg.dtype)
    log(f"Opening {cfg.provider} backend for {cfg.model_id} ...")
    log(f"Results directory: {run_dir}")
    backend = None
    backend = BackendSpec(
        kind=cfg.provider, model_id=cfg.model_id, options=options,
        max_gen_tokens=cfg.max_gen_tokens,
    ).build()

    divergence = per_scenario = interventions = None
    candidates: list[int] = []
    logit_check: dict = {"checked": False, "reason": f"{cfg.provider} backend"}
    stopped_early = False

    try:
        mechanistic = (
            cfg.mechanistic
            and isinstance(backend, LocalBackend)
            and backend.supports_mechanistic_now
        )
        if isinstance(backend, LocalBackend):
            if not cfg.mechanistic:
                log("NOTE: mechanistic=False -- running experiment 1 only.")
            if backend.handle.enable_thinking and not backend.handle.has_reasoning_mode:
                log("NOTE: enable_thinking was requested but this model has no reasoning mode; "
                    "it reads no such template control, so the run is unchanged.")
            elif backend.response_mode == "generate":
                log("NOTE: reasoning is ON, so the answer is parsed from the completion rather "
                    "than read from the first token, and experiments 2-4 are skipped -- the "
                    "answer no longer sits at the final prompt position for patching to reach.")
            if cfg.mechanistic and not backend.handle.can_capture:
                # Distinct from the reasoning case above: this one is the backend, and it is
                # silently recoverable by re-running on eager, so name the remedy.
                log(f"NOTE: the {cfg.backend!r} engine backend replays CUDA graphs and runs no "
                    "forward hooks, so nothing can be captured or patched through it. "
                    "Experiments 2-4 are skipped; re-run with backend='eager' to get them.")
        else:
            log(
                f"NOTE: the {cfg.provider} backend is behavioural-only. Closed weights cannot be "
                "captured or patched, so experiments 2-4 are skipped for this run."
            )

        if mechanistic:
            probe = interp.tokenize_prompt(backend.handle, scenarios[0].prompt(cfg.arms[0]))
            logit_check = interp.verify_logit_path(backend.handle, probe)
            log(f"Logit path check: {logit_check}")

        n_measurements = len(scenarios) * len(cfg.arms)
        csv_path = run_dir / "behavioural.csv"
        already = 0
        if csv_path.exists():
            already = len(pd.read_csv(csv_path))
        start_hint = cfg.start_at or (already + 1)
        log(f"Experiment 1: behaviour over {len(scenarios)} scenarios x {len(cfg.arms)} arms "
            f"= {n_measurements} measurements ...")
        progress = (
            Progress(
                n_measurements, log,
                every_seconds=cfg.progress_every_seconds,
                done=max(0, start_hint - 1),
            )
            if verbose else None
        )
        try:
            with phase("behavioural"):
                behavioural = run_behavioural(
                    backend, scenarios, cfg, progress, run_dir=run_dir,
                )
        except KeyboardInterrupt:
            stopped_early = True
            log("Interrupted -- writing artifacts from rows already on disk.")
            if csv_path.exists():
                behavioural = pd.read_csv(csv_path)
            else:
                raise
        else:
            if _stop_requested(run_dir) or len(behavioural) < n_measurements:
                stopped_early = True

        if mechanistic and not stopped_early:
            low, high = cfg.contrast
            log(f"Experiment 2: capturing resid_post across {backend.handle.n_layers} layers "
                f"for {low} vs {high} ...")
            n_captures = len(scenarios) * len(cfg.contrast)
            with phase("activations"):
                per_scenario, captured = run_activations(
                    backend.handle, scenarios, cfg.contrast,
                    Progress(n_captures, log, every_seconds=cfg.progress_every_seconds) if verbose else None,
                )
            divergence = metrics.summarise_divergence(per_scenario)
            divergence.to_csv(run_dir / "activation_analysis.csv", index=False)
            per_scenario.to_csv(run_dir / "activation_per_scenario.csv", index=False)
            candidates = metrics.candidate_layers(divergence, k=cfg.n_candidate_layers)
            log(f"Candidate layers (largest divergence, not yet a mechanism): {candidates}")

            # Two patches per layer per scenario, plus two controls at each candidate layer.
            n_patches = len(scenarios) * backend.handle.n_layers
            n_controls = len(scenarios) * len(candidates)
            log(f"Experiments 3 and 4: bidirectional patching across {backend.handle.n_layers} "
                f"layers, plus controls -- {n_patches} layer-steps "
                f"({2 * n_patches + 2 * n_controls} forwards) ...")
            with phase("interventions"):
                interventions = run_interventions(
                    backend.handle, scenarios, captured, candidates, cfg,
                    Progress(n_patches, log, every_seconds=cfg.progress_every_seconds) if verbose else None,
                )
            interventions.to_csv(run_dir / "interventions.csv", index=False)

            if cfg.save_activations:
                np.savez_compressed(
                    run_dir / "activations.npz",
                    **{
                        f"{sid}|{cond}|{layer}": act.numpy()
                        for sid, conds in captured.items()
                        for cond, layers in conds.items()
                        for layer, act in layers.items()
                    },
                )
        elif mechanistic and stopped_early:
            log("Skipping experiments 2-4 because the behavioural sweep did not finish.")

        write_run_artifacts(
            run_dir, cfg,
            backend=backend,
            timings=timings,
            stale=stale,
            logit_check=logit_check,
            mechanistic=mechanistic,
            candidates=candidates,
            verbose=verbose,
            stopped_early=stopped_early,
        )
        log(f"Done. Results in {run_dir}")
        return run_dir
    finally:
        if backend is not None:
            backend.shutdown()


# --------------------------------------------------------------------------- sweep


def device_memory_gb() -> float | None:
    """GB currently reserved on the card, or None on CPU."""
    if not torch.cuda.is_available():
        return None
    return torch.cuda.memory_reserved(0) / 1e9


def free_device_memory() -> float | None:
    """Release whatever the last model left on the card, and report what came back.

    ``shutdown()`` tells the engine to let go, but Python still holds the model until the last
    reference dies, and the allocator still holds the freed blocks until told to release them.
    Loading a 12B and then a 14B on one card fails on either omission -- and it fails as an OOM
    during the *second* load, which reads like the second model being too big rather than the
    first still being resident.

    Returns the memory still reserved afterwards so a sweep can show the release happening
    rather than assume it. A figure that does not fall back near zero between models is the
    warning that the next load is about to OOM.
    """
    gc.collect()
    if not torch.cuda.is_available():
        return None
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return device_memory_gb()


def sweep_label(cfg: RunConfig) -> str:
    """A name that distinguishes runs of one checkpoint that differ only in configuration.

    A reasoning-on and a reasoning-off run of the same model are two different measurements and
    must not collide in a results mapping.
    """
    if cfg.provider != "local":
        return cfg.model_id
    return f"{cfg.model_id} ({'thinking' if cfg.enable_thinking else 'no thinking'})"


def run_sweep(configs: "list[RunConfig]", *, verbose: bool = True) -> dict[str, Path]:
    """Run several configurations in one session, one model resident at a time.

    A failure is reported and the sweep continues: one model being unavailable should not cost
    the results of the others, and a partial sweep is still a comparison. Order the list so the
    run most likely to fail comes first -- a new code path is worth discovering in the first two
    minutes rather than after an hour of GPU time.
    """
    results: dict[str, Path] = {}
    for i, cfg in enumerate(configs, start=1):
        label = sweep_label(cfg)
        if verbose:
            print(f"\n{'=' * 74}\n[{i}/{len(configs)}] {label}\n{'=' * 74}", flush=True)
        # Never let one completed run overwrite another. Configs in a real sweep differ, so
        # labels differ -- but a repeated config costs GPU time either way and losing its
        # result silently is the wrong failure.
        if label in results:
            label = f"{label} #{sum(1 for k in results if k.startswith(label)) + 1}"
        try:
            results[label] = run_all(cfg, verbose=verbose)
        except Exception as exc:  # noqa: BLE001 - a sweep must survive one model failing
            print(f"FAILED {label}: {type(exc).__name__}: {exc}", flush=True)
        finally:
            before = device_memory_gb()
            after = free_device_memory()
            if verbose and after is not None:
                note = "" if after < 2.0 else "  <-- did not release; the next load may OOM"
                print(f"  GPU reserved: {before:.1f} GB -> {after:.1f} GB{note}", flush=True)
    if verbose:
        print(f"\nSweep complete: {len(results)}/{len(configs)} succeeded.", flush=True)
        for label, path in results.items():
            print(f"  {label:44s} {path.name}", flush=True)
    return results


def compare_runs(run_dirs: dict[str, Path]) -> pd.DataFrame:
    """FPAR per arm across runs -- the one measure every backend can report.

    Probabilities are not comparable across backends: the Anthropic API exposes none, and a
    reasoning run is parsed from text rather than read from logits. The binary acceptance rate
    needs only the answer letter, which is why it is the cross-model measure.
    """
    rows = []
    for label, path in run_dirs.items():
        summary = json.loads((Path(path) / "summary.json").read_text())
        rows.append({"model": label, **summary["behavioural"]["fpar_by_arm"]})
    frame = pd.DataFrame(rows).set_index("model")
    return frame[[a for a in CONDITIONS if a in frame.columns]]
