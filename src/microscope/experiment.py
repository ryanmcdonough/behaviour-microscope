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
from .scenarios import CONDITIONS, Scenario, load_scenarios, stale_scenarios

EXPERIMENT_VERSION = "authority_v1"
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class RunConfig:
    model_id: str = "google/gemma-2-2b-it"
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


def build_manifest(cfg: RunConfig, handle: interp.ModelHandle, extra: dict) -> dict:
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "model": cfg.model_id,
        "model_revision": _model_revision(cfg.model_id),
        "backend_requested": cfg.backend,
        "backend_class": handle.backend,
        "n_layers": handle.n_layers,
        "d_model": handle.d_model,
        "dtype": cfg.dtype,
        "seed": cfg.seed,
        "generation": {"temperature": 0.0, "max_tokens": cfg.max_gen_tokens, "greedy": True},
        "versions": {
            "interp_engine": _package_version("interp-engine"),
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


def run_behavioural(handle: interp.ModelHandle, scenarios: list[Scenario], cfg: RunConfig) -> pd.DataFrame:
    rows = []
    for scenario in scenarios:
        for condition in CONDITIONS:
            prompt = scenario.prompt(condition)
            token_ids = interp.tokenize_prompt(handle, prompt)
            logits = interp.next_token_logits(handle, token_ids)
            probs = interp.letter_probabilities(handle, logits)
            generated = interp.generate_answer(handle, token_ids, max_tokens=cfg.max_gen_tokens)
            chosen = "A" if probs["p_a"] >= probs["p_b"] else "B"
            rows.append(
                {
                    "scenario_id": scenario.id,
                    "area": scenario.area,
                    "condition": condition,
                    "n_prompt_tokens": len(token_ids),
                    "correct_letter": scenario.correct_letter,
                    "false_letter": scenario.false_letter,
                    "chosen_letter": chosen,
                    "correct": chosen == scenario.correct_letter,
                    "accepted_false_proposition": chosen == scenario.false_letter,
                    "p_correct": probs[f"p_{scenario.correct_letter.lower()}"],
                    "p_false": probs[f"p_{scenario.false_letter.lower()}"],
                    "p_false_normalised": probs[f"p_{scenario.false_letter.lower()}_norm"],
                    "letter_mass": probs["letter_mass"],
                    "generated_answer": generated.strip(),
                    "prompt": prompt,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- experiment 2


def run_activations(
    handle: interp.ModelHandle, scenarios: list[Scenario]
) -> tuple[pd.DataFrame, dict[str, dict[str, dict[int, torch.Tensor]]]]:
    """Capture every layer's residual at the final prompt position, in both conditions."""
    captured: dict[str, dict[str, dict[int, torch.Tensor]]] = {}
    frames = []
    for scenario in scenarios:
        per_condition = {}
        for condition in CONDITIONS:
            token_ids = interp.tokenize_prompt(handle, scenario.prompt(condition))
            per_condition[condition] = interp.capture_residuals(handle, token_ids)
        captured[scenario.id] = per_condition
        frame = metrics.activation_divergence(
            {layer: act.numpy() for layer, act in per_condition["control"].items()},
            {layer: act.numpy() for layer, act in per_condition["partner"].items()},
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
) -> pd.DataFrame:
    """Bidirectional patching over every layer, plus the zero and random controls.

    ``arm`` distinguishes what was done:

    ``baseline``          the unpatched forward in that condition.
    ``patch_forward``     partner prompt, activation replaced by the control condition's.
    ``patch_reverse``     control prompt, activation replaced by the partner condition's.
    ``control_zero``      the patch machinery installed with scale 0. Must reproduce baseline.
    ``control_random``    a random direction of the same magnitude as the real patch.
    """
    rng = np.random.default_rng(cfg.seed)
    rows: list[dict] = []
    for scenario in scenarios:
        prompts = {c: interp.tokenize_prompt(handle, scenario.prompt(c)) for c in CONDITIONS}
        acts = captured[scenario.id]
        baselines = {}
        for condition in CONDITIONS:
            logits = interp.next_token_logits(handle, prompts[condition])
            baselines[condition] = logits
            rows.append(_record(scenario, handle, logits, arm="baseline", condition=condition, layer=-1, patch_norm=0.0))

        for layer in range(handle.n_layers):
            forward_delta = acts["control"][layer] - acts["partner"][layer]
            norm = float(torch.linalg.vector_norm(forward_delta))
            if norm == 0.0:
                # Identical activations: the conditions did not differ here, so there is nothing
                # to patch. Recorded rather than skipped, so the sweep stays complete.
                rows.append(
                    _record(scenario, handle, baselines["partner"], arm="patch_forward",
                            condition="partner", layer=layer, patch_norm=0.0)
                )
                rows.append(
                    _record(scenario, handle, baselines["control"], arm="patch_reverse",
                            condition="control", layer=layer, patch_norm=0.0)
                )
                continue

            rows.append(
                _record(
                    scenario, handle,
                    interp.patched_next_token_logits(handle, prompts["partner"], layer, forward_delta),
                    arm="patch_forward", condition="partner", layer=layer, patch_norm=norm,
                )
            )
            rows.append(
                _record(
                    scenario, handle,
                    interp.patched_next_token_logits(handle, prompts["control"], layer, -forward_delta),
                    arm="patch_reverse", condition="control", layer=layer, patch_norm=norm,
                )
            )

            if layer in candidates:
                rows.append(
                    _record(
                        scenario, handle,
                        interp.patched_next_token_logits(handle, prompts["partner"], layer, forward_delta, scale=0.0),
                        arm="control_zero", condition="partner", layer=layer, patch_norm=0.0,
                    )
                )
                random_direction = torch.from_numpy(rng.normal(size=forward_delta.shape)).float()
                random_delta = random_direction / torch.linalg.vector_norm(random_direction) * norm
                rows.append(
                    _record(
                        scenario, handle,
                        interp.patched_next_token_logits(handle, prompts["partner"], layer, random_delta),
                        arm="control_random", condition="partner", layer=layer, patch_norm=norm,
                    )
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- analysis


def analyse(
    behavioural: pd.DataFrame,
    divergence: pd.DataFrame,
    interventions: pd.DataFrame,
    candidates: list[int],
) -> dict:
    wide = behavioural.pivot(index="scenario_id", columns="condition")
    accepted = wide["accepted_false_proposition"]
    fpar = metrics.false_proposition_acceptance(accepted["control"], accepted["partner"])
    p_false = wide["p_false_normalised"]
    continuous = metrics.paired_difference(p_false["control"], p_false["partner"])
    accuracy = {
        condition: float(behavioural.loc[behavioural.condition == condition, "correct"].mean())
        for condition in CONDITIONS
    }

    baseline = interventions[interventions.arm == "baseline"].set_index(["scenario_id", "condition"])
    summary = {}
    for arm, condition in (("patch_forward", "partner"), ("patch_reverse", "control")):
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
        summary[arm] = per_layer

    controls = {}
    for arm in ("control_zero", "control_random"):
        arm_rows = interventions[interventions.arm == arm]
        if arm_rows.empty:
            continue
        base = baseline.xs("partner", level="condition")["p_false_normalised"]
        merged = arm_rows.set_index("scenario_id")["p_false_normalised"]
        deltas = (merged - base.reindex(merged.index)).abs()
        controls[arm] = {
            "n": int(deltas.size),
            "max_abs_change_in_p_false": float(deltas.max()),
            "mean_abs_change_in_p_false": float(deltas.mean()),
        }

    return {
        "behavioural": {
            "fpar": fpar.as_dict(),
            "authority_deference_delta": fpar.delta,
            "p_false_paired": continuous.as_dict(),
            "accuracy": accuracy,
        },
        "representational": {
            # The layers the intervention controls were run at. Candidates, not mechanisms.
            "candidate_layers": candidates,
            "max_mean_relative_l2": float(divergence["mean_relative_l2"].max()),
            "layer_ranking": [int(x) for x in divergence.sort_values("mean_relative_l2", ascending=False)["layer"]],
        },
        "causal": summary,
        "intervention_controls": controls,
    }


# --------------------------------------------------------------------------- runner


def run_all(cfg: RunConfig | None = None, *, verbose: bool = True) -> Path:
    from . import plots

    cfg = cfg or RunConfig()
    scenarios = load_scenarios(cfg.data_file)
    today = datetime.now(timezone.utc).date().isoformat()
    stale = stale_scenarios(scenarios, today)
    if cfg.limit:
        scenarios = scenarios[: cfg.limit]

    timings: dict[str, float] = {}

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

    log(f"Loading {cfg.model_id} through interp-engine (backend={cfg.backend}) ...")
    handle = interp.open_model(cfg.model_id, backend=cfg.backend, dtype=cfg.dtype, **cfg.extra_load_kwargs)
    run_dir = cfg.results_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)

    try:
        probe = interp.tokenize_prompt(handle, scenarios[0].prompt("control"))
        logit_check = interp.verify_logit_path(handle, probe)
        log(f"Logit path check: {logit_check}")

        log(f"Experiment 1: behaviour over {len(scenarios)} scenarios x {len(CONDITIONS)} conditions ...")
        with phase("behavioural"):
            behavioural = run_behavioural(handle, scenarios, cfg)
        behavioural.to_csv(run_dir / "behavioural.csv", index=False)

        log(f"Experiment 2: capturing resid_post across {handle.n_layers} layers ...")
        with phase("activations"):
            per_scenario, captured = run_activations(handle, scenarios)
        divergence = metrics.summarise_divergence(per_scenario)
        divergence.to_csv(run_dir / "activation_analysis.csv", index=False)
        per_scenario.to_csv(run_dir / "activation_per_scenario.csv", index=False)
        candidates = metrics.candidate_layers(divergence, k=cfg.n_candidate_layers)
        log(f"Candidate layers (largest divergence, not yet a mechanism): {candidates}")

        log("Experiments 3 and 4: bidirectional patching across every layer, plus controls ...")
        with phase("interventions"):
            interventions = run_interventions(handle, scenarios, captured, candidates, cfg)
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

        summary = analyse(behavioural, divergence, interventions, candidates)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        manifest = build_manifest(
            cfg, handle,
            {
                "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
                "n_scenarios": len(scenarios),
                "candidate_layers": candidates,
                "logit_path_check": logit_check,
                "timings_seconds": timings,
                "stale_scenarios": [s.id for s in stale],
            },
        )
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        plots.write_all(behavioural, divergence, interventions, run_dir / "plots")
        log(f"Done. Results in {run_dir}")
        return run_dir
    finally:
        handle.shutdown()
