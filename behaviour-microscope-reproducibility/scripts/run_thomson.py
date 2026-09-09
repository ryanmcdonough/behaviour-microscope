#!/usr/bin/env python
"""Headless Thomson-1.0-Small runs, for a rented box rather than a notebook.

The notebook is the wrong shape for a machine you SSH into: it needs a browser attached, and a
dropped connection takes the kernel with it. This is the same sweep as a script, so it can run
under tmux or nohup and survive the laptop closing.

    python scripts/run_thomson.py                  # reasoning on, the re-run that matters
    python scripts/run_thomson.py --plain          # reasoning off, with the mechanistic sweep
    python scripts/run_thomson.py --both           # one after the other, one model resident
    python scripts/run_thomson.py --preflight      # one prompt, then exit

Weights are 70.2 GB. Point HF_HOME at a persistent volume before the first run or you will pay
for that download again every time the box is recycled:

    export HF_HOME=/workspace/hf

Each measurement is written to results/<run-name>/behavioural.csv before the next starts.
Resume with the same --run-name. Touch that folder's STOP file to halt after the current
item; --finalize rebuilds plots from the CSV without loading the model.

    python scripts/run_thomson.py --start-at 48
    python scripts/run_thomson.py --finalize
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

MODEL_ID = "thomsonreuters/Thomson-1.0-Small"
WEIGHTS_GB = 70.2


def describe_hardware() -> str:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device. This model needs an 80 GB card; see --help.")
    props = torch.cuda.get_device_properties(0)
    total = props.total_memory / 1e9
    print(f"{props.name}  |  {total:.1f} GB  |  compute {props.major}.{props.minor}")
    if total - WEIGHTS_GB < 4:
        raise SystemExit(
            f"This card has {total:.0f} GB and the weights are {WEIGHTS_GB} GB. "
            "Use an 80 GB card (A100-80GB or H100-80GB)."
        )
    print(f"Headroom after weights: ~{total - WEIGHTS_GB:.0f} GB")
    print(f"HF_HOME: {os.environ.get('HF_HOME', '(unset -- weights will not persist)')}")
    return "bfloat16" if torch.cuda.is_bf16_supported() else "float16"


def build(thinking: bool, dtype: str, mechanistic: bool, backend: str | None,
          concurrency: int, *, run_name: str, results_root: Path | None,
          start_at: int | None, finalize_only: bool):
    from microscope.experiment import RunConfig

    # A reasoning run captures nothing, so it does not need the eager backend's forward hooks.
    # The plain run does: capture and patching only work where Python hooks run.
    resolved = backend or ("vllm-generate" if thinking else "eager")
    name = run_name
    if run_name == "thomson-1-thinking" and not thinking:
        name = "thomson-1-plain"
    kwargs = dict(
        model_id=MODEL_ID,
        provider="local",
        backend=resolved,
        dtype=dtype,
        extra_load_kwargs={} if resolved.startswith("vllm") else {"device": "cuda"},
        n_candidate_layers=4,
        enable_thinking=thinking,
        mechanistic=mechanistic,
        max_concurrency=concurrency,
        run_name=name,
        start_at=start_at,
        finalize_only=finalize_only,
    )
    if results_root is not None:
        kwargs["results_root"] = Path(results_root)
    return RunConfig(**kwargs)


def preflight(cfg) -> None:
    """One prompt, end to end. Cheap insurance against discovering a mis-parse at hour five."""
    from microscope.backends import BackendSpec
    from microscope.scenarios import load_scenarios

    opts = dict(cfg.extra_load_kwargs)
    opts.update(backend=cfg.backend, enable_thinking=cfg.enable_thinking, dtype=cfg.dtype)
    backend = BackendSpec(kind="local", model_id=MODEL_ID, options=opts,
                          max_gen_tokens=cfg.max_gen_tokens).build()
    try:
        scenario = load_scenarios()[0]
        m = backend.measure(scenario.prompt("floor"))
        print(f"  engine backend : {backend.handle.backend}")
        print(f"  can capture    : {backend.handle.can_capture}")
        print(f"  response mode  : {backend.response_mode}")
        print(f"  gen budget     : {backend.max_gen_tokens}")
        print(f"  parsed letter  : {m.chosen_letter}  (expected {scenario.correct_letter})")
        print(f"  parse_ok       : {m.parse_ok}  source: {m.probability_source}")
        print(f"  completion tail: {m.generated[-200:]!r}")
        if not m.parse_ok:
            raise SystemExit("Preflight: no answer parsed. Do not start the full run.")
        if m.probability_source == "text_truncated":
            raise SystemExit("Preflight: truncated mid-reasoning. Raise max_gen_tokens first.")
        print("  Preflight OK.")
    finally:
        backend.shutdown()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plain", action="store_true", help="reasoning off (mechanistic sweep runs)")
    ap.add_argument("--both", action="store_true", help="reasoning off, then on")
    ap.add_argument("--no-mechanistic", action="store_true", help="experiment 1 only, far faster")
    ap.add_argument("--backend", default=None,
                    help="override the engine backend (eager, vllm-generate, vllm)")
    ap.add_argument("--concurrency", type=int, default=16,
                    help="measurements in flight at once on the vLLM path (default 16; "
                         "1 restores the serial behaviour earlier runs used)")
    ap.add_argument("--run-name", default="thomson-1-thinking",
                    help="stable results folder name, reused across resumes")
    ap.add_argument("--results-root", default=None,
                    help="directory for run folders (default: <repo>/results)")
    ap.add_argument("--start-at", type=int, default=None,
                    help="1-based measurement index to resume at (default: after rows on disk)")
    ap.add_argument("--finalize", action="store_true",
                    help="rebuild plots/tables from behavioural.csv; do not load the model")
    ap.add_argument("--preflight", action="store_true", help="one prompt, then exit")
    ap.add_argument("--skip-preflight", action="store_true")
    args = ap.parse_args()

    dtype = "bfloat16" if args.finalize else describe_hardware()
    from microscope.experiment import run_sweep

    mechanistic = not args.no_mechanistic
    if args.both:
        wanted = [False, True]
    elif args.plain:
        wanted = [False]
    else:
        wanted = [True]
    configs = [
        build(
            t, dtype, mechanistic, args.backend, args.concurrency,
            run_name=args.run_name,
            results_root=Path(args.results_root) if args.results_root else None,
            start_at=args.start_at,
            finalize_only=args.finalize,
        )
        for t in wanted
    ]

    for cfg in configs:
        print(f"  reasoning {'on ' if cfg.enable_thinking else 'off'}  backend={cfg.backend}  "
              f"mechanistic={cfg.mechanistic}  concurrency={cfg.max_concurrency}  "
              f"run_name={cfg.run_name}")

    if args.finalize:
        print("\nFinalize only (no model load).")
        results = run_sweep(configs)
        for label, path in results.items():
            print(f"{label}: {path}")
        return

    if args.preflight or not args.skip_preflight:
        print("\nPreflight:")
        preflight(configs[-1])
        if args.preflight:
            return

    print()
    results = run_sweep(configs)
    for label, path in results.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
