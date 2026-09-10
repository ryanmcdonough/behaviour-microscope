#!/usr/bin/env python
"""Run the minimum behavioural controls needed before submitting the paper.

The controls are deliberately separate named runs so each can be resumed after a Colab timeout
and audited in its own manifest. Existing main-result directories are never overwritten.

Examples:

    python scripts/run_submission_controls.py --model google/gemma-3-12b-it
    python scripts/run_submission_controls.py --model Qwen/Qwen3-14B --backend eager
    python scripts/run_submission_controls.py --model qwen/qwen3.6-35b-a3b \
        --provider openrouter --temperature 0 --max-concurrency 6 \
        --provider-options-json '{"enable_thinking": false, "providers": ["AkashML"]}'

The five runs cost 840 behavioural forwards per model:

* neutral headings, all seven main arms (210)
* reversed A/B order, all seven main arms (210)
* explicit supervision wording, the 2x2 only (120)
* alternative legal-role wording, the 2x2 only (120)
* true-proposition versions of the six attribution cues (180)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from microscope.experiment import RunConfig, run_sweep  # noqa: E402
from microscope.scenarios import CONDITIONS, CONTROL_CONDITIONS  # noqa: E402

FACTORIAL = (
    "junior_said",
    "junior_confirmed",
    "partner_said",
    "partner_confirmed",
)


def _slug(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_id.lower()).strip("-")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="model id")
    parser.add_argument(
        "--provider", choices=("local", "openai", "anthropic", "openrouter"), default="local"
    )
    parser.add_argument("--backend", default="eager", help="local interp-engine backend")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument(
        "--provider-options-json",
        default="{}",
        help="additional backend constructor options as JSON",
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=("neutral-headings", "answer-swap", "hierarchy-explicit", "legal-roles", "true-cues"),
        help="run only the named control; repeat to select several",
    )
    args = parser.parse_args()

    provider_options = json.loads(args.provider_options_json)
    if args.temperature is not None:
        if args.provider == "anthropic":
            parser.error("Anthropic does not accept a temperature control")
        if args.provider == "local":
            if args.temperature != 0:
                parser.error("The local behavioural logits path is greedy; use temperature 0")
        else:
            provider_options["temperature"] = args.temperature

    # Include the provider so a local run and an API run of the same model cannot resume into
    # one another's result directory.
    prefix = f"{args.provider}-{_slug(args.model)}"
    common = dict(
        model_id=args.model,
        provider=args.provider,
        backend=args.backend,
        dtype=args.dtype,
        results_root=args.results_root,
        provider_options=provider_options,
        mechanistic=False,
        max_concurrency=args.max_concurrency,
    )
    specs = {
        "neutral-headings": RunConfig(
            **common,
            arms=CONDITIONS,
            prompt_style="neutral",
            run_name=f"{prefix}-control-neutral-headings",
        ),
        "answer-swap": RunConfig(
            **common,
            arms=CONDITIONS,
            swap_options=True,
            run_name=f"{prefix}-control-answer-swap",
        ),
        "hierarchy-explicit": RunConfig(
            **common,
            arms=FACTORIAL,
            cue_variant="hierarchy_explicit",
            run_name=f"{prefix}-control-hierarchy-explicit",
        ),
        "legal-roles": RunConfig(
            **common,
            arms=FACTORIAL,
            cue_variant="legal_roles",
            run_name=f"{prefix}-control-legal-roles",
        ),
        "true-cues": RunConfig(
            **common,
            arms=CONTROL_CONDITIONS,
            run_name=f"{prefix}-control-true-cues",
        ),
    }
    selected = args.only or list(specs)
    run_sweep([specs[name] for name in selected])


if __name__ == "__main__":
    main()
