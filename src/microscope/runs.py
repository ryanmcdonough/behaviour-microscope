"""Discovering, identifying and selecting run directories across a whole results tree.

One run grades itself; the paper compares them. That comparison needs a rule for *which* runs
are in it, and the rule has to survive a re-run landing halfway through a week's work: the
answer must change when new evidence arrives, without anything being edited by hand.

The rule is `select_current`: for each configuration, keep every run at the most recent commit
that configuration has, and drop runs that graded `fail`. A re-run at a newer commit therefore
supersedes its predecessors automatically, replicates at the same commit are all kept so a
median and a range can be computed, and a configuration nobody has re-run keeps the evidence
that exists for it.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ARMS = [
    "floor", "junior_said", "junior_confirmed", "partner_said",
    "partner_confirmed", "court", "adverse",
]

# The one probability_source that is not the model's answer to the prompt's format instruction.
# Mirrors metrics._NOT_TEXT_SCORED; see that module for why.
MECHANISTIC_FILES = ("activation_analysis.csv", "interventions.csv")


class Run:
    """One result directory, with its metadata and behavioural rows loaded."""

    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.label = self.dir.name
        manifest = self.dir / "manifest.json"
        self.manifest = json.loads(manifest.read_text()) if manifest.exists() else {}
        quality = self.dir / "quality_report.json"
        self.quality = json.loads(quality.read_text())["overall"] if quality.exists() else "?"
        with (self.dir / "behavioural.csv").open() as handle:
            self.rows = list(csv.DictReader(handle))
        backend = self.manifest.get("backend", {})
        self.model = self.manifest.get("model", "(unknown)")
        self.thinking = bool(self.manifest.get("enable_thinking"))
        self.effort = backend.get("effort") or backend.get("reasoning_effort")
        self.commit = (self.manifest.get("git_commit") or "")[:7]
        self.timestamp = self.manifest.get("timestamp_utc", "")
        self.engine = (self.manifest.get("versions") or {}).get("interp_engine")
        self.provider = self.manifest.get("provider", "?")

        recorded = self.manifest.get("config") or {}
        self.prompt_style = self.manifest.get(
            "prompt_style", recorded.get("prompt_style", "original")
        )
        self.swap_options = bool(self.manifest.get(
            "swap_options", recorded.get("swap_options", False)
        ))
        self.cue_variant = self.manifest.get(
            "cue_variant", recorded.get("cue_variant", "original")
        )
        self.arms = tuple(self.manifest.get("arms") or recorded.get("arms") or ARMS)

    @property
    def complete(self) -> bool:
        summary_path = self.dir / "summary.json"
        if summary_path.exists():
            behavioural = json.loads(summary_path.read_text()).get("behavioural", {})
            if behavioural.get("n_planned") is not None:
                return len(self.rows) >= int(behavioural["n_planned"])
        # All current designs use the same 30 checked scenarios. Falling back to the recorded
        # arm list lets 120- and 180-row control runs be complete without relaxing old runs.
        return len(self.rows) >= 30 * len(self.arms)

    @property
    def is_main_design(self) -> bool:
        """Whether this run belongs in the seven-arm headline comparison."""
        return (
            self.prompt_style == "original"
            and not self.swap_options
            and self.cue_variant == "original"
            and self.arms == tuple(ARMS)
        )

    @property
    def mechanistic(self) -> bool:
        """Whether the causal experiments actually produced output.

        Read from the files rather than the manifest's `mechanistic` flag: a run finalised in a
        second session records `mechanistic: false` because *that session* loaded no model, even
        though the earlier session wrote a full sweep.
        """
        return all((self.dir / name).exists() for name in MECHANISTIC_FILES)

    @property
    def config(self) -> str:
        """Configuration identity: what makes two runs the same measurement.

        The commit is part of it. Grouping across code versions once averaged a quality-`fail`
        Thomson run with a good one and reported the mean as a replicate median.
        """
        bits = [self.model.split("/")[-1]]
        if self.thinking:
            bits.append("think")
        if self.effort:
            bits.append(f"effort={self.effort}")
        if self.prompt_style != "original":
            bits.append(f"prompt={self.prompt_style}")
        if self.swap_options:
            bits.append("answers=swapped")
        if self.cue_variant != "original":
            bits.append(f"cue={self.cue_variant}")
        if self.arms != tuple(ARMS):
            bits.append("arms=" + "+".join(self.arms))
        bits.append(f"@{self.commit or '???????'}")
        return " ".join(bits)

    @property
    def family(self) -> str:
        """Configuration without the commit: what a re-run supersedes."""
        return self.config.rsplit(" @", 1)[0]

    def arm(self, arm: str, column: str = "accepted_false_proposition"):
        """Values for one arm, unanswered rows dropped. Missing data stays missing."""
        return [
            r[column] for r in self.rows
            if r["condition"] == arm and r[column] not in ("", "None")
        ]

    def rate(self, arm: str) -> tuple[float | None, int]:
        vals = self.arm(arm)
        if not vals:
            return None, 0
        return sum(1 for v in vals if v == "True") / len(vals), len(vals)

    def floor_accuracy(self) -> float | None:
        vals = self.arm("floor", "correct")
        return sum(1 for v in vals if v == "True") / len(vals) if vals else None

    def __repr__(self) -> str:
        return f"<Run {self.label} {self.config} {self.quality}>"


def discover(roots, *, complete_only: bool = True) -> list[Run]:
    """Every directory under `roots` holding a behavioural.csv."""
    found = []
    for root in roots:
        for csv_path in sorted(Path(root).rglob("behavioural.csv")):
            run = Run(csv_path.parent)
            if complete_only and not run.complete:
                continue
            found.append(run)
    return found


def select_current(runs: list[Run], *, drop_failed: bool = True) -> dict[str, list[Run]]:
    """The runs the paper is based on, keyed by configuration.

    For each family (model + reasoning mode + effort), keeps every run at the newest commit that
    family has. Drop a re-run into `results/` and this returns it instead of its predecessor,
    with no edit anywhere else.

    Newest is decided by manifest timestamp rather than by git order, because a results tree
    does not carry the commit graph -- a run cannot be compared to one whose commit is not an
    ancestor. In practice runs arrive in commit order, and where they do not the timestamp is
    the honest tiebreak.
    """
    by_family: dict[str, list[Run]] = {}
    for run in runs:
        if drop_failed and run.quality == "fail":
            continue
        by_family.setdefault(run.family, []).append(run)

    selected: dict[str, list[Run]] = {}
    for family, family_runs in by_family.items():
        newest = max(family_runs, key=lambda r: r.timestamp).commit
        keep = [r for r in family_runs if r.commit == newest]
        selected[keep[0].config] = sorted(keep, key=lambda r: r.timestamp)
    return selected


def superseded(runs: list[Run], selected: dict[str, list[Run]]) -> list[Run]:
    """Runs discovered but not selected, so a report can say what it left out and why."""
    kept = {r.dir for group in selected.values() for r in group}
    return [r for r in runs if r.dir not in kept]


def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """Exact upper bound on a rate. 0/30 does not mean 0; it bounds the rate at 9.5%."""
    if n == 0:
        return float("nan")
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        tail = sum(math.comb(n, i) * mid**i * (1 - mid) ** (n - i) for i in range(k + 1))
        lo, hi = (mid, hi) if tail > alpha else (lo, mid)
    return (lo + hi) / 2
