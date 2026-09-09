#!/usr/bin/env python
"""Cross-run analysis over every result directory on disk.

One run grades itself; this compares them. Runs are grouped by *configuration* -- model,
reasoning mode, effort -- because a repeated configuration is repeated measurement of the same
30 items, not more items. Where a configuration ran more than once the median arm rate is
reported with its observed range, per PAPER_BRIEF.md; rates are never pooled into a single
denominator.

    python scripts/analyse_runs.py                    # results/
    python scripts/analyse_runs.py --root ../results_archive_superseded_api
    python scripts/analyse_runs.py --incomplete        # include partial runs too
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

ARMS = ["floor", "junior_said", "junior_confirmed", "partner_said",
        "partner_confirmed", "court", "adverse"]


def discover(roots: list[Path]) -> list[dict]:
    """Every directory holding a behavioural.csv, with whatever metadata it carries."""
    found = []
    for root in roots:
        for csv_path in sorted(root.rglob("behavioural.csv")):
            d = csv_path.parent
            manifest = d / "manifest.json"
            m = json.loads(manifest.read_text()) if manifest.exists() else {}
            quality = d / "quality_report.json"
            q = json.loads(quality.read_text())["overall"] if quality.exists() else "?"
            rows = list(csv.DictReader(csv_path.open()))
            backend = m.get("backend", {})
            effort = backend.get("effort") or backend.get("reasoning_effort")
            found.append({
                "dir": d,
                "label": d.name,
                "model": m.get("model", "(unknown)"),
                "thinking": bool(m.get("enable_thinking")),
                "effort": effort,
                "quality": q,
                "commit": (m.get("git_commit") or "")[:7],
                "rows": rows,
                "complete": len(rows) >= 210,
            })
    return found


def config_key(run: dict) -> str:
    """Configuration identity. Two runs share a key iff they are the same measurement.

    The commit is part of that identity. Grouping across code versions silently averaged a
    quality-`fail` Thomson run at `c1417c9` with the good one at `bb18eda` and reported the
    mean as if it were a replicate median. Two runs at different commits may well be the same
    measurement -- HANDOVER 3 establishes that for `f8e14eb` and `0915afb` -- but that is a
    conclusion recorded in PAPER_BRIEF.md, not something to assume in a grouping key.
    """
    bits = [run["model"].split("/")[-1]]
    if run["thinking"]:
        bits.append("think")
    if run["effort"]:
        bits.append(f"effort={run['effort']}")
    bits.append(f"@{run['commit'] or '???????'}")
    return " ".join(bits)


def arm_rate(rows, arm, column="accepted_false_proposition"):
    """Rate over the scenarios this arm actually scored. Missing data stays missing."""
    vals = [r[column] for r in rows if r["condition"] == arm and r[column] not in ("", "None")]
    if not vals:
        return None, 0
    hits = sum(1 for v in vals if v == "True")
    return hits / len(vals), len(vals)


def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """Exact upper bound. A rate of 0/30 is not evidence of 0 -- it bounds the rate at ~9.5%."""
    if n == 0:
        return float("nan")
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        tail = sum(math.comb(n, i) * mid**i * (1 - mid) ** (n - i) for i in range(k + 1))
        lo, hi = (mid, hi) if tail > alpha else (lo, mid)
    return (lo + hi) / 2


def median_range(values: list[float]) -> str:
    if not values:
        return "  --  "
    med = statistics.median(values) * 100
    if len(values) == 1:
        return f"{med:5.1f} "
    return f"{med:5.1f}*" if min(values) != max(values) else f"{med:5.1f} "


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", default=None, help="result root (repeatable)")
    ap.add_argument("--incomplete", action="store_true", help="include partial runs")
    args = ap.parse_args()

    roots = [Path(r) for r in (args.root or [REPO_ROOT / "results"])]
    runs = discover(roots)
    if not args.incomplete:
        partial = [r for r in runs if not r["complete"]]
        runs = [r for r in runs if r["complete"]]
        for r in partial:
            print(f"skipping partial: {r['label']}  {r['model']}  "
                  f"{len(r['rows'])}/210 rows  (--incomplete to include)")
        if partial:
            print()

    groups: dict[str, list[dict]] = {}
    for r in runs:
        groups.setdefault(config_key(r), []).append(r)

    # ---------------------------------------------------------------- provenance
    print("=" * 100)
    print("PROVENANCE")
    print("=" * 100)
    print(f"{'configuration':44}{'runs':>5}  {'quality':16}{'commits':18}rows")
    for key, rs in sorted(groups.items()):
        qual = ",".join(sorted({r["quality"] for r in rs}))
        commits = ",".join(sorted({r["commit"] for r in rs}))
        rows = ",".join(str(len(r["rows"])) for r in rs)
        print(f"{key:44}{len(rs):>5}  {qual:16}{commits:18}{rows}")

    # ---------------------------------------------------------------- FPAR
    print()
    print("=" * 100)
    print("FALSE PROPOSITION ACCEPTANCE RATE (%)   * = varied across replicates")
    print("=" * 100)
    header = f"{'configuration':44}" + "".join(f"{a[:9]:>10}" for a in ARMS)
    print(header)
    for key, rs in sorted(groups.items()):
        cells = []
        for arm in ARMS:
            vals = [v for v, _ in (arm_rate(r["rows"], arm) for r in rs) if v is not None]
            cells.append(median_range(vals))
        print(f"{key:44}" + "".join(f"{c:>10}" for c in cells))

    # ---------------------------------------------------------------- floor + precision
    print()
    print("=" * 100)
    print("FLOOR-ARM ACCURACY, AND WHAT EACH CONFIGURATION'S ZEROS LICENCE")
    print("=" * 100)
    print(f"{'configuration':44}{'floor acc':>11}{'max arm':>10}{'95% upper bound on a 0/n arm':>32}")
    for key, rs in sorted(groups.items()):
        accs, tops, ns = [], [], []
        for r in rs:
            vals = [v for v in (r_["correct"] for r_ in r["rows"]
                                if r_["condition"] == "floor") if v not in ("", "None")]
            if vals:
                accs.append(sum(1 for v in vals if v == "True") / len(vals))
            rates = [(arm_rate(r["rows"], a)) for a in ARMS]
            tops.append(max((v for v, _ in rates if v is not None), default=0.0))
            ns.append(max((n for _, n in rates), default=0))
        n = max(ns) if ns else 0
        bound = clopper_pearson_upper(0, n) * 100 if n else float("nan")
        acc = statistics.median(accs) * 100 if accs else float("nan")
        print(f"{key:44}{acc:10.0f}%{max(tops)*100:9.1f}%{bound:31.1f}%")

    # ---------------------------------------------------------------- stability
    print()
    print("=" * 100)
    print("REPLICATE STABILITY (configurations run more than once)")
    print("=" * 100)
    for key, rs in sorted(groups.items()):
        if len(rs) < 2:
            continue
        maps = [{(x["scenario_id"], x["condition"]): x["chosen_letter"] for x in r["rows"]}
                for r in rs]
        keys = set(maps[0])
        unstable = sorted(k for k in keys if len({m.get(k) for m in maps}) > 1)
        where = {}
        for _, arm in unstable:
            where[arm] = where.get(arm, 0) + 1
        detail = ", ".join(f"{a}:{c}" for a, c in sorted(where.items())) or "none"
        print(f"{key:44}{len(unstable):>3}/{len(keys)} items differ across "
              f"{len(rs)} runs   [{detail}]")

    # ---------------------------------------------------------------- elaboration
    print()
    print("=" * 100)
    print("ELABORATION: responses exceeding the single letter asked for (%)  [exploratory]")
    print("=" * 100)
    import pandas as pd
    from microscope import metrics
    from microscope.scenarios import PLANNED_CONTRASTS

    print(f"{'configuration':44}" + "".join(f"{a[:9]:>10}" for a in ARMS))
    for key, rs in sorted(groups.items()):
        per_arm_vals: dict[str, list[float]] = {a: [] for a in ARMS}
        available, reasons = False, set()
        for r in rs:
            result = metrics.elaboration_by_arm(
                pd.read_csv(r["dir"] / "behavioural.csv"), PLANNED_CONTRASTS,
                # Without this an untagged truncated thought scores as a huge elaboration.
                reasoning_expected=r["thinking"],
            )
            if not result.get("available"):
                reasons.add(result.get("reason", "unavailable"))
                continue
            available = True
            for arm, rate in result["rate_by_arm"].items():
                if arm in per_arm_vals:
                    per_arm_vals[arm].append(rate)
        if not available:
            # Say which of the two it is: "not applicable to this scoring path" and "applicable
            # and the model complied every time" are different facts about the run.
            print(f"{key:44}n/a -- {'; '.join(sorted(reasons))[:100]}")
            continue
        print(f"{key:44}" + "".join(f"{median_range(per_arm_vals[a]):>10}" for a in ARMS))


if __name__ == "__main__":
    main()
