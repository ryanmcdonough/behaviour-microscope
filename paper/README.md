# `paper/` — generated artefacts

**`tables/`, `figures/` and `RESULTS.md` are generated. Do not edit them.** Hand-written drafts
live alongside them — `frontier-models.md` is one — and are never touched by the build.

```bash
python scripts/build_stack.py
```

Reads `results/`, selects the runs the paper is based on, and rewrites `tables/`, `figures/` and
`RESULTS.md`. It creates and overwrites only those; anything else in this directory is yours.

Hand-written documents: `PAPER_BRIEF.md` at the repo root (theory, argument, and the limits on
what may be claimed), `RESEARCH.md`, `HANDOVER.md`, and any draft sections here. When a draft
quotes a rate, the generated tables are the authority — re-run the build and check the draft
against it rather than the other way round.

## Dropping in a new run

Put the run directory in `results/` and re-run the command. Nothing else needs touching.

Selection is `microscope.runs.select_current`: for each configuration — model, reasoning mode,
effort — it keeps every run at the **most recent commit that configuration has**, and drops any
run graded `fail`. So a re-run at a newer commit supersedes its predecessor automatically,
replicates at the same commit are all kept so a median and range can be computed, and a
configuration nobody has re-run keeps whatever evidence exists for it. `RESULTS.md` lists what
was discovered but not selected, and why.

The two Qwen3-14B runs are currently at earlier commits than everything else. When re-runs at
the current commit land, they will supersede these without any edit here — `PAPER_BRIEF.md` §9.1
documents why the current ones stand in the meantime, and that section should be cut once they
are replaced.

## What is here

| file | contents |
| --- | --- |
| `RESULTS.md` | the digest: every headline table, formatted to read |
| `tables/fpar.csv` | acceptance by arm, with per-arm denominators and replicate ranges |
| `tables/contrasts.csv` | the seven planned contrasts per configuration, binary and continuous, Holm-corrected |
| `tables/factorial.csv` | 2×2 main effects, interaction, cell means, and which measure each used |
| `tables/sensitivity.csv` | Post-hoc answer-position strata and leave-one-legal-area-out ranges |
| `tables/mechanism.csv` | divergence, bidirectional patch, and both intervention controls |
| `tables/elaboration.csv` | format-compliance rates, or the reason the measure does not apply |
| `tables/stability.csv` | item-level disagreement across replicates |
| `tables/provenance.csv` | every selected run: commit, engine version, quality, timestamp |
| `figures/fig1_fpar_by_arm.png` | the headline result |
| `figures/fig2_tier_gradient.png` | partner against court; the diagonal is the failure |
| `figures/fig3_reasoning.png` | reasoning off against on, per model measured both ways |
| `figures/fig4_mechanism.png` | where the cue's effect becomes causal, by relative depth |
| `figures/fig5_elaboration.png` | the sub-threshold signal |

Per-run figures and `summary.json` stay in each run's own directory: those answer "is this run
any good", which is a different question from "what does the study say".
