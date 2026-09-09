# Recorded runs

This directory contains the 17 runs used by `RESULTS.md`. Do not pool repeated API runs: each
repeat measures the same 30 scenarios. `scripts/build_stack.py` groups runs by model configuration
and reports the median arm rate with its observed range.

Every complete run contains:

- `behavioural.csv`: one row per scenario and experimental arm;
- `manifest.json`: model, revision, environment, configuration, source commit, and timings;
- `summary.json`: within-run statistics;
- `quality_report.json`: validity checks and overall pass/warn/fail verdict;
- `plots/`: figures for that run;
- for mechanistic runs, `activation_analysis.csv`, `activation_per_scenario.csv`, and
  `interventions.csv`.

The included configurations and directory mapping are listed in section 6 of `../RESULTS.md`.
