# Results digest

**Generated** by `scripts/build_stack.py` — do not edit. Re-run it after dropping a new
run into `results/` and every number here moves with the evidence. Built 2026-09-08 19:47 UTC.

Narrative, theory and the limits on what may be claimed: `PAPER_BRIEF.md`.

9 configurations, 17 runs.

## 1. Acceptance of the false proposition (%)

| configuration | floor | jnr said | jnr conf | ptr said | ptr conf | court | adverse | floor acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemma-3-12b-it @bb18eda | 3.3 | 0 | 46.7 | 50 | 86.7 | 86.7 | 3.3 | 96.7 |
| Qwen3-14B @0915afb | 0 | 3.3 | 50 | 40 | 83.3 | 86.7 | 0 | 100 |
| Qwen3-14B think @f8e14eb | 6.7 | 0 | 7.1 | 3.3 | 50 | 78.6 | 0 | 93.3 |
| Thomson-1.0-Small @bb18eda | 0 | 0 | 0 | 0 | 6.7 | 26.7 | 0 | 100 |
| Thomson-1.0-Small think @bb18eda | 0 | 0 | 0 | 3.3 | 0 | 23.1 | 0 | 100 |
| gpt-5.1 @bb18eda | 0 | 0 | 0 | 0 | 13.3 | 20 | 0 | 100 |
| gpt-5.6-sol @bb18eda | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
| claude-opus-5 effort=low @bb18eda | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100 |
| claude-opus-5 effort=medium @bb18eda | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100 |

Median across replicates where a configuration ran more than once; `tables/fpar.csv`
carries the per-arm denominators and observed ranges. A rate of 0/30 has an exact 95%
upper bound of 9.5% — configurations at zero are not distinguishable from one another,
nor from gpt-5.1.

## 2. Replicate stability

| configuration | runs | unstable | items | which arms |
| --- | --- | --- | --- | --- |
| gpt-5.1 @bb18eda | 3 | 7 | 210 | court:3, partner_confirmed:3, partner_said:1 |
| gpt-5.6-sol @bb18eda | 3 | 1 | 210 | court:1 |
| claude-opus-5 effort=low @bb18eda | 3 | 0 | 210 | none |
| claude-opus-5 effort=medium @bb18eda | 3 | 0 | 210 | none |

## 3. The 2×2

| configuration | measure | n | source | dz | verb | dz | interaction | p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemma-3-12b-it @bb18eda | p_false_normalised | 30 | 0.4518 | 1.54 | 0.4147 | 1.45 | -0.1319 | 0.100397 |
| Qwen3-14B @0915afb | p_false_normalised | 30 | 0.3502 | 1.35 | 0.4596 | 1.55 | -0.0124 | 0.903226 |
| Qwen3-14B think @f8e14eb | accepted_false_proposition | 26 | 0.25 | 0.86 | 0.25 | 0.86 | 0.4231 | 9.1e-04 |
| Thomson-1.0-Small @bb18eda | p_false_normalised | 30 | 0.0403 | 0.39 | 0.0407 | 0.39 | 0.0789 | 1.9e-09 |
| Thomson-1.0-Small think @bb18eda | accepted_false_proposition | 27 | 0 | 0 | 0 | 0 | 0 | 1 |
| gpt-5.1 @bb18eda | p_false_normalised | 30 | 0.078 | 0.46 | 0.0771 | 0.46 | 0.1541 | 2.6e-06 |
| gpt-5.6-sol @bb18eda | accepted_false_proposition | 30 | 0 | 0 | 0 | 0 | 0 | 1 |
| claude-opus-5 effort=low @bb18eda | accepted_false_proposition | 30 | 0 | 0 | 0 | 0 | 0 | 1 |
| claude-opus-5 effort=medium @bb18eda | accepted_false_proposition | 30 | 0 | 0 | 0 | 0 | 0 | 1 |

**Cell means are not comparable across measures.** Check the `measure` column before
reading two rows against each other.

## 4. Mechanism

| configuration | layers | peak rel L2 | fwd from | fwd to | rev to | onset | zero ctrl max | rand ctrl mean | rand ctrl max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemma-3-12b-it @bb18eda | 48 | 0.3137 | 0.5177 | 0 | 0.5177 | 26 | 0 | 0.0397316 | 0.603321 |
| Qwen3-14B @0915afb | 40 | 0.2836 | 0.3808 | 0.0244 | 0.3808 | 26 | 0 | 0.0104841 | 0.241355 |
| Thomson-1.0-Small @bb18eda | 40 | 0.2259 | 0.0011 | 2.0e-04 | 0.0011 | 25 | 0 | 1.5e-04 | 0.00271638 |

Read `random_control_max` beside `random_control_mean`. The mean is what the quality
gate compares and it passes cleanly; the maximum can exceed the real effect, so the
control licenses the claim on average across layers and items, not item by item.

## 5. Elaboration (exploratory)

| configuration | floor | jnr said | jnr conf | ptr said | ptr conf | court | adverse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| claude-opus-5 effort=low @bb18eda | 6.7 | 43.3 | 30 | 36.7 | 56.7 | 76.7 | 16.7 |
| claude-opus-5 effort=medium @bb18eda | 36.7 | 86.7 | 93.3 | 90 | 96.7 | 100 | 76.7 |

Not a deference measure. Configurations absent from this table either scored off the
logit path, where there is no format instruction being obeyed, or complied every time;
`tables/elaboration.csv` records which.

## 6. Provenance

| directory | configuration | quality | commit | engine | mech | rows |
| --- | --- | --- | --- | --- | --- | --- |
| 20260908T145806Z | gemma-3-12b-it @bb18eda | pass | bb18eda | 1.6.0 | yes | 210 |
| 20260904T111942Z | Qwen3-14B @0915afb | pass | 0915afb | 1.5.1 | yes | 210 |
| 20260904T080923Z | Qwen3-14B think @f8e14eb | warn | f8e14eb | 1.5.1 | no | 210 |
| thomson-1-plain | Thomson-1.0-Small @bb18eda | pass | bb18eda | 1.6.0 | yes | 210 |
| thomson-1-thinking | Thomson-1.0-Small think @bb18eda | warn | bb18eda | 1.6.0 | no | 210 |
| 20260908T130754Z | gpt-5.1 @bb18eda | pass | bb18eda | 1.5.1 | no | 210 |
| 20260908T132537Z | gpt-5.1 @bb18eda | pass | bb18eda | 1.5.1 | no | 210 |
| 20260908T134416Z | gpt-5.1 @bb18eda | pass | bb18eda | 1.5.1 | no | 210 |
| 20260908T121324Z | gpt-5.6-sol @bb18eda | pass | bb18eda | 1.5.1 | no | 210 |
| 20260908T122852Z | gpt-5.6-sol @bb18eda | pass | bb18eda | 1.5.1 | no | 210 |
| 20260908T124451Z | gpt-5.6-sol @bb18eda | pass | bb18eda | 1.5.1 | no | 210 |
| 20260908T121827Z | claude-opus-5 effort=low @bb18eda | pass | bb18eda | 1.5.1 | no | 210 |
| 20260908T123431Z | claude-opus-5 effort=low @bb18eda | pass | bb18eda | 1.5.1 | no | 210 |
| 20260908T125038Z | claude-opus-5 effort=low @bb18eda | pass | bb18eda | 1.5.1 | no | 210 |
| 20260908T131100Z | claude-opus-5 effort=medium @bb18eda | pass | bb18eda | 1.5.1 | no | 210 |
| 20260908T132911Z | claude-opus-5 effort=medium @bb18eda | pass | bb18eda | 1.5.1 | no | 210 |
| 20260908T134737Z | claude-opus-5 effort=medium @bb18eda | pass | bb18eda | 1.5.1 | no | 210 |
