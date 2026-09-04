# Handover — Authority Deference in Legal Language Models

Status note for a session picking this up cold. What the research is, what has been run, what
was learned the hard way, and what is outstanding.

**Last updated:** 4 September 2026
**Repo:** https://github.com/ryanmcdonough/behaviour-microscope

---

## 1. The research question

When a false proposition contradicts authoritative UK legal material placed directly in the
prompt, does *who is credited with the false proposition* change whether the model accepts it?

Each of 30 England-and-Wales scenarios gives the model the governing rule, a question, and two
answer options — one supported by the rule, one contradicting it. Seven **arms** vary only the
sentence attributing the false proposition. Everything else is byte-identical.

| arm | cue | isolates |
| --- | --- | --- |
| `floor` | *(no assertion at all)* | unpressured accuracy — the anchor |
| `junior_said` | "A junior colleague has said:" | factorial reference cell |
| `junior_confirmed` | "A junior colleague has confirmed:" | the epistemic verb |
| `partner_said` | "The supervising partner has said:" | **seniority** — the hypothesis |
| `partner_confirmed` | "The supervising partner has confirmed:" | original two-arm condition |
| `court` | "The court has held:" | legal, not organisational, authority |
| `adverse` | "The opposing party asserts:" | whether an adverse source is discounted |

The middle four form a 2×2. **This matters.** The original design compared "partner **confirmed**"
against "someone **said**", varying source *and* epistemic verb at once, so the effect could not
be attributed to seniority. The factorial separates them; both turned out real and roughly equal.

Three experiments beyond behaviour, on open weights only (via Neuronpedia's `interp-engine`):
capture residuals per layer, patch them bidirectionally between arms, and run zero / random /
unrelated-layer controls.

---

## 2. What the paper argues

**Not sycophancy — a miscalibrated credibility hierarchy.** That models agree with confident
users is known. The sharper claim is that models carry an *ordered* model of source credibility,
apply it consistently, and that the ordering is wrong for law in a specific way.

Two findings make it more than a sycophancy replication:

- **They are not credulous.** Told the *opposing party* asserts the false proposition, both open
  models discount it completely — scoring identically to no assertion at all. They discriminate
  by source. The problem is where they place people.
- **A supervising partner is weighted like a court.** On both open models `partner_confirmed` and
  `court` land within a few points of each other. In law those are wildly different tiers of
  authority. gpt-5.1 keeps them apart and far lower.

**Why legal tech specifically.** The two largest legal AI vendors both announced proprietary
models built on open weights in the same week of August 2026:

- **Harvey Tenet** — their blog states it verbatim: *"Harvey Tenet is a Kimi K3 base that we
  post-trained."* Stated agenda: *"building frontier legal intelligence using open-weight models."*
- **Thomson Reuters "Thomson"** — press release says only *"starts from a strong open-source
  foundation"*; the model card for `thomsonreuters/Thomson-1.0-Small` gives base
  `tri-fair-lab/Snowdon1.1-Small` and architecture `Qwen3_5MoeForConditionalGeneration`, i.e. a
  Qwen3.5 MoE derivative. Their CTO says the starting point has changed *"close to half a dozen
  times"* — the base rotates, the strategy does not.

**The benchmark-validity argument.** Both vendors claim frontier parity, on *capability*
benchmarks (LegalBench, contract understanding, task completion, retrieval). None of those tells
the model who wants the answer to be true. Our own data makes the point: the open models are
97–100% accurate on the neutral question and 13–17% once a senior attribution is attached.
Capability was never the failing quantity.

**What the paper must NOT claim.** That Tenet or Thomson exhibit this — we measured
instruction-tuned bases, not their post-trained variants, and post-training could move deference
either way. Nor anything about CoCounsel or Harvey as *systems*: a shipped product wraps the model
in retrieval, prompting and guardrails this measurement never touches. The defensible claim is
narrower: *the class of base these products are built from carries a measurable authority-deference
failure, the benchmarks they are marketed on cannot detect it, and nobody has published a check.*

Working title: **The Partner Said So**. Venues: arXiv (cs.CL) as the anchor, JURIX or ICAIL as the
primary peer-reviewed venue, FAccT if the framing leans sociotechnical, plus a practitioner
writeup and the dataset release.

---

## 3. Results so far

All figures are False Proposition Acceptance Rate (FPAR) — the fraction of scenarios where the
model chose the false option. FPAR is the **primary cross-model measure** because it needs only
the answer letter, which every backend can report (the Anthropic API exposes no logprobs, and a
reasoning run is parsed from text).

| model | floor | junior said | junior conf. | partner said | partner conf. | court | adverse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gemma-3-12b-it | 3% | 0% | 47% | 50% | **87%** | **87%** | 3% |
| Qwen3-14B (no thinking) | 0% | 3% | 50% | 40% | **83%** | **87%** | 0% |
| gpt-5.1 | 0% | 0% | 0% | 0% | **17%** | **27%** | 0% |

**Factorial (both open models):** source and verb are each large, significant and roughly equal,
and they add. gemma: source +0.45, verb +0.42, interaction n.s. Qwen: source +0.35, verb +0.46,
interaction −0.01 (p = 0.90).

**Verb-matched headline** (`junior_said` → `partner_said`, verb held constant): gemma 0% → 50%
(McNemar p = 6.1e-5, 15 flipped / 0 reverse); Qwen 3% → 40% (p = 9.8e-4, 11 flipped / 0 reverse).

**Mechanism** (open models, verb-matched contrast): divergence flat through the early layers, sharp
onset around two-thirds depth (gemma L25 of 48, Qwen L24 of 40), saturating shortly after.
Bidirectional patching moves behaviour in opposite directions at the same layers. Controls clean:
zero-magnitude patch is an exact no-op (0.0 across 120 trials), random direction moves the output
1/12th (gemma) to 1/34th (Qwen) as far as the real patch.

**gpt-5.1 is resistant, not immune.** Binary FPAR says 0% on the verb-matched contrast, but 29 of
30 scenarios shift toward the false proposition on the continuous measure (+0.075, CI
[0.018, 0.145]). The pressure registers; it rarely crosses the decision boundary. Report both.
Its p-values sit at the test's resolution floor — quote effect sizes and CIs, not p.

### Runs that are NOT usable

- **Qwen3-14B with reasoning ON** (`20260904T055415Z`) — **failed the quality gate**: 91 of 210
  responses had no parseable answer (43%). It reported `partner_confirmed` at 17% against 83% with
  reasoning off, but see §5.3 — up to 43 points of that is artefact. **The "reasoning fixes
  deference" reading is not established.** Needs re-running at the raised token budget.

### In flight / outstanding

| run | status |
| --- | --- |
| Qwen3-14B reasoning ON, 2048-token budget | to re-run — `notebooks/run_qwen_thinking.ipynb` |
| Thomson-1.0-Small, reasoning off | running — the reasoning-off half is unaffected by §5.3 |
| Thomson-1.0-Small, reasoning on | affected by §5.3 if started on old code |
| claude-opus-5 | never ran — dropped by the key-ordering bug (§5.7) |
| Qwen3.5 / Kimi K3 | the actual vendor bases; closes the last inferential step |

---

## 4. How runs are done

**One notebook per configuration, run in parallel Colab sessions.** The combined sweep
(`authority_v1_colab.ipynb`) still works but serialises everything.

| notebook | model | GPU | time |
| --- | --- | --- | --- |
| `run_qwen_plain.ipynb` | Qwen3-14B, reasoning off | 40GB | ~4 min |
| `run_gemma_plain.ipynb` | gemma-3-12b-it (gated — needs `HF_TOKEN`) | 40GB | ~15 min |
| `run_qwen_thinking.ipynb` | Qwen3-14B, reasoning on | 40GB | 3–6 h |
| `thomson1_colab.ipynb` | Thomson-1.0-Small (70.2 GB) | **80GB** | many hours |

Each clones and `git reset --hard origin/main`, so a run is pinned to whatever was on main when it
started. Each writes a timestamped folder to `results/` and zips it for download. Every run
records its own provenance in `manifest.json` (model revision, engine and transformers versions,
git commit, GPU, seed, timings, cue strings, template controls) and grades itself in
`quality_report.json`.

**Read the quality gate before any number.** A `FAIL` means the numbers are not usable.

### Thomson-1 licence position

PolyForm Strict 1.0.0. Permits *"research, experiment, and testing for the benefit of public
knowledge"* and use by *"educational institution[s], public research organization[s]"*; explicitly
preserves fair use; **contains no restriction on benchmarking or publishing results**. Forbids
distributing or modifying the model — we do neither. The permission attaches to the **purpose**
being noncommercial; an open paper qualifies, work in service of a commercial offering is a
different question.

If a result is unfavourable, give Thomson Reuters notice and right of reply before publishing.
Norm for evaluation research naming a commercial product, and it makes the paper stronger.

---

## 5. Learnings — read this before changing anything

Every item below cost real time or produced a wrong number. They are not style notes.

### 5.1 interp-engine's sync facade refuses to run inside a running event loop
Colab's kernel keeps one running for **every** cell, so `run_all(cfg)` raises `NestedEventLoop`.
Hand it to a plain worker thread, which has no loop of its own:
```python
with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
    run_dir = pool.submit(run_all, cfg).result()
```

### 5.2 A reasoning model's answer is not the first token
Qwen3 and Thomson-1 leave the assistant turn open by default, so the first generated token is
`<think>`. The forced-choice measurement reads the first token, so the run collapses to near-zero
letter mass. `tokenize_prompt` sets `enable_thinking=False` where the template reads it — asked
via `tok.accepted_template_kwargs`, never assumed. **Gemma 3 has no reasoning mode at all**, which
is itself a finding: the reasoning mitigation, if it holds, is unavailable to it.

### 5.3 An unanswered question is missing data, not a refusal — *the worst bug so far*
`_measurement_row` scored `chosen_letter is None` as `accepted_false_proposition = False`, i.e.
counted a model that **never answered** as one that **correctly rejected** the false proposition.
Every rate was biased downward by however often parsing failed, and biased hardest on exactly the
runs where parsing is hard. This invalidated the Qwen reasoning run and nearly produced a
headline finding that was an artefact. Unanswered rows are now `None`, excluded from every rate,
denominators reported per arm, paired tests intersected on scenarios both arms answered.

Root cause of the failures: 512 generation tokens was not enough for the model to finish reasoning
*and then* answer. Raised to 2048; a run ending mid-reasoning is now recorded as `text_truncated`
so "ran out of budget" is distinguishable from "answered unreadably".

### 5.4 Forcing `backend="eager"` bypasses the CUDA-detection ladder
`load_model`'s device auto-detection only runs for `backend="auto"`. With `"eager"` forced and no
`device`, `EagerModel` falls through to whatever `from_pretrained` does with neither — **CPU,
silently, with the GPU idle**. Found on a real A100 run: gemma-3-12b ran an entire behavioural
experiment on CPU with the card at 0.0/40.0 GB throughout. `interp.open_model` now fills it in.

### 5.5 Token ids must be on the model's device
`generate_stream` places them; the `run_with_cache` free function does not. A Python list becomes
a CPU tensor and the embedding lookup raises. `interp._ids_on_device` handles it.

### 5.6 `RunConfig` fields must actually reach the backend
`max_gen_tokens` was defined on `RunConfig` and never passed to `LocalBackend` — a silent no-op on
precisely the run where the budget decides whether there is an answer at all. Check the plumbing
when adding a config field.

### 5.7 Notebook cells that depend on state a later cell creates
Three separate bugs of this shape, all invisible until a clean Run All:
- the sweep gated on API keys loaded nine cells later → **gpt-5.1 and claude silently skipped**
- the sweep referenced `OPENAI_MODEL`, defined three cells later → `NameError`
- section 7 ran the single model and 7b re-ran it in the sweep → duplicated work
Fixed by a single `MODE` switch and by moving definitions above first use. **When adding a cell,
check every name it reads is defined earlier.**

### 5.8 `pip install -e .` does not make the package importable mid-session
Editable installs register via a `.pth` file that Python's `site` machinery reads only at
interpreter startup. The notebook puts `src/` on `sys.path` directly at clone time instead.

### 5.9 OpenAI caps `top_logprobs` at 5
Not 20. A logprob-related 400 now falls back to text parsing and records
`probability_source="text"` rather than killing a 210-call run. Anthropic exposes **no** logprobs
at all and rejects sampling parameters (`temperature` etc.) on current models — hence FPAR as the
cross-model measure.

### 5.10 Eager MoE is dominated by routing overhead, not size
Thomson-1 (35B MoE) runs ~3× slower per token than Qwen3-14B (14B dense) despite having *fewer
active parameters*. HuggingFace's non-fused MoE loops in Python over hit experts
(`for expert_idx in expert_hit:` with an `index_add_` each) at every layer, and single-token
decoding gives nothing to amortise it over. **A reasoning run is behavioural-only, so it needs no
hooks** — `backend="vllm-generate"` (CUDA graphs, fused kernels, no taps) is the obvious
optimisation and is untested here.

### 5.11 There is an A-position bias, and it is controlled but should be reported
All three models accept the false proposition more readily when it is option **A** (gemma 53% vs
36%; qwen 53% vs 32%; gpt-5.1 9% vs 5%). `correct_letter` is hash-fixed per scenario and held
constant across arms, so the bias is an identical constant in every arm and the **paired design
differences it out** — every between-arm claim is unaffected. It does inflate *absolute* rates
(the dataset is 17 false=A against 13 false=B). Counterbalance in the dataset rebuild; report as a
limitation meanwhile.

### 5.12 The measurement is deterministic
Two Qwen3-14B runs at the same config, on different GPUs (40GB and 80GB) at different commits,
reproduced **arm for arm, exactly**. Worth a line in the paper's reproducibility section.

### 5.13 Things verified that turned out fine
- **No length confound** within arms (88% vs 86% short/long prompts on gemma).
- **No repetition confound.** The floor arm states the false proposition once, assertion arms
  twice; both sit at 0–3%. Restating it does nothing — the attribution does.
- **Ground truth holds.** One floor failure in 90 trials (gemma, `scenario_009`, Limitation Act
  s.8). Both other models get it right and the statute is unambiguous: a model error, not a data
  error.
- **The 2×2 arms are token-identical on gemma in all 30 scenarios**, so position-aligned
  full-prompt patching is available without a dataset rebuild. On Qwen it is a constant 1-token
  offset (partner arms 170 vs junior 169), which is a known fixed shift rather than ragged.

---

## 6. Outstanding work, in priority order

1. **Re-run Qwen reasoning-on at 2048 tokens.** The reasoning question is the biggest open
   variable and the current answer is an artefact. Watch `parse_rate` before reading anything.
2. **Finish the model set** — claude-opus-5 (never ran), and Thomson-1 both ways.
3. **Measure the actual vendor bases** — Qwen3.5 (9B fits 40GB, 27B needs 80GB) and Kimi K3. This
   closes the last inferential step between the finding and the shipped products.
4. **Counterbalance the answer key** — run each scenario in both letter orders and average.
   Doubles the run, removes the A-bias from absolute rates entirely.
5. **Open-generation arm.** Does the false proposition survive into a drafted memo unqualified?
   Practitioners will object that forced choice is not the workflow.
6. **Dataset rebuild to ~100 items** with an interpretive subset (currently almost every scenario
   is a bright-line numeric rule — the easiest ground truth) and a held-out split.
7. **Mitigation study.** A verification instruction; reordering so the authority follows the
   material; an activation steer against the identified direction. The measurement that matters is
   not whether deference drops but **what else breaks** — a model that stops weighting sources
   would also stop following the client's instructions.

---

## 7. Discipline

From `CLAUDE.md`, and it is the reason the results are worth anything:

| | |
| --- | --- |
| **behavioural** | the cue changed the output |
| **representational** | the activations differ |
| **causal** | intervening on those activations changed the behaviour |

Keep them separate in code and in prose. Divergence is not a mechanism. An intervention that
changes an output is not a located representation until the zero and random-direction controls are
clean. A null result is a result. Do not write "the authority neuron" or "the model believes".

`src/microscope/quality.py` makes this machine-checked — extend it when a new failure mode turns
up, never bypass it. And **interp-engine is the interpretability runtime**: no bespoke hooks,
capture, or steering. `tests/test_interp_discipline.py` enforces that.
