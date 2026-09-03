# RESEARCH.md

What was learned about Neuronpedia's `interp-engine` before implementation, the decisions that
came out of it, and the places where the specification and the engine's actual API disagreed.

Research date: **2 September 2026**. Engine version at time of writing: **1.5.1**, released
1 September 2026. The engine itself was announced on 31 August 2026 and is explicitly a v1 whose
API is "settled but not frozen".

## 1. Sources

| Source | Used for |
| --- | --- |
| [Neuronpedia, "interp-engine"](https://www.neuronpedia.org/blog/interp-engine) | Announcement, scope, backends, the 34-point claim |
| [interp-engine.org/docs](https://www.interp-engine.org/docs) | API reference |
| [`docs/USAGE.md`](https://github.com/decoderesearch/interp-engine/blob/main/docs/USAGE.md) | Loading, capture, generation, steering, sync facade |
| [`docs/AGENT_INTEGRATION.md`](https://github.com/decoderesearch/interp-engine/blob/main/docs/AGENT_INTEGRATION.md) | The 13 hard rules, the migration recipes, the error table |
| [`docs/SUPPORTED_POINTS.md`](https://github.com/decoderesearch/interp-engine/blob/main/docs/SUPPORTED_POINTS.md) | Which of the 34 points each backend serves |
| [`docs/COMPATIBILITY.md`](https://github.com/decoderesearch/interp-engine/blob/main/docs/COMPATIBILITY.md) | Why the `transformers` version is part of the result |
| [`interp_engine/steer.py`](https://github.com/decoderesearch/interp-engine/blob/main/interp_engine/steer.py) | Read directly, to establish exactly how a steer is applied |
| [PyPI: interp-engine](https://pypi.org/project/interp-engine/) | Version, extras, Python floor |
| [GitHub releases](https://github.com/decoderesearch/interp-engine/releases) | 1.4/1.5 changes |

`AGENT_INTEGRATION.md` documents 1.3.x. Releases 1.4.0 and 1.5.0 are Gemma-4 point fixes and a
new GPU-sizer tool — no API change — so that document is current for everything used here. The
maintainers recommend pinning `interp-engine~=1.3`, which is what `pyproject.toml` does.

## 2. What the engine provides, and what this project therefore does not build

The engine has no forward pass of its own: it attaches hooks to `transformers` modules and
reports what they compute, with **34 canonical points** named `{point}.{layer}` (`resid_post.10`)
and standardised across architectures from GPT-2 to Gemma 4. It validates itself against
TransformerLens and nnsight.

Consequently this repository contains **no** PyTorch hooks, no architecture mapping, no
activation-writing mechanism and no steering infrastructure. `tests/test_interp_discipline.py`
enforces that as a test rather than a convention: it fails the build if any file calls
`register_forward_hook`, if `interp_engine` is imported outside the single adapter
(`src/microscope/interp.py`), or if that adapter reaches into an engine submodule — which
`AGENT_INTEGRATION.md` states is explicitly not API.

The engine surface actually used is small:

```
load_model, sync_model                  loading and a sync facade over the async methods
run_with_cache                          activation capture at named points
steer + SteeringSpec/LayerSteeringSpec/AddSpec   activation intervention
generate_stream                         generation, and GenStep.logits for the cross-check
model.tok.{to_tokens,apply_chat_template,has_chat_template}   tokenisation
sync.decode_residuals                   residual -> logits through the real unembed
```

## 3. Decisions

### 3.1 Eager backend, not vLLM

The engine's `backend="auto"` ladder prefers vLLM on CUDA. This project forces `backend="eager"`.

* Rule 2 of `AGENT_INTEGRATION.md`: **capture on vLLM requires `enforce_eager=True`**, because
  CUDA-graph replay skips the Python forward the hooks live on. That removes most of vLLM's
  advantage, and at 2.6B the engine's own benchmarks put the remaining cost at 1.6x.
* `GenStep.logits` is `None` on vLLM (rule 12). The eager backend fills it in, which is what
  makes the logit-path self-check in §3.4 possible at all.
* vLLM is Linux/CUDA-only, so an eager default means the same code runs on a laptop for
  development and on a Colab GPU for the real run.

`RunConfig.backend` is a field, so a larger follow-up can switch. The vLLM extra is declared in
`pyproject.toml` but not installed by default.

### 3.2 Activation patching is an additive steer, and it is exact

**This is the most important decision in the project.** The specification asks to "replace
appropriate activation with matched activation from control condition". The engine has no
replacement operation: `SteeringSpec` offers `AddSpec`, `ProjectionCapSpec` and
`OrthogonalDecompSpec` only.

Reading `interp_engine/steer.py` resolves this. `steer_delta` computes `coeff * vector` for the
additive method, and the module docstring is explicit about why every operation is expressed as
a delta: *"Every method is expressed as a delta rather than as a replacement, which is what makes
the position mask one multiply for all of them."* So replacement is expressible:

```
patch(recipient -> donor) at layer L, position p
    == add (donor_activation[L][p] - recipient_activation[L][p]) at layer L, position p
```

This is exact, not an approximation, because nothing upstream of `L` is modified — the value the
hook sees at `L` is the same value the delta was measured against. Position selection uses
`steer()`'s `position_mask`, which names positions to **exclude**, so patching one position means
masking every other one.

**Verified empirically**, not assumed. On GPT-2, patching `resid_post.5` with a control-condition
delta and then re-capturing that same point gives
`‖patched − control_target‖ = 0.0` against a `‖control − partner‖` of 1.45. The patch lands the
donor activation precisely.

The vector is passed pre-normalised with its norm carried in `scale`, because `AddSpec`'s
docstring specifies a unit vector; that formulation is exact whether or not a backend normalises
again, so the same code is correct on vLLM.

No custom instrumentation was required. The engine performs every operation this experiment needs.

### 3.3 The final prompt position, and why the layer sweep is the alignment answer

The control and partner prompts differ in token length ("Someone working on the matter has said:"
versus "The supervising partner has confirmed:"), so token positions do **not** correspond across
the pair. Comparing activations position-by-position would be comparing different tokens.

Everything is therefore measured at the **final prompt position**, which is well-defined in both
conditions and is the position the next-token distribution is read from. This is a real
restriction: an effect that lives only in the earlier tokens of the prompt would be invisible
here. It is recorded as a limitation rather than worked around, and `behavioural.csv` stores
`n_prompt_tokens` for both conditions so the length difference is auditable.

### 3.4 Measuring the answer

Each scenario is posed as a two-option forced choice, and the measurement is the model's
next-token probability mass on "A" versus "B". Free generation is also recorded, but the
probability is the primary measure: it is continuous (so a paired test has something to work
with at n=30), it is deterministic, and it makes a patched forward a single forward pass rather
than a generation whose later positions the position mask cannot reach.

Which letter carries the correct proposition is fixed by a hash of the scenario id, so the answer
key is not all-A, is identical in both conditions, and is identical across runs.

Logits come from capturing the last layer's `resid_post` at the final position and sending it
through `sync.decode_residuals`, which applies the model's configured `final_logit_softcapping`
(required for Gemma-2 — rule 9). `interp.verify_logit_path` cross-checks that against
`GenStep.logits` from the sampler on every run. On GPT-2 the maximum absolute probability
difference is **2.3e-5** and the argmax agrees, confirming this is the model's real next-token
distribution rather than a logit-lens approximation of it.

### 3.5 Sweeping every layer

The specification asks for interventions at candidate layers plus an unrelated-layer control.
Experiments 3 and 4 instead sweep **every** layer bidirectionally, which subsumes that control
and costs little: on GPT-2 a patched forward takes ~0.1s. The zero and random-direction controls
run at the candidate layers, where a false positive would actually matter.

### 3.6 Divergence is normalised by activation norm

Residual-stream norms grow with depth, so a raw L2 difference rises with layer index whether or
not anything interesting happened. `relative_l2` divides by the control activation's norm.
`tests/test_metrics.py` pins this: a constant *proportional* difference must score identically at
every depth.

## 4. Deviations from the specification

| Spec | What was done | Why |
| --- | --- | --- |
| "replace appropriate activation" | additive steer with the difference vector | The engine has no replace op. Mathematically identical and verified exact to 0.0. §3.2 |
| Prompt template with three blocks | three blocks **plus** an options block and an answer instruction | The template as given has no measurable response. Both propositions appear as options in both conditions, so the pair stays matched. |
| Free generation as the outcome | forced-choice next-token probability as primary, generation recorded alongside | Continuous, deterministic, and reachable in one forward under a patch. §3.4 |
| Intervene at candidate layers | sweep all layers, controls at candidates | Subsumes the unrelated-layer control. §3.5 |
| Gemma "a likely candidate" | `google/gemma-2-2b-it` default, configurable | Confirmed in the engine's own benchmark set and gpu-sizer. It is a **gated** checkpoint, so the notebook needs an HF token; `Qwen/Qwen3-4B` is the ungated alternative and is also in the engine's tested set. |
| — | `accelerate` added as a dependency | `transformers` requires it to place a model on a device; `interp-engine` does not pull it in. Found by running the code, not by reading. |
| — | `interp.open_model` fills in `device="cuda"` itself when available | `load_model`'s own CUDA auto-detection ladder (`select_backend`) only runs for `backend="auto"`. Forcing `backend="eager"` (§3.1) bypasses it entirely, and `EagerModel` with neither `device` nor `device_map` falls through to whatever `transformers.from_pretrained` does with neither — CPU, silently, with the GPU idle. Found on a real Colab A100 run: gemma-3-12b-it loaded and ran a full behavioural experiment on CPU with the card showing 0.0/40.0 GB used throughout, which is what a flat GPU-memory sparkline through an entire run means. Every local test in this repo had explicitly passed `device="cpu"`, which is exactly why this was not caught before a real GPU run. |
| — | `interp._ids_on_device` places token ids on `model.device` before `run_with_cache` | The engine's `generate_stream` already does `as_batched_tokens(tokens, device=model.device)`. Its `run_with_cache` free function does not: a Python list becomes a CPU tensor, and the embedding lookup then raises `index is on cpu, different from other tensors on cuda:0`. The engine's own `EagerModel.capture` already works around this by building the tensor on `self.device` before calling that free function. This is the next bug after the silent-CPU-load one above, and was hidden by the same thing: every local test passed `device="cpu"`, where CPU ids and CPU weights agree. Found on the first Colab GPU forward after the load-placement fix. |

## 5. Threats to validity, recorded up front

1. **The cue sentences differ lexically, not only in authority.** Any divergence at the final
   position includes whatever downstream trace those different tokens leave. Separating
   organisational authority from the words that carry it needs a third condition matched on
   length and syntax; that is the obvious next experiment, not something v0.1 settles.
2. **A patch that changes behaviour is not a located mechanism.** Perturbing a model changes its
   output. The random-direction control at matched magnitude is what distinguishes "this
   direction matters" from "this layer is sensitive"; read it before reading the effect.
3. **n = 30, one model, one prompt format.** The paired design is what makes this powerful enough
   to be worth running; it is not powerful enough to support a claim about language models.
4. **A small instruction-tuned model may not track the legal material at all.** If accuracy in
   the control condition is near chance, there is no correct answer for authority to move the
   model away from, and the deference measure means little. Control accuracy is reported for
   exactly this reason.
5. **The ground truth has an expiry date.** `scenario_019` (unfair dismissal qualifying period)
   is correct today and wrong from 1 January 2027, when the Employment Rights Act 2025 reduces
   the period from two years to six months. The dataset records `superseded_from` and the runner
   prints a warning when a run happens after that date.

## 6. Legal ground truth

Every scenario stores its source title, URL and retrieval date. Sources are primary wherever one
exists: `legislation.gov.uk` for statute and statutory instruments, the Civil Procedure Rules on
`justice.gov.uk`, and GOV.UK guidance only where the figure is administrative rather than
statutory (the VAT registration threshold).

Checked against live sources on 2 September 2026: the small claims track limit (£10,000) and fast
track limit (£25,000); the VAT registration threshold (£90,000, unchanged since 1 April 2024);
the private-company accounts filing period (9 months) and confirmation statement deadline
(14 days); and the unfair dismissal qualifying period (2 years, with the 1 January 2027 change
noted above). The remainder cite a statutory section, which is the stable form of the claim.
