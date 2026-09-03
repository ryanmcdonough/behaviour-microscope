# Working in this repository

## The one rule that matters

**Neuronpedia's `interp-engine` is the interpretability runtime. Do not build a second one.**

No PyTorch hooks, no activation capture infrastructure, no architecture mappings, no
activation-writing mechanism, no steering infrastructure, no model-specific interpretability
code. If you need one of those, the engine already has it — read
[`docs/SUPPORTED_POINTS.md`](https://github.com/decoderesearch/interp-engine/blob/main/docs/SUPPORTED_POINTS.md)
and [`docs/AGENT_INTEGRATION.md`](https://github.com/decoderesearch/interp-engine/blob/main/docs/AGENT_INTEGRATION.md).

`tests/test_interp_discipline.py` enforces this. It fails if any file registers a torch hook, if
`interp_engine` is imported outside `src/microscope/interp.py`, or if that adapter imports an
engine submodule (submodules are explicitly not API).

Custom instrumentation requires a documented reason in RESEARCH.md explaining what the engine
could not do. So far there are none: the engine has served every operation this experiment needs.

## Engine rules that bite

From the engine's own `AGENT_INTEGRATION.md`, the ones this project depends on:

- Every model *method* is async. The sync free functions (`run_with_cache`, `generate_stream`,
  `steer`) take either backend; `sync_model(model)` mirrors the methods.
- `steer()`'s `position_mask` names positions to **exclude**. Patching one position means masking
  every other one.
- `AddSpec` applies `coeff * vector`. That is why an exact activation patch is expressible as
  adding the difference between two activations — see RESEARCH.md §3.2.
- Use `sync.decode_residuals` (the method), not the free function: the method applies the model's
  `final_logit_softcapping`, which Gemma-2 has and Gemma-2 results are wrong without.
- `GenStep.logits` is eager-only. Anything meant to run on vLLM must not depend on it.
- Never guess a point name by analogy. `mlp_out` and `mlp_out_post` are different tensors on
  post-norm architectures.

## Scientific discipline

Keep three things separate, in code and in prose:

| | |
| --- | --- |
| **behavioural** | the cue changed the output |
| **representational** | the activations differ |
| **causal** | intervening on those activations changed the behaviour |

Do not write "the authority neuron", "the model believes", or "the model defers to the partner
because". Divergence is not a mechanism. An intervention that changes an output is not a located
representation until the zero and random-direction controls are clean. A null result is a result
and should be reported as one — do not reach for a mechanistic story for an effect that was not
demonstrated.

## Legal ground truth

Never generate it. Every scenario cites a primary source (`legislation.gov.uk`, the CPR, an
official regulator) with a retrieval date. If a statutory change with a known commencement date
will make an answer wrong, set `superseded_from` on the scenario; `stale_scenarios()` and the
runner surface it.

## Scope

v0.1 stays small. No database, no web app, no Streamlit, no agent framework, no generic
interpretability abstraction. Implementation lives in `src/`; the notebook stays thin.
