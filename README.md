# Behaviour Microscope v0.1

Does an organisational authority cue change how a language model handles conflicting UK legal
information — and if the model's behaviour changes, can that change be located and causally
tested inside the model?

Interpretability is [Neuronpedia's `interp-engine`](https://www.neuronpedia.org/blog/interp-engine).
This repository supplies the legal scenarios, the orchestration, the statistics and the plots.
It deliberately contains no interpretability infrastructure of its own — see
[RESEARCH.md](RESEARCH.md).

```
UK legal scenarios -> prompt pairs -> orchestration -> interp-engine -> open model
                                                                            |
                          plots <- statistics <- measurements <-------------+
```

> **Picking this up cold?** Start with [HANDOVER.md](HANDOVER.md) — the research question, the
> results so far, what has already gone wrong and why, and what is outstanding.

## The experiment

Thirty matched UK legal scenarios (England and Wales). Each pairs a piece of authoritative
material with a false proposition that contradicts it, and generates two prompts that are
identical except for who is credited with the false proposition:

| arm | cue | isolates |
| --- | --- | --- |
| `floor` | *(no assertion at all)* | unpressured accuracy — the anchor |
| `junior_said` | "A junior colleague has said:" | the factorial's reference cell |
| `junior_confirmed` | "A junior colleague has confirmed:" | the epistemic verb |
| `partner_said` | "The supervising partner has said:" | **seniority** — the actual hypothesis |
| `partner_confirmed` | "The supervising partner has confirmed:" | the original two-arm condition |
| `court` | "The court has held:" | legal, not organisational, authority |
| `adverse` | "The opposing party asserts:" | whether an adverse source is discounted |

The four middle arms form a 2×2. This matters: the original design compared "partner **confirmed**"
against "someone **said**", varying source *and* epistemic verb at once, so an effect measured
across that pair could not be attributed to seniority. Holding the verb constant down each column
isolates source; holding the source constant along each row isolates verb. The mechanistic
experiments patch between `junior_said` and `partner_said` by default, so the mechanism answers
the source question rather than the verb question.

Four experiments run over that pair:

1. **Behaviour** — does the cue change the answer? *False Proposition Acceptance Rate* in each
   condition, and the *Authority Deference Delta* between them, tested with an exact paired
   (McNemar) test on the binary outcome and a paired bootstrap on the probability.
2. **Representation** — where do the two conditions' residual streams diverge, layer by layer?
3. **Causal, forward** — patch the control activation into the partner condition. Does deference
   fall?
4. **Causal, reverse** — patch the partner activation into the control condition. Does deference
   rise?

Plus three controls: a zero-magnitude patch through the same machinery, a magnitude-matched
random direction, and the full layer sweep itself as the unrelated-layer control.

**The point is the method, not a result.** Divergence between two conditions is not a mechanism,
and an intervention that changes an output is not a located representation until the controls say
it is. If there is no behavioural effect, the run reports that.

## Running it

### Colab

Open [`notebooks/authority_v1_colab.ipynb`](notebooks/authority_v1_colab.ipynb), select a GPU
runtime, Run All. The notebook clones this repo, installs dependencies, verifies the GPU and the
engine, and calls `run_all()`. Everything else lives in `src/`.

The default model `google/gemma-3-12b-it` is a gated checkpoint: accept the licence on Hugging Face
and set an `HF_TOKEN` Colab secret first. A Colab **A100** (40GB) is enough; a T4 is not.
`Qwen/Qwen3-4B` is the ungated alternative.

### Cross-model replication

Closed-weights models are **behavioural only** — they cannot be captured or patched, which is
enforced by the type system rather than by convention (`run_activations` and `run_interventions`
require a `LocalBackend`).

```python
run_all(RunConfig(model_id="gpt-5.1", provider="openai"))
run_all(RunConfig(model_id="claude-opus-5", provider="anthropic",
                  provider_options={"effort": "low"}))
```

The Anthropic Messages API exposes no token logprobs, so Claude yields a chosen letter and no
probability. That is why the **binary** acceptance rate is the primary cross-model measure — it
needs only the letter — and the continuous probability is secondary-where-available. Pass
`samples=k` to estimate a proportion empirically instead.

### Locally

```bash
pip install -e '.[dev]'          # add '.[apis]' for the OpenAI and Anthropic backends
pytest                           # 54 tests, no GPU, model, or API key needed
```

```python
from microscope.experiment import RunConfig, run_all

run_all(RunConfig(model_id="google/gemma-3-12b-it"))
```

A CPU smoke run that exercises every code path in a couple of minutes:

```python
run_all(RunConfig(model_id="openai-community/gpt2", limit=5,
                  extra_load_kwargs={"device": "cpu"}))
```

## Results

Each run writes a timestamped directory under `results/`:

```
manifest.json                 model, revision, engine and transformers versions, git commit,
                              GPU, seed, generation config, phase timings, logit-path check
behavioural.csv               one row per scenario per condition
activation_analysis.csv       per-layer divergence, averaged with a bootstrap CI
activation_per_scenario.csv   the same, unaggregated
interventions.csv             every patch and control, one row each
summary.json                  the headline statistics
quality_report.json           pass/warn/fail on whether *this run's* numbers are trustworthy
plots/                        the four figures
```

## Is this run any good?

Every run grades its own output. `quality_report.json` (`src/microscope/quality.py`) checks
things a plot cannot show on its own: whether the model actually engaged with the forced-choice
format, whether control-condition accuracy is above chance (if it isn't, there is no correct
answer for authority to move the model away from), whether the zero-magnitude patch control is a
true no-op, and whether the random-direction control is smaller than the real intervention effect
it is there to be compared against. The notebook prints this straight after the behavioural
summary, before the plots. A `fail` here is itself a finding — usually that the chosen model or
prompt format is wrong for the task — and should stop you from reading the plots as results.

To review a run without re-running the model — a run downloaded from Colab, or handed to someone
else for a second opinion:

```python
from microscope import quality
quality.review("results/20260903T120000Z")   # or wherever it was unzipped to
```

`manifest.json` is what makes a run reproducible: it pins the checkpoint revision, the
`interp-engine` version and the `transformers` version, which the engine's own
[COMPATIBILITY.md](https://github.com/decoderesearch/interp-engine/blob/main/docs/COMPATIBILITY.md)
is emphatic is part of the numerical result rather than a footnote.

## Legal ground truth

Every scenario carries its source and URL, primary wherever one exists — `legislation.gov.uk`,
the Civil Procedure Rules, GOV.UK guidance for administrative figures. Ground truth was checked
against those sources on 2 September 2026, not generated.

Where a statutory change with a known commencement date will make a stored answer wrong, the
scenario records `superseded_from` and the runner warns on any run after that date. One scenario
currently carries such a date: the unfair dismissal qualifying period changes on 1 January 2027.

## Layout

```
data/authority_v1.jsonl         the 30 scenarios
src/microscope/scenarios.py     dataset, prompt construction
src/microscope/interp.py        the only file that talks to interp-engine
src/microscope/experiment.py    orchestration and the result directory
src/microscope/metrics.py       paired statistics
src/microscope/plots.py         the four figures
tests/                          dataset, statistics, plots, and the no-second-engine rule
```

## Next

The obvious follow-up, once there is a behavioural effect to explain, is to separate
*organisational* authority from *legal* authority: supervising partner versus court order versus
statute versus regulator, matched for sentence length and syntax. v0.1 cannot distinguish
deference to a person from the lexical trace of a different sentence.
