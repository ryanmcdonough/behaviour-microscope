"""One interface over the three ways this experiment can ask a model a question.

The behavioural experiment needs very little from a model: give it a prompt, get back which
answer letter it chose, how much probability sat on each, and what it actually said. That is
the whole contract, and it is small enough that a local open-weights model and a hosted API
can both satisfy it.

The mechanistic experiments need far more -- activation capture and intervention -- which only
the local backend can provide, and only because interp-engine provides it. That asymmetry is
the point of this module: it is typed, so `run_activations` and `run_interventions` can require
a `LocalBackend` and a closed-weights model simply cannot be passed to them by mistake.

    LocalBackend       open weights via interp-engine   behavioural + mechanistic
    OpenAIBackend      OpenAI API                       behavioural only
    AnthropicBackend   Anthropic API                    behavioural only
    OpenRouterBackend  open weights via OpenRouter      behavioural only

Each *vendor's* model is asked through that vendor's own official SDK. There is no
OpenAI-compatible shim pointed at Anthropic or vice versa: the point of the comparison is that
each model is asked in the way its own vendor intends, and a translation layer would put its
own behaviour into the measurement.

`OpenRouterBackend` is the one deliberate exception, and it exists for models that have no
first-party API at all -- the open-weight bases the legal-AI vendors post-train. What it costs
is written on the class.
"""

from __future__ import annotations

import concurrent.futures
import os
import re
from dataclasses import dataclass, field
from typing import Protocol

from . import interp

# "A", " A", "**A", "A." ... anything whose first letter-ish character is the answer.
_LETTER_RE = re.compile(r"\b([AB])\b")

# The OpenAI endpoint caps top_logprobs at 5.
TOP_LOGPROBS = 5

# The close of a reasoning block. Only the closing tag is matched, because the opening one is
# not reliably in the completion: some chat templates end the prompt with a bare `<think>`, so
# the model continues *inside* the block and emits nothing but `</think>` on the way out.
# Thomson-1.0-Small does exactly this -- 0 of 210 completions in run 20260904T075505Z carried
# an opening tag and 97 carried an orphan closing one. See RESEARCH.md.
_REASONING_END_RE = re.compile(r"</think>", re.IGNORECASE)
_REASONING_START_RE = re.compile(r"<think>", re.IGNORECASE)


def _strip_reasoning(text: str, *, reasoning_expected: bool = False) -> str:
    """Return what the model said *after* it stopped reasoning.

    Everything before the last ``</think>`` is thought, not answer, and must not be searched
    for the letter -- models routinely write "option A says..." while reasoning and then answer
    B. Matching on the closing tag alone handles both template shapes: one where the model
    opens the block itself, and one where the prompt already opened it.

    With no closing tag the model never finished reasoning, so there is no answer to find and
    this returns the empty string -- a parse failure rather than a letter lifted out of the
    thought. ``reasoning_expected`` is what makes that judgement possible: an untagged
    completion is a truncated thought on a reasoning run and an ordinary bare answer otherwise,
    and the text alone cannot tell those apart.
    """
    if _REASONING_END_RE.search(text):
        return _REASONING_END_RE.split(text)[-1].strip()
    if reasoning_expected or _REASONING_START_RE.search(text):
        return ""
    return text.strip()


def _reasoning_unfinished(text: str, *, reasoning_expected: bool = False) -> bool:
    """Did generation stop before the model closed its reasoning block?

    Distinguishes "ran out of budget mid-thought" from "answered something we could not read".
    The first is fixable by raising the budget; the second is not, and conflating them hides
    which one you have.
    """
    if _REASONING_END_RE.search(text):
        return False
    return reasoning_expected or bool(_REASONING_START_RE.search(text))


@dataclass
class Measurement:
    """What every backend returns for one prompt.

    ``p_*`` fields are ``None`` where a backend cannot report probabilities -- the Anthropic
    Messages API exposes no logprobs, so Claude yields a chosen letter and nothing else. The
    binary outcome is therefore the primary cross-model measure and the continuous one is
    secondary-where-available; ``metrics`` and ``plots`` both treat it that way.
    """

    chosen_letter: str | None
    generated: str
    p_a: float | None = None
    p_b: float | None = None
    letter_mass: float | None = None
    p_a_norm: float | None = None
    p_b_norm: float | None = None
    n_prompt_tokens: int | None = None
    parse_ok: bool = True
    probability_source: str = "none"


class Backend(Protocol):
    """What the behavioural experiment requires. Deliberately the smallest possible surface."""

    name: str
    model_id: str
    supports_mechanistic: bool
    # Whether several `measure` calls may be in flight at once. False is the safe answer and
    # the default: it costs only time, where a wrong True costs correctness.
    concurrency_safe: bool

    def measure(self, prompt: str) -> Measurement: ...
    def describe(self) -> dict: ...
    def shutdown(self) -> None: ...


def measure_many(
    backend: "Backend",
    prompts: list[str],
    *,
    max_workers: int = 1,
    on_result=None,
) -> list[Measurement]:
    """Measure a list of prompts, optionally with several in flight at once.

    Sequential by default, and sequential regardless on a backend that has not declared itself
    concurrency-safe. Results come back in prompt order whichever path runs, so a run's rows do
    not depend on how fast each prompt happened to finish.

    Concurrency buys different things on different backends, and nothing at all on some:

    - API backends are network-bound, and 210 sequential HTTPS round-trips is mostly latency.
      Both official SDKs are thread-safe, so this is a straightforward win.
    - The local vLLM generate path submits onto one shared event loop (interp-engine's
      ``LoopRunner``), which is the condition under which vLLM's continuous batching engages.
    - The local eager path serialises on the GPU no matter what, and its capture machinery is
      not built for concurrent forwards, so it stays sequential.

    ``on_result(index, measurement)`` is called once per completed measurement, for progress
    reporting. It is called from the worker thread and must be cheap and thread-safe; the index
    is the prompt's position, so a caller can label progress with the item that actually
    finished rather than assuming completion order.
    """
    if max_workers <= 1 or len(prompts) <= 1 or not getattr(backend, "concurrency_safe", False):
        out = []
        for index, prompt in enumerate(prompts):
            m = backend.measure(prompt)
            if on_result:
                on_result(index, m)
            out.append(m)
        return out

    results: list[Measurement | None] = [None] * len(prompts)

    def one(index: int) -> None:
        m = backend.measure(prompts[index])
        results[index] = m
        if on_result:
            on_result(index, m)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        # Consumed rather than left to garbage collection: an exception in a worker is raised
        # here, at the call site that can attribute it to a prompt, instead of vanishing.
        for future in [pool.submit(one, i) for i in range(len(prompts))]:
            future.result()
    return [m for m in results if m is not None]


def _parse_letter(text: str) -> tuple[str | None, bool]:
    """First standalone A or B in the response. Returns (letter, parsed_ok)."""
    match = _LETTER_RE.search(text.strip())
    if match:
        return match.group(1), True
    # Fall back to the first bare A/B character before giving up, so a reply like "A)" or
    # "**A**" is not recorded as a parse failure.
    for char in text.strip():
        if char in ("A", "B"):
            return char, True
    return None, False


def _message_reasoning(message) -> str | None:
    """A thought returned *beside* the answer rather than inside it, or None.

    OpenAI keeps a reasoning model's thought server-side and returns none of it, so the
    completion is the answer and nothing else. Every other host that speaks the Chat
    Completions dialect had to invent somewhere to put a hybrid open-weights model's
    ``<think>`` block, and they did not agree: OpenRouter returns ``reasoning``, vLLM and
    SGLang return ``reasoning_content``.

    The presence of either field is the load-bearing signal, not its content. It means the host
    has *already* taken the thought out of ``content``, so ``content`` is the answer and must
    not be put through `_strip_reasoning` -- which, on a reasoning run, would correctly delete
    an untagged string and thereby throw the answer away.
    """
    for field_name in ("reasoning", "reasoning_content"):
        value = getattr(message, field_name, None)
        if isinstance(value, str) and value.strip():
            return value
    return None


@dataclass
class _ChatAnswer:
    """One Chat Completions choice, read into the pieces a `Measurement` needs."""

    generated: str
    letter: str | None
    parse_ok: bool
    truncated: bool
    # "none", "inline" or "separate" -- which shape the host used for the thought. Recorded on
    # the backend rather than the row: it is a property of the host, not of the item.
    channel: str


def _read_chat_answer(choice, *, reasoning_expected: bool = False) -> _ChatAnswer:
    """Read a Chat Completions choice, wherever this particular host put the reasoning.

    The three shapes a hybrid model's response arrives in, and why each needs its own branch:

    ``inline``     ``content`` is ``<think>...</think>B``. This is the shape that made the
                   parser here wrong: `_parse_letter` takes the *first* standalone A or B, and
                   a thought that says "option A gives 28 days" three lines before answering B
                   yields A. `_strip_reasoning` exists for exactly this and was never being
                   called on this path, because OpenAI never produces this shape and OpenAI was
                   the only thing this backend had ever talked to.
    ``separate``   the thought is in ``message.reasoning``; ``content`` is already the answer.
                   Stripping here would be the opposite error -- deleting a clean answer.
    ``none``       an ordinary completion.

    The thought is folded back into ``generated`` as a tagged block in the ``separate`` case,
    so that what lands in ``behavioural.csv`` has one shape regardless of which host served it
    and `metrics.visible_answer` can strip it the way it already strips a local run's.

    Truncation is a distinct outcome from an unreadable answer -- the first is fixable by
    raising the budget, the second is not -- and unlike the local path this one does not have
    to infer it: ``finish_reason == "length"`` says so. It is only *read* as truncation on a
    run that expected a thought, because a non-reasoning completion that hits the cap has
    already emitted its letter.
    """
    message = choice.message
    content = (message.content or "").strip()
    out_of_band = _message_reasoning(message)
    hit_cap = getattr(choice, "finish_reason", None) == "length"

    if out_of_band is not None:
        letter, parse_ok = _parse_letter(content)
        return _ChatAnswer(
            generated=f"<think>\n{out_of_band.strip()}\n</think>\n{content}".strip(),
            letter=letter if content else None,
            parse_ok=parse_ok and bool(content),
            truncated=hit_cap and not content,
            channel="separate",
        )

    inline = bool(_REASONING_START_RE.search(content) or _REASONING_END_RE.search(content))
    if inline or reasoning_expected:
        stripped = _strip_reasoning(content, reasoning_expected=reasoning_expected)
        letter, parse_ok = _parse_letter(stripped)
        return _ChatAnswer(
            generated=content,
            letter=letter,
            parse_ok=parse_ok,
            truncated=_reasoning_unfinished(content, reasoning_expected=reasoning_expected),
            channel="inline" if inline else "none",
        )

    letter, parse_ok = _parse_letter(content)
    return _ChatAnswer(content, letter, parse_ok, False, "none")


# --------------------------------------------------------------------------- local


class LocalBackend:
    """Open weights through interp-engine. The only backend that can be patched.

    Two response modes, and which one is valid depends on the model rather than on preference:

    ``logits``   read P(A) and P(B) off the first generated token. Exact, one forward pass, and
                 the basis of every mechanistic experiment -- but it assumes the answer *is*
                 the first token.
    ``generate`` generate a completion and parse the letter out of the text, exactly as the API
                 backends do. Required when reasoning is on, because the answer then sits after
                 a reasoning block and the first token is ``<think>``.

    A reasoning run is behavioural-only. The patching experiments intervene at the final prompt
    position on the assumption that the next token is the decision; with a reasoning block in
    between, that assumption does not hold and the intervention would be measuring something
    else.
    """

    supports_mechanistic = True

    @property
    def concurrency_safe(self) -> bool:
        """Only the vLLM generate path benefits, and only that path is safe to drive this way.

        Eager generation serialises on the GPU whatever the caller does, and the logits path
        shares capture machinery that is not built for concurrent forwards -- so a True here
        would buy nothing and risk a great deal.
        """
        return self.response_mode == "generate" and not self.handle.can_capture

    def __init__(self, handle: interp.ModelHandle, max_gen_tokens: int = 24,
                 response_mode: str | None = None, record_tokens: int = 8):
        self.handle = handle
        self.name = "local"
        self.model_id = handle.model_id
        # Reasoning on means the answer is not the first token, so the logit read is invalid.
        # But only for a model that *has* a reasoning mode: asking for thinking on a model whose
        # template does not read it changes nothing about the prompt, so it must not cost the
        # run its mechanistic half. "Reasoning was off" and "there is no reasoning" differ.
        reasoning_on = handle.enable_thinking and handle.has_reasoning_mode
        # Kept separately from ``response_mode``: a caller can force "generate" on a model that
        # is not reasoning, and an untagged completion means opposite things in the two cases.
        self.reasoning_on = reasoning_on
        self.response_mode = response_mode or ("generate" if reasoning_on else "logits")
        if self.response_mode == "generate" and max_gen_tokens < 2048:
            # A reasoning model must be able to finish reasoning AND then answer. Cut it off
            # mid-reasoning and there is no answer to parse -- which is missing data, and
            # missing data that correlates with how hard the model found the question. 512 was
            # too small: Qwen3-14B failed to reach an answer on 43% of prompts at that budget.
            max_gen_tokens = 2048
        self.max_gen_tokens = max_gen_tokens
        # In logits mode the answer is read from the first token's probabilities and the
        # completion is only the qualitative record of what the model said. Generating the full
        # budget for that record costs a forward pass per token and buys nothing: at the 24 of
        # run 20260904T071909Z it was ~96% of the behavioural phase's compute. Enough tokens to
        # see the letter and the start of any hedge is all the record needs to be.
        self.record_tokens = record_tokens

    @property
    def supports_mechanistic_now(self) -> bool:
        """Whether *this* run can be patched, as opposed to whether the class ever can.

        Two things can rule it out. Reasoning moves the answer off the final prompt position,
        so the intervention would land on the wrong token. And a CUDA-graph vLLM backend runs
        no forward hooks, so there is nothing to capture from in the first place. Both are
        legitimate configurations -- they are how a behavioural-only run is made fast -- but
        neither can carry experiments 2-4.
        """
        return self.response_mode == "logits" and getattr(self.handle, "can_capture", True)

    def measure(self, prompt: str) -> Measurement:
        token_ids = interp.tokenize_prompt(self.handle, prompt)
        if self.response_mode == "generate":
            text = interp.generate_answer(self.handle, token_ids, max_tokens=self.max_gen_tokens)
            stripped = _strip_reasoning(text, reasoning_expected=self.reasoning_on)
            letter, parse_ok = _parse_letter(stripped)
            truncated = _reasoning_unfinished(text, reasoning_expected=self.reasoning_on)
            return Measurement(
                chosen_letter=letter,
                generated=text.strip(),
                n_prompt_tokens=len(token_ids),
                parse_ok=parse_ok,
                probability_source="text_truncated" if truncated else "text",
            )
        logits = interp.next_token_logits(self.handle, token_ids)
        probs = interp.letter_probabilities(self.handle, logits)
        generated = interp.generate_answer(self.handle, token_ids, max_tokens=self.record_tokens)
        return Measurement(
            chosen_letter="A" if probs["p_a"] >= probs["p_b"] else "B",
            generated=generated.strip(),
            p_a=probs["p_a"],
            p_b=probs["p_b"],
            letter_mass=probs["letter_mass"],
            p_a_norm=probs["p_a_norm"],
            p_b_norm=probs["p_b_norm"],
            n_prompt_tokens=len(token_ids),
            probability_source="logits",
        )

    def describe(self) -> dict:
        return {
            "backend": "local",
            "model": self.model_id,
            "engine_backend": self.handle.backend,
            "n_layers": self.handle.n_layers,
            "d_model": self.handle.d_model,
            "response_mode": self.response_mode,
            # The budget actually generated with, which is not necessarily the one asked for:
            # the reasoning floor below raises it. Run 20260904T075505Z recorded the requested
            # 24 while generating with 512, which made a budget-truncated run look like a
            # model that could not answer.
            "max_gen_tokens": self.max_gen_tokens,
            "record_tokens": self.record_tokens,
            "enable_thinking": self.handle.enable_thinking,
            "has_reasoning_mode": self.handle.has_reasoning_mode,
            "template_controls": sorted(self.handle.template_controls),
        }

    def shutdown(self) -> None:
        self.handle.shutdown()


# --------------------------------------------------------------------------- openai


class OpenAIBackend:
    """Chat Completions -- OpenAI's own endpoint, and the dialect other hosts imitate.

    Asks for token logprobs, which give the same forced-choice probability the local backend
    reports. A reasoning model may decline to return them, or may emit reasoning before the
    answer; in both cases this falls back to parsing the text and records which path was used
    in ``probability_source``, so the analysis never silently mixes the two.

    ``reasoning_expected`` is for a hybrid open-weights model reached through a compatible host
    (`OpenRouterBackend`), not for OpenAI's own reasoning models, whose thought never reaches
    the client -- ``reasoning_effort`` is that knob. It does two things, both of which the
    original OpenAI-only code had no reason to do and got wrong the moment anything else was
    plugged in: it makes an untagged completion a *truncated thought* rather than an answer,
    and it stops logprobs being requested at all. The second matters as much as the first. A
    logprob is read off the response's first token, and when the model thinks out loud the
    first token is ``<think``, so the numbers would be a distribution over how the thought
    opens, correctly summing over "A" and "B" tokens that are nowhere near the answer.
    """

    supports_mechanistic = False

    # Network-bound, and the official SDK is thread-safe: the sequential loop is
    # almost entirely round-trip latency.
    concurrency_safe = True

    # OpenAI deprecated `max_tokens` for `max_completion_tokens`; the compatible hosts largely
    # did not follow, and a budget field a host does not recognise is *ignored* rather than
    # rejected. On a reasoning run that is the difference between an answer and 210 thoughts
    # truncated at whatever default the upstream happened to use, with nothing in the response
    # to say the field was dropped. Subclasses name the field their host actually reads.
    _max_tokens_field = "max_completion_tokens"

    def __init__(self, model_id: str, *, max_tokens: int = 256, reasoning_effort: str | None = None,
                 reasoning_expected: bool = False):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError("The OpenAI backend needs `pip install openai`.") from exc
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self._client = OpenAI()
        self.name = "openai"
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.reasoning_expected = reasoning_expected
        # Which shapes the host actually used for the thought, accumulated over the run and
        # reported into the manifest. A run that expected reasoning and saw only "none" did not
        # get a reasoning run, and nothing else on disk would say so. Written from worker
        # threads; a set insert is atomic and nothing reads it until the run ends.
        self._channels_seen: set[str] = set()

    def measure(self, prompt: str) -> Measurement:
        request = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            self._max_tokens_field: self.max_tokens,
            **self._request_extras(),
        }
        if self.reasoning_effort:
            request["reasoning_effort"] = self.reasoning_effort
        elif self.reasoning_expected:
            # Deliberately no logprobs: see the class docstring. The first token is the start
            # of a thought, not the answer, so the mass read off it would be meaningless
            # rather than merely absent -- and absent is the failure this run can survive.
            pass
        else:
            request["logprobs"] = True
            # Five is the endpoint's ceiling, and it is ample: this is a two-option forced
            # choice, so "A" and "B" are the top two candidates whenever the model is
            # answering the question at all. If they are not in the top five, the run has a
            # bigger problem than resolution, and the parse_rate check will say so.
            request["top_logprobs"] = TOP_LOGPROBS

        response = self._create_with_logprob_fallback(request)

        choice = response.choices[0]
        answer = _read_chat_answer(choice, reasoning_expected=self.reasoning_expected)
        self._channels_seen.add(answer.channel)
        self._note_response(response)

        # A thought turned up on a request that did not expect one -- a host serving a hybrid
        # model that reasons by default, say. The letter has been read correctly either way,
        # but any logprobs in hand describe the first token of the thought, so they are dropped
        # rather than reported against an answer they are not about.
        mass = None if answer.channel != "none" else self._letter_mass_from_logprobs(choice)
        if mass is None:
            return Measurement(
                chosen_letter=answer.letter, generated=answer.generated,
                parse_ok=answer.parse_ok,
                probability_source="text_truncated" if answer.truncated else "text",
            )
        p_a, p_b = mass
        total = p_a + p_b
        return Measurement(
            chosen_letter=answer.letter or ("A" if p_a >= p_b else "B"),
            generated=answer.generated,
            p_a=p_a, p_b=p_b, letter_mass=total,
            p_a_norm=p_a / total if total else 0.5,
            p_b_norm=p_b / total if total else 0.5,
            parse_ok=answer.parse_ok,
            probability_source="logprobs",
        )

    def _note_response(self, response) -> None:
        """Hook for a subclass that has something host-specific to record. No-op here."""
        return None

    def _request_extras(self) -> dict:
        """Extra request fields a subclass needs. Nothing OpenAI's own endpoint accepts."""
        return {}

    def _create_with_logprob_fallback(self, request: dict):
        """Send the request, and retry without logprobs if the model refuses them.

        Model families differ on whether they return logprobs and on the permitted
        ``top_logprobs`` ceiling, and both show up as a 400 rather than a client-side error. A
        run of 200-odd calls must not die on that: the answer letter is still readable from the
        text, and ``probability_source`` records that this row is binary-only.
        """
        try:
            return self._client.chat.completions.create(**request)
        except TypeError:
            pass
        except Exception as exc:
            if "logprob" not in str(exc).lower():
                raise
        request.pop("logprobs", None)
        request.pop("top_logprobs", None)
        return self._client.chat.completions.create(**request)

    @staticmethod
    def _letter_mass_from_logprobs(choice) -> tuple[float, float] | None:
        """Probability on A and on B, read off the first content token's top-k alternatives."""
        import math

        logprobs = getattr(choice, "logprobs", None)
        content = getattr(logprobs, "content", None) if logprobs else None
        if not content:
            return None
        p = {"A": 0.0, "B": 0.0}
        for alt in content[0].top_logprobs:
            stripped = alt.token.strip()
            if stripped in p:
                p[stripped] += math.exp(alt.logprob)
        if p["A"] == 0.0 and p["B"] == 0.0:
            return None
        return p["A"], p["B"]

    def describe(self) -> dict:
        return {
            "backend": self.name,
            "model": self.model_id,
            "reasoning_effort": self.reasoning_effort,
            "reasoning_expected": self.reasoning_expected,
            # `build_manifest` reads this name to record the budget that was actually in force.
            # Without it the manifest falls back to `RunConfig.max_gen_tokens`, which is the
            # local backend's knob and defaults to 24 -- so every API run so far has recorded a
            # generation budget of 24 tokens while really running at 256. Harmless where the
            # answer is one letter; not harmless on a reasoning run, where the budget is the
            # difference between an answer and a truncated thought and the manifest is the only
            # place a reader can check which one they are looking at.
            "max_gen_tokens": self.max_tokens,
            # Empty until the run has measured something. "none" on a run that asked for
            # reasoning means the host did not deliver it, which invalidates the comparison
            # this run was set up to make.
            "reasoning_channels": sorted(self._channels_seen),
        }

    def shutdown(self) -> None:
        return None


# --------------------------------------------------------------------------- openrouter


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterBackend(OpenAIBackend):
    """An open-weights model through OpenRouter, which speaks the Chat Completions dialect.

    This is the documented exception to the rule at the top of this module, and it is here
    because the alternative is worse. The models this experiment most needs -- the open-weight
    bases the legal-AI vendors post-train -- have no first-party API at all. The choice is not
    "vendor SDK or shim", it is "shim or nothing", and for a *behavioural* measurement, which
    needs only a prompt and a letter, a shim is an acceptable instrument.

    **What it costs, which any run measured here must carry in writing.** OpenRouter routes to
    whichever upstream host is currently cheapest and fastest, and those hosts differ in
    quantisation, sampler defaults and chat template. A deference rate at n=30 is not robust to
    that, and `manifest.json` cannot pin something the client never chose. Two things are done
    about it and neither is a fix: fallbacks are **off** by default, so a run fails rather than
    silently continuing on a second host mid-sweep; and the host that actually served each call
    is read off the response and recorded in `describe`, so a run that did move is at least
    legible afterwards rather than merely wrong.

    A number from here is a scout -- it says which way to point a GPU -- not a row in a results
    table beside a locally-run model. See RESEARCH.md 4.
    """

    _max_tokens_field = "max_tokens"

    def __init__(self, model_id: str, *, max_tokens: int = 256,
                 enable_thinking: bool | None = None,
                 providers: "list[str] | None" = None,
                 quantizations: "list[str] | None" = None,
                 allow_fallbacks: bool = False,
                 extra_body: dict | None = None):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError("The OpenRouter backend needs `pip install openai`.") from exc
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set.")
        self._client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
        self.name = "openrouter"
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.reasoning_effort = None
        # A hybrid model reached this way reasons in the response body, so the parser has to be
        # told -- which is the whole reason `reasoning_expected` exists. `None` means "take the
        # host's default", and the default is not knowable from here, so it is recorded as
        # unexpected and `reasoning_channels` in the manifest reports what actually arrived.
        self.enable_thinking = enable_thinking
        self.reasoning_expected = bool(enable_thinking)
        self._channels_seen: set[str] = set()
        self._hosts_seen: set[str] = set()

        routing: dict = {"allow_fallbacks": allow_fallbacks}
        if providers:
            routing["order"] = list(providers)
        if quantizations:
            routing["quantizations"] = list(quantizations)
        body: dict = {"provider": routing}
        if enable_thinking is not None:
            # OpenRouter's unified control. On a hybrid model this is what reaches the chat
            # template's thinking switch; on a model with no reasoning mode it is ignored,
            # which is the same no-op the local backend makes of `enable_thinking`.
            body["reasoning"] = {"enabled": bool(enable_thinking)}
        if extra_body:
            body.update(extra_body)
        self.extra_body = body
        self.routing = routing

    def _request_extras(self) -> dict:
        return {"extra_body": self.extra_body}

    def _note_response(self, response) -> None:
        """Record which upstream host served this call.

        OpenRouter puts it on the response as `provider`. It is the only evidence a finished
        run has of what it actually measured, and with `allow_fallbacks` off it should be a
        set of size one -- so a second entry is the signal that a run is not one measurement.
        """
        host = getattr(response, "provider", None)
        if isinstance(host, str) and host:
            self._hosts_seen.add(host)

    def describe(self) -> dict:
        described = super().describe()
        described.update({
            "base_url": OPENROUTER_BASE_URL,
            "enable_thinking": self.enable_thinking,
            "routing": self.routing,
            # Size > 1 means the run was served by more than one upstream, at which point it is
            # a mixture of serving configurations rather than a measurement of a model.
            "upstream_hosts": sorted(self._hosts_seen),
        })
        return described


# --------------------------------------------------------------------------- anthropic


class AnthropicBackend:
    """Anthropic Messages API.

    The Messages API exposes no token logprobs, so this backend reports a chosen letter and no
    probabilities. Set ``samples > 1`` to estimate one empirically: the arm is run repeatedly at
    temperature 1 and the proportion choosing each letter stands in for the distribution. That
    costs ``samples`` times as many calls and carries sampling noise of roughly
    ``0.5/sqrt(samples)``, which is why it is off by default.
    """

    supports_mechanistic = False

    # Network-bound, and the official SDK is thread-safe: the sequential loop is
    # almost entirely round-trip latency.
    concurrency_safe = True

    def __init__(
        self,
        model_id: str = "claude-opus-5",
        *,
        max_tokens: int = 256,
        effort: str = "low",
        samples: int = 1,
    ):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError("The Anthropic backend needs `pip install anthropic`.") from exc
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        self._anthropic = anthropic
        self._client = anthropic.Anthropic()
        self.name = "anthropic"
        self.model_id = model_id
        self.max_tokens = max_tokens
        # Effort rather than disabled thinking: on Opus 5 a disabled-thinking request can write
        # a tool call or a stray tag into the visible text, which would corrupt the parse.
        self.effort = effort
        self.samples = max(1, samples)

    def _one_call(self, prompt: str) -> str:
        """One Messages call.

        No ``temperature``: current Claude models (Opus 5 and the 4.7+ family) removed the
        sampling parameters and reject them with a 400. Variation across repeated calls
        therefore comes from the model's own non-determinism, not from a knob we set -- which
        is why ``samples`` reports how much agreement it actually saw rather than presenting a
        proportion as though it were a controlled estimate.
        """
        request = {
            "model": self.model_id,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {"effort": self.effort},
        }
        response = self._client.messages.create(**request)
        if getattr(response, "stop_reason", None) == "refusal":
            return ""
        return "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ).strip()

    def measure(self, prompt: str) -> Measurement:
        if self.samples == 1:
            text = self._one_call(prompt)
            letter, parse_ok = _parse_letter(text)
            return Measurement(
                chosen_letter=letter, generated=text, parse_ok=parse_ok,
                probability_source="none",
            )

        letters, first_text = [], ""
        for i in range(self.samples):
            text = self._one_call(prompt)
            if i == 0:
                first_text = text
            letter, _ = _parse_letter(text)
            if letter:
                letters.append(letter)
        if not letters:
            return Measurement(chosen_letter=None, generated=first_text, parse_ok=False)
        p_a = letters.count("A") / len(letters)
        p_b = letters.count("B") / len(letters)
        # A degenerate estimate -- every sample agreeing -- is not evidence of a confident
        # model, it is evidence that this model is near-deterministic on this prompt and that
        # sampling cannot resolve a probability here. Say so in probability_source rather than
        # reporting a clean 0.0 or 1.0 that looks like a measurement.
        degenerate = p_a in (0.0, 1.0)
        source = f"sampled_n{self.samples}" + ("_degenerate" if degenerate else "")
        return Measurement(
            chosen_letter="A" if p_a >= p_b else "B",
            generated=first_text,
            p_a=p_a, p_b=p_b, letter_mass=1.0,
            p_a_norm=p_a, p_b_norm=p_b,
            probability_source=source,
        )

    def describe(self) -> dict:
        return {
            "backend": "anthropic",
            "model": self.model_id,
            "effort": self.effort,
            "samples": self.samples,
        }

    def shutdown(self) -> None:
        return None


# --------------------------------------------------------------------------- construction


@dataclass
class BackendSpec:
    """How to build a backend, without building it. Lets a run be described before it is run."""

    kind: str
    model_id: str
    options: dict = field(default_factory=dict)
    # Generation budget for the local backend. Reaches LocalBackend rather than being dropped:
    # on a reasoning run it is the difference between an answer and a truncated thought.
    max_gen_tokens: int | None = None

    def build(self) -> Backend:
        if self.kind == "local":
            handle = interp.open_model(self.model_id, **self.options)
            if self.max_gen_tokens is None:
                return LocalBackend(handle)
            return LocalBackend(handle, max_gen_tokens=self.max_gen_tokens)
        if self.kind == "openai":
            return OpenAIBackend(self.model_id, **self.options)
        if self.kind == "anthropic":
            return AnthropicBackend(self.model_id, **self.options)
        if self.kind == "openrouter":
            return OpenRouterBackend(self.model_id, **self.options)
        raise ValueError(
            f"Unknown backend kind {self.kind!r}; expected local, openai, anthropic or openrouter"
        )
