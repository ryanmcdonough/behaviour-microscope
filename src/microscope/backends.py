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

Each provider uses its own official SDK. There is no OpenAI-compatible shim pointed at
Anthropic or vice versa: the point of the comparison is that each model is asked in the way its
own vendor intends, and a translation layer would put its own behaviour into the measurement.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Protocol

from . import interp

# "A", " A", "**A", "A." ... anything whose first letter-ish character is the answer.
_LETTER_RE = re.compile(r"\b([AB])\b")

# The OpenAI endpoint caps top_logprobs at 5.
TOP_LOGPROBS = 5


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

    def measure(self, prompt: str) -> Measurement: ...
    def describe(self) -> dict: ...
    def shutdown(self) -> None: ...


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


# --------------------------------------------------------------------------- local


class LocalBackend:
    """Open weights through interp-engine. The only backend that can be patched."""

    supports_mechanistic = True

    def __init__(self, handle: interp.ModelHandle, max_gen_tokens: int = 24):
        self.handle = handle
        self.name = "local"
        self.model_id = handle.model_id
        self.max_gen_tokens = max_gen_tokens

    def measure(self, prompt: str) -> Measurement:
        token_ids = interp.tokenize_prompt(self.handle, prompt)
        logits = interp.next_token_logits(self.handle, token_ids)
        probs = interp.letter_probabilities(self.handle, logits)
        generated = interp.generate_answer(self.handle, token_ids, max_tokens=self.max_gen_tokens)
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
        }

    def shutdown(self) -> None:
        self.handle.shutdown()


# --------------------------------------------------------------------------- openai


class OpenAIBackend:
    """OpenAI Chat Completions.

    Asks for token logprobs, which give the same forced-choice probability the local backend
    reports. A reasoning model may decline to return them, or may emit reasoning before the
    answer; in both cases this falls back to parsing the text and records which path was used
    in ``probability_source``, so the analysis never silently mixes the two.
    """

    supports_mechanistic = False

    def __init__(self, model_id: str, *, max_tokens: int = 256, reasoning_effort: str | None = None):
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

    def measure(self, prompt: str) -> Measurement:
        request = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": self.max_tokens,
        }
        if self.reasoning_effort:
            request["reasoning_effort"] = self.reasoning_effort
        else:
            request["logprobs"] = True
            # Five is the endpoint's ceiling, and it is ample: this is a two-option forced
            # choice, so "A" and "B" are the top two candidates whenever the model is
            # answering the question at all. If they are not in the top five, the run has a
            # bigger problem than resolution, and the parse_rate check will say so.
            request["top_logprobs"] = TOP_LOGPROBS

        response = self._create_with_logprob_fallback(request)

        choice = response.choices[0]
        text = (choice.message.content or "").strip()
        letter, parse_ok = _parse_letter(text)

        mass = self._letter_mass_from_logprobs(choice)
        if mass is None:
            return Measurement(
                chosen_letter=letter, generated=text, parse_ok=parse_ok,
                probability_source="text",
            )
        p_a, p_b = mass
        total = p_a + p_b
        return Measurement(
            chosen_letter=letter or ("A" if p_a >= p_b else "B"),
            generated=text,
            p_a=p_a, p_b=p_b, letter_mass=total,
            p_a_norm=p_a / total if total else 0.5,
            p_b_norm=p_b / total if total else 0.5,
            parse_ok=parse_ok,
            probability_source="logprobs",
        )

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
            "backend": "openai",
            "model": self.model_id,
            "reasoning_effort": self.reasoning_effort,
        }

    def shutdown(self) -> None:
        return None


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

    def _one_call(self, prompt: str, temperature: float | None) -> str:
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
            text = self._one_call(prompt, temperature=None)
            letter, parse_ok = _parse_letter(text)
            return Measurement(
                chosen_letter=letter, generated=text, parse_ok=parse_ok,
                probability_source="none",
            )

        letters, first_text = [], ""
        for i in range(self.samples):
            text = self._one_call(prompt, temperature=1.0)
            if i == 0:
                first_text = text
            letter, _ = _parse_letter(text)
            if letter:
                letters.append(letter)
        if not letters:
            return Measurement(chosen_letter=None, generated=first_text, parse_ok=False)
        p_a = letters.count("A") / len(letters)
        p_b = letters.count("B") / len(letters)
        return Measurement(
            chosen_letter="A" if p_a >= p_b else "B",
            generated=first_text,
            p_a=p_a, p_b=p_b, letter_mass=1.0,
            p_a_norm=p_a, p_b_norm=p_b,
            probability_source=f"sampled_n{self.samples}",
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

    def build(self) -> Backend:
        if self.kind == "local":
            handle = interp.open_model(self.model_id, **self.options)
            return LocalBackend(handle)
        if self.kind == "openai":
            return OpenAIBackend(self.model_id, **self.options)
        if self.kind == "anthropic":
            return AnthropicBackend(self.model_id, **self.options)
        raise ValueError(f"Unknown backend kind {self.kind!r}; expected local, openai or anthropic")
