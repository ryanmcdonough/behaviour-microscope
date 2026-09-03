"""The backend abstraction, including both API providers, without spending any API budget.

The provider clients are mocked. What is being tested is our side of the contract: that a
response shape becomes the right Measurement, that the letter parser handles what models
actually emit, and that a backend without logprobs degrades to a binary outcome rather than
inventing probabilities.
"""

import sys
import types
from unittest.mock import MagicMock

import pytest

from microscope import backends
from microscope.backends import AnthropicBackend, Measurement, OpenAIBackend, _parse_letter


# --------------------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    "text,expected",
    [
        ("A", "A"),
        ("B", "B"),
        (" A ", "A"),
        ("A.", "A"),
        ("**B**", "B"),
        ("A)", "A"),
        ("The answer is B", "B"),
        ("Answer: A\nBecause the CPR says 14 days.", "A"),
    ],
)
def test_letter_parser_handles_what_models_actually_emit(text, expected):
    letter, ok = _parse_letter(text)
    assert (letter, ok) == (expected, True)


def test_letter_parser_reports_failure_rather_than_guessing():
    letter, ok = _parse_letter("I cannot advise on this matter.")
    assert letter is None
    assert ok is False


def test_a_hedged_refusal_is_a_parse_failure_not_a_silent_default():
    """A refusal must not be scored as an answer -- that would be fabricated data."""
    letter, ok = _parse_letter("")
    assert letter is None and ok is False


# --------------------------------------------------------------------------- openai


def _openai_response(text, top_logprobs=None):
    choice = MagicMock()
    choice.message.content = text
    if top_logprobs is None:
        choice.logprobs = None
    else:
        alts = []
        for token, logprob in top_logprobs:
            alt = MagicMock()
            alt.token = token
            alt.logprob = logprob
            alts.append(alt)
        token_entry = MagicMock()
        token_entry.top_logprobs = alts
        choice.logprobs.content = [token_entry]
    response = MagicMock()
    response.choices = [choice]
    return response


def _install_fake_openai(monkeypatch, response):
    module = types.ModuleType("openai")
    client = MagicMock()
    client.chat.completions.create.return_value = response
    module.OpenAI = MagicMock(return_value=client)
    monkeypatch.setitem(sys.modules, "openai", module)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return client


def test_openai_reads_probabilities_from_logprobs(monkeypatch):
    import math

    response = _openai_response("A", [("A", math.log(0.8)), ("B", math.log(0.2))])
    _install_fake_openai(monkeypatch, response)
    m = OpenAIBackend("some-model").measure("prompt")
    assert m.chosen_letter == "A"
    assert m.probability_source == "logprobs"
    assert m.p_a == pytest.approx(0.8, abs=1e-6)
    assert m.p_b == pytest.approx(0.2, abs=1e-6)
    assert m.p_a_norm == pytest.approx(0.8, abs=1e-6)


def test_openai_falls_back_to_text_when_logprobs_are_absent(monkeypatch):
    _install_fake_openai(monkeypatch, _openai_response("B"))
    m = OpenAIBackend("some-model").measure("prompt")
    assert m.chosen_letter == "B"
    assert m.probability_source == "text"
    assert m.p_a is None and m.p_b is None


def test_openai_reasoning_effort_skips_the_logprobs_request(monkeypatch):
    """Reasoning models may reject logprobs, so we must not ask for them and lose the answer."""
    client = _install_fake_openai(monkeypatch, _openai_response("A"))
    OpenAIBackend("some-model", reasoning_effort="high").measure("prompt")
    sent = client.chat.completions.create.call_args.kwargs
    assert sent["reasoning_effort"] == "high"
    assert "logprobs" not in sent


def test_openai_retries_without_logprobs_when_the_model_refuses_them(monkeypatch):
    """A 400 on logprobs must cost the probabilities, not the whole run."""
    module = types.ModuleType("openai")
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        Exception("Invalid value for 'top_logprobs': must be less than or equal to 5."),
        _openai_response("B"),
    ]
    module.OpenAI = MagicMock(return_value=client)
    monkeypatch.setitem(sys.modules, "openai", module)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    m = OpenAIBackend("some-model").measure("prompt")
    assert m.chosen_letter == "B"
    assert m.probability_source == "text"
    # The retry must have dropped the offending parameters rather than resending them.
    retry = client.chat.completions.create.call_args.kwargs
    assert "logprobs" not in retry and "top_logprobs" not in retry


def test_openai_does_not_swallow_unrelated_errors(monkeypatch):
    """An auth or quota failure must surface, not be retried into a confusing second failure."""
    module = types.ModuleType("openai")
    client = MagicMock()
    client.chat.completions.create.side_effect = Exception("Incorrect API key provided")
    module.OpenAI = MagicMock(return_value=client)
    monkeypatch.setitem(sys.modules, "openai", module)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    with pytest.raises(Exception, match="Incorrect API key"):
        OpenAIBackend("some-model").measure("prompt")


def test_openai_requests_at_most_five_logprobs(monkeypatch):
    """The endpoint's ceiling. Asking for more is a 400 on every call."""
    client = _install_fake_openai(monkeypatch, _openai_response("A"))
    OpenAIBackend("some-model").measure("prompt")
    assert client.chat.completions.create.call_args.kwargs["top_logprobs"] <= 5


def test_openai_backend_cannot_be_used_mechanistically(monkeypatch):
    _install_fake_openai(monkeypatch, _openai_response("A"))
    assert OpenAIBackend("some-model").supports_mechanistic is False


# --------------------------------------------------------------------------- anthropic


def _anthropic_response(text, stop_reason="end_turn"):
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    response.stop_reason = stop_reason
    return response


def _install_fake_anthropic(monkeypatch, responses):
    module = types.ModuleType("anthropic")
    client = MagicMock()
    client.messages.create.side_effect = responses
    module.Anthropic = MagicMock(return_value=client)
    monkeypatch.setitem(sys.modules, "anthropic", module)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return client


def test_anthropic_returns_a_letter_and_no_probabilities(monkeypatch):
    """The Messages API exposes no logprobs; the backend must say so, not fabricate them."""
    _install_fake_anthropic(monkeypatch, [_anthropic_response("B")])
    m = AnthropicBackend("claude-opus-5").measure("prompt")
    assert m.chosen_letter == "B"
    assert m.p_a is None and m.p_b is None and m.letter_mass is None
    assert m.probability_source == "none"


def test_anthropic_sampling_estimates_a_proportion(monkeypatch):
    responses = [_anthropic_response(t) for t in ["A", "A", "A", "B"]]
    _install_fake_anthropic(monkeypatch, responses)
    m = AnthropicBackend("claude-opus-5", samples=4).measure("prompt")
    assert m.p_a == pytest.approx(0.75)
    assert m.p_b == pytest.approx(0.25)
    assert m.probability_source == "sampled_n4"
    assert m.chosen_letter == "A"


def test_anthropic_refusal_is_recorded_as_a_parse_failure(monkeypatch):
    """A safety refusal is missing data, not a vote for either option."""
    _install_fake_anthropic(monkeypatch, [_anthropic_response("", stop_reason="refusal")])
    m = AnthropicBackend("claude-opus-5").measure("prompt")
    assert m.chosen_letter is None
    assert m.parse_ok is False


def test_anthropic_sends_effort_rather_than_disabling_thinking(monkeypatch):
    """Disabled thinking on Opus 5 can leak tags or tool calls into the text we parse."""
    client = _install_fake_anthropic(monkeypatch, [_anthropic_response("A")])
    AnthropicBackend("claude-opus-5", effort="low").measure("prompt")
    sent = client.messages.create.call_args.kwargs
    assert sent["output_config"] == {"effort": "low"}
    assert "thinking" not in sent
    assert sent["model"] == "claude-opus-5"


# --------------------------------------------------------------------------- spec


def test_backend_spec_refuses_an_unknown_kind():
    with pytest.raises(ValueError, match="Unknown backend kind"):
        backends.BackendSpec(kind="bedrock", model_id="x").build()


def test_measurement_defaults_to_no_probabilities():
    m = Measurement(chosen_letter="A", generated="A")
    assert m.p_a is None and m.letter_mass is None and m.parse_ok is True


def test_anthropic_never_sends_sampling_parameters(monkeypatch):
    """Current Claude models reject temperature/top_p with a 400."""
    client = _install_fake_anthropic(monkeypatch, [_anthropic_response("A")])
    AnthropicBackend("claude-opus-5").measure("prompt")
    sent = client.messages.create.call_args.kwargs
    for banned in ("temperature", "top_p", "top_k"):
        assert banned not in sent


def test_anthropic_flags_a_degenerate_sampling_estimate(monkeypatch):
    """Every sample agreeing means sampling could not resolve a probability, not p=1.0."""
    _install_fake_anthropic(monkeypatch, [_anthropic_response("A") for _ in range(4)])
    m = AnthropicBackend("claude-opus-5", samples=4).measure("prompt")
    assert m.p_a == 1.0
    assert m.probability_source == "sampled_n4_degenerate"


def test_anthropic_sampling_is_not_flagged_when_it_actually_varies(monkeypatch):
    _install_fake_anthropic(monkeypatch, [_anthropic_response(t) for t in ["A", "A", "B", "A"]])
    m = AnthropicBackend("claude-opus-5", samples=4).measure("prompt")
    assert m.p_a == 0.75
    assert m.probability_source == "sampled_n4"
