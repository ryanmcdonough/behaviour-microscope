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


# --------------------------------------------------------------------------- reasoning mode


def test_reasoning_block_is_stripped_before_parsing():
    """Models say 'option A says...' while reasoning and then answer B."""
    from microscope.backends import _strip_reasoning
    text = "<think>Option A claims 28 days, but CPR 10.3 says 14.</think>\n\nB"
    assert _strip_reasoning(text) == "B"
    assert _parse_letter(_strip_reasoning(text)) == ("B", True)


def test_an_unclosed_reasoning_block_is_a_parse_failure_not_a_guess():
    """Generation that ran out of tokens mid-reasoning has no answer to find."""
    from microscope.backends import _strip_reasoning
    text = "<think>Option A claims 28 days, which would mean"
    assert _strip_reasoning(text) == ""
    assert _parse_letter(_strip_reasoning(text)) == (None, False)


def test_reasoning_is_stripped_when_only_the_closing_tag_is_in_the_completion():
    """Some templates end the prompt with a bare `<think>`, so the model emits only `</think>`.

    Thomson-1.0-Small does this. Requiring the opening tag scored the first letter of the
    echoed option list as the answer -- 152 of 210 rows wrong in run 20260904T075505Z.
    """
    from microscope.backends import _strip_reasoning
    text = "Options:\nA: 28 days\nB: 14 days\nFinal answer: B.\n</think>\n\nB"
    assert _strip_reasoning(text, reasoning_expected=True) == "B"
    assert _parse_letter(_strip_reasoning(text, reasoning_expected=True)) == ("B", True)


def test_untagged_completion_is_truncation_on_a_reasoning_run_and_an_answer_otherwise():
    """The same text means opposite things; only the caller knows which run this is."""
    from microscope.backends import _strip_reasoning
    text = "Options:\nA: 28 days\nB: 14 days\nThe text says 14, so"
    assert _strip_reasoning(text, reasoning_expected=True) == ""
    assert _strip_reasoning(text, reasoning_expected=False) == text


def test_a_bare_letter_still_parses_when_no_reasoning_is_expected():
    """The non-reasoning path must not be collateral damage."""
    from microscope.backends import _strip_reasoning
    assert _strip_reasoning("B", reasoning_expected=False) == "B"


def test_missing_closing_tag_is_reported_as_truncated_not_as_a_read_answer():
    """A budget failure and an unreadable answer are different problems."""
    from microscope.backends import _reasoning_unfinished
    assert _reasoning_unfinished("thinking, no close", reasoning_expected=True) is True
    assert _reasoning_unfinished("thought.</think>\n\nB", reasoning_expected=True) is False
    assert _reasoning_unfinished("B", reasoning_expected=False) is False
    assert _reasoning_unfinished("<think>unclosed", reasoning_expected=False) is True


def test_the_last_closing_tag_wins():
    """A model that writes about `</think>` mid-thought must not truncate its own answer."""
    from microscope.backends import _strip_reasoning
    text = "I should emit </think> when done.\nAnswer: B.\n</think>\n\nB"
    assert _strip_reasoning(text, reasoning_expected=True) == "B"


def test_thinking_on_a_model_without_a_reasoning_mode_is_a_no_op():
    """It must not silently cost the run its mechanistic half."""
    from microscope.backends import LocalBackend

    class _Handle:
        model_id = "x"; backend = "EagerModel"; n_layers = 4; d_model = 8
        enable_thinking = True
        template_controls = frozenset()          # no reasoning mode
        has_reasoning_mode = False

    b = LocalBackend(_Handle())
    assert b.response_mode == "logits"
    assert b.supports_mechanistic_now is True


def test_thinking_on_a_reasoning_model_switches_to_generate_and_drops_mechanistic():
    from microscope.backends import LocalBackend

    class _Handle:
        model_id = "x"; backend = "EagerModel"; n_layers = 4; d_model = 8
        enable_thinking = True
        template_controls = frozenset({"enable_thinking"})
        has_reasoning_mode = True

    b = LocalBackend(_Handle())
    assert b.response_mode == "generate"
    assert b.supports_mechanistic_now is False
    assert b.max_gen_tokens >= 256          # room to finish reasoning and still answer


def test_unparseable_answers_are_missing_not_refusals():
    """Scoring a non-answer as 'did not accept' biases every rate downward."""
    from microscope.experiment import _measurement_row
    from microscope.scenarios import load_scenarios

    s = load_scenarios()[0]
    row = _measurement_row(s, "partner_said", "prompt",
                           Measurement(chosen_letter=None, generated="", parse_ok=False))
    assert row["accepted_false_proposition"] is None
    assert row["correct"] is None

    answered = _measurement_row(s, "partner_said", "prompt",
                                Measurement(chosen_letter=s.false_letter, generated="B"))
    assert answered["accepted_false_proposition"] is True


def test_generation_budget_reaches_the_local_backend():
    """A budget set in RunConfig must not be silently dropped on the way to the backend."""
    from microscope.backends import BackendSpec

    spec = BackendSpec(kind="local", model_id="x", max_gen_tokens=4096)
    assert spec.max_gen_tokens == 4096


def test_reasoning_budget_floor_still_applies_when_none_is_given():
    from microscope.backends import LocalBackend

    class _H:
        model_id = "x"; backend = "EagerModel"; n_layers = 4; d_model = 8
        enable_thinking = True
        template_controls = frozenset({"enable_thinking"})
        has_reasoning_mode = True

    assert LocalBackend(_H()).max_gen_tokens >= 2048
    assert LocalBackend(_H(), max_gen_tokens=8192).max_gen_tokens == 8192


def test_the_qualitative_record_does_not_cost_the_full_answer_budget():
    """In logits mode the answer is the first token; the completion is only the transcript."""
    from microscope.backends import LocalBackend

    class _H:
        model_id = "x"; backend = "EagerModel"; n_layers = 4; d_model = 8
        enable_thinking = False
        template_controls = frozenset()
        has_reasoning_mode = False

    b = LocalBackend(_H(), max_gen_tokens=2048)
    assert b.response_mode == "logits"
    assert b.record_tokens < b.max_gen_tokens
    assert b.describe()["record_tokens"] == b.record_tokens
