"""The reasoning-block parser, against the completions that broke it.

Run 20260904T075505Z is Thomson-1.0-Small with reasoning on. It is not a result -- 113 of its
210 completions were truncated mid-thought -- but it is the only corpus of real completions in
this shape: a chat template that ends the prompt with a bare `<think>`, so the model generates
*inside* the block and emits an orphan `</think>` on the way out, with no opening tag anywhere.
That shape is what the original parser got wrong, and it is not reproducible from hand-written
strings, because the reasoning bodies genuinely say things like "option A says 28 days" three
lines before answering B.

`expected_letter` in the fixture was computed by an independent re-parse at extraction time,
not by the code under test.
"""

import gzip
import json
import pathlib

import pytest

from microscope.backends import _parse_letter, _reasoning_unfinished, _strip_reasoning

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "thomson_20260904T075505Z_completions.json.gz"


@pytest.fixture(scope="module")
def rows():
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as fh:
        return json.load(fh)["rows"]


def read_answer(completion):
    """The production path: strip the thought, then look for a letter in what is left."""
    stripped = _strip_reasoning(completion, reasoning_expected=True)
    return _parse_letter(stripped)[0]


def test_fixture_is_the_run_we_think_it_is(rows):
    assert len(rows) == 210
    assert sum(1 for r in rows if r["expected_letter"]) == 97
    # No completion carries an opening tag; every recoverable one carries a closing tag alone.
    assert not any("<think>" in r["completion"] for r in rows)
    assert sum(1 for r in rows if "</think>" in r["completion"]) == 97


def test_recovers_every_completed_answer(rows):
    wrong = [r for r in rows if r["expected_letter"] and read_answer(r["completion"]) != r["expected_letter"]]
    assert not wrong, f"{len(wrong)} completed answers misread, e.g. {wrong[:1]}"


def test_never_invents_an_answer_from_a_truncated_thought(rows):
    """The original bug. A thought cut off mid-sentence has no answer in it, and the parser
    must say so rather than lifting a letter out of the echoed option list."""
    invented = [r for r in rows if not r["expected_letter"] and read_answer(r["completion"]) is not None]
    assert not invented, f"{len(invented)} truncated completions produced a letter anyway"


def test_truncation_is_reported_as_truncation(rows):
    """Distinct from an unreadable answer: this one is fixable by raising the budget, and the
    run is only diagnosable if the two are not conflated."""
    for r in rows:
        assert _reasoning_unfinished(r["completion"], reasoning_expected=True) is (r["expected_letter"] is None)


def test_reproduces_the_original_misparse_count(rows):
    """39 of the 97 readable rows were recorded wrong at the time. If this number moves, the
    parser changed behaviour on real data -- deliberately or not."""
    fixed = [r for r in rows if r["expected_letter"] and r["old_chosen_letter"] != r["expected_letter"]]
    assert len(fixed) == 39
    assert all(read_answer(r["completion"]) == r["expected_letter"] for r in fixed)


def test_a_bare_answer_is_untouched_when_no_reasoning_is_expected():
    """The strip must not fire on a non-reasoning run, where the completion is just the answer."""
    assert _strip_reasoning("B", reasoning_expected=False) == "B"
    assert _parse_letter(_strip_reasoning("B", reasoning_expected=False))[0] == "B"
    # ...but the same bare text on a reasoning run is a truncated thought, not an answer.
    assert _strip_reasoning("B", reasoning_expected=True) == ""


def test_salvaging_this_run_would_be_selection_on_the_outcome(rows):
    """Not a parser test. It is here so that anyone who reconsiders analysing the 97 readable
    rows trips over the reason not to, in CI, rather than rediscovering it in a plot.

    Survival correlates with the arm being measured -- the stronger the cue, the longer the
    model reasons and the more likely it hit the budget -- and every row that survived is
    correct, so the salvaged set reports zero deference everywhere by construction.
    """
    by_arm = {}
    for r in rows:
        got, tot = by_arm.get(r["condition"], (0, 0))
        by_arm[r["condition"]] = (got + (r["expected_letter"] is not None), tot + 1)

    assert by_arm["floor"][0] == 26
    assert by_arm["court"][0] == 1
    readable = [r for r in rows if r["expected_letter"]]
    assert all(r["expected_letter"] == r["correct_letter"] for r in readable)
