"""The dataset and the prompt pair. No model required."""

import pytest

from microscope.scenarios import CONDITIONS, CUES, load_scenarios, stale_scenarios

SCENARIOS = load_scenarios()


def test_thirty_scenarios_with_unique_ids():
    assert len(SCENARIOS) == 30
    assert len({s.id for s in SCENARIOS}) == 30


def test_every_scenario_carries_its_provenance():
    for s in SCENARIOS:
        assert s.source.url.startswith("https://"), s.id
        assert s.source.title, s.id
        assert s.source.retrieved, s.id


def test_jurisdiction_is_held_constant():
    assert {s.jurisdiction for s in SCENARIOS} == {"england_wales"}


def test_correct_letter_is_stable_and_not_all_one_side():
    letters = [s.correct_letter for s in SCENARIOS]
    assert set(letters) == {"A", "B"}
    assert 8 <= letters.count("A") <= 22, "answer key is too lopsided to rule out letter bias"
    assert [s.correct_letter for s in load_scenarios()] == letters, "assignment must be deterministic"


def test_prompt_pair_differs_only_in_the_authority_cue():
    for s in SCENARIOS:
        pair = s.prompt_pair()
        control, partner = pair["control"], pair["partner"]
        assert control != partner
        assert control.replace(CUES["control"], CUES["partner"]) == partner, s.id


def test_both_propositions_appear_as_options_in_both_conditions():
    for s in SCENARIOS:
        for condition in CONDITIONS:
            prompt = s.prompt(condition)
            assert s.correct_proposition in prompt, s.id
            assert s.false_proposition in prompt, s.id
            assert prompt.count(s.false_proposition) == 2, "cue sentence + option"


def test_unknown_condition_is_refused():
    with pytest.raises(ValueError):
        SCENARIOS[0].prompt("supervising_partner")


def test_no_scenario_is_stale_before_its_supersession_date():
    assert stale_scenarios(SCENARIOS, "2026-12-31") == []
    superseded = [s for s in SCENARIOS if s.source.superseded_from]
    assert superseded, "at least one scenario should record a known future change in the law"
    assert stale_scenarios(SCENARIOS, "2027-06-01"), "the runner must notice when law has moved on"
