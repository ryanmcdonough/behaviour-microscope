"""The dataset and the prompt pair. No model required."""

import pytest

from microscope.scenarios import (
    ARMS,
    ARMS_BY_NAME,
    CONDITIONS,
    DEFAULT_CONTRAST,
    FACTORIAL_CELLS,
    PLANNED_CONTRASTS,
    load_scenarios,
    stale_scenarios,
)

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


def test_asserting_arms_differ_only_in_the_cue_sentence():
    """Every assertion arm must be reachable from any other by swapping one sentence."""
    for s in SCENARIOS:
        prompts = s.prompts()
        base = prompts["junior_said"]
        for name, arm in ARMS_BY_NAME.items():
            if not arm.asserts or name == "junior_said":
                continue
            swapped = base.replace(ARMS_BY_NAME["junior_said"].cue, arm.cue)
            assert swapped == prompts[name], f"{s.id} / {name}"


def test_the_factorial_is_a_real_two_by_two():
    """Each cell must vary exactly one thing from its neighbours."""
    assert set(FACTORIAL_CELLS) == {
        ("junior", "said"), ("junior", "confirmed"),
        ("partner", "said"), ("partner", "confirmed"),
    }
    # Verb held constant down a column, source held constant along a row.
    for verb in ("said", "confirmed"):
        junior = ARMS_BY_NAME[FACTORIAL_CELLS[("junior", verb)]]
        partner = ARMS_BY_NAME[FACTORIAL_CELLS[("partner", verb)]]
        assert junior.verb == partner.verb == verb
        assert junior.source != partner.source


def test_the_floor_arm_asserts_nothing_but_still_offers_both_options():
    for s in SCENARIOS:
        floor = s.prompt("floor")
        assert "ADDITIONAL INFORMATION" not in floor, s.id
        assert s.correct_proposition in floor, s.id
        # The false proposition appears exactly once -- as an option, never as an assertion.
        assert floor.count(s.false_proposition) == 1, s.id


def test_the_default_contrast_holds_the_verb_constant():
    """The mechanistic pair must isolate source, or the mechanism answers the wrong question."""
    low, high = (ARMS_BY_NAME[name] for name in DEFAULT_CONTRAST)
    assert low.verb == high.verb
    assert low.source != high.source


def test_every_planned_contrast_names_real_arms():
    for arm_a, arm_b, isolates in PLANNED_CONTRASTS:
        assert arm_a in ARMS_BY_NAME, arm_a
        assert arm_b in ARMS_BY_NAME, arm_b
        assert isolates


def test_both_propositions_appear_as_options_in_every_arm():
    for s in SCENARIOS:
        for condition in CONDITIONS:
            prompt = s.prompt(condition)
            assert s.correct_proposition in prompt, s.id
            assert s.false_proposition in prompt, s.id
            expected = 2 if ARMS_BY_NAME[condition].asserts else 1
            assert prompt.count(s.false_proposition) == expected, f"{s.id}/{condition}"


def test_unknown_condition_is_refused():
    with pytest.raises(ValueError):
        SCENARIOS[0].prompt("supervising_partner")


def test_no_scenario_is_stale_before_its_supersession_date():
    assert stale_scenarios(SCENARIOS, "2026-12-31") == []
    superseded = [s for s in SCENARIOS if s.source.superseded_from]
    assert superseded, "at least one scenario should record a known future change in the law"
    assert stale_scenarios(SCENARIOS, "2027-06-01"), "the runner must notice when law has moved on"
