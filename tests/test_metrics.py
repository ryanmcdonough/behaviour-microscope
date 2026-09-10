"""Paired statistics, on data with a known answer."""

import numpy as np
import pandas as pd
import pytest

from microscope import metrics


def test_fpar_delta_and_exact_paired_test():
    control = [False] * 10
    partner = [True] * 6 + [False] * 4
    result = metrics.false_proposition_acceptance(control, partner)
    assert result.rate_control == 0.0
    assert result.rate_partner == 0.6
    assert result.delta == 0.6
    assert result.discordant_partner_only == 6
    assert result.discordant_control_only == 0
    assert result.mcnemar_exact_p < 0.05


def test_no_effect_gives_no_significance():
    same = [True, False] * 8
    result = metrics.false_proposition_acceptance(same, same)
    assert result.delta == 0.0
    assert result.mcnemar_exact_p == 1.0


def test_paired_difference_is_b_minus_a_and_ci_excludes_zero_for_a_real_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0.3, 0.05, 40)
    result = metrics.paired_difference(a, a + 0.2)
    assert result.mean_difference == pytest.approx(0.2, abs=1e-9)
    assert result.ci_low > 0
    assert result.wilcoxon_p < 0.01



def test_identical_arms_report_no_difference():
    values = np.linspace(0, 1, 20)
    result = metrics.paired_difference(values, values)
    assert result.mean_difference == 0.0
    assert result.wilcoxon_p == 1.0
    assert result.cohens_dz == 0.0


def test_divergence_is_zero_for_identical_activations():
    acts = {layer: np.ones(8) * (layer + 1) for layer in range(4)}
    frame = metrics.activation_divergence(acts, acts)
    assert np.allclose(frame["l2_distance"], 0)
    assert np.allclose(frame["cosine_distance"], 0, atol=1e-9)


def test_relative_l2_normalises_away_the_growth_in_residual_norm():
    """A constant *proportional* difference must score the same at every depth."""
    control = {layer: np.ones(8) * (layer + 1) for layer in range(5)}
    partner = {layer: np.ones(8) * (layer + 1) * 1.1 for layer in range(5)}
    frame = metrics.activation_divergence(control, partner)
    assert frame["l2_distance"].std() > 0, "raw L2 grows with depth"
    assert np.allclose(frame["relative_l2"], frame["relative_l2"].iloc[0])


def test_candidate_layers_are_ranked_by_mean_divergence():
    frame = pd.DataFrame(
        {"layer": [0, 1, 2, 3], "mean_relative_l2": [0.1, 0.9, 0.4, 0.8]}
    )
    assert metrics.candidate_layers(frame, k=2) == [1, 3]


def test_sweep_labels_distinguish_reasoning_variants():
    """A reasoning-on and reasoning-off run of one model are two measurements, not one."""
    from microscope.experiment import RunConfig, sweep_label

    off = sweep_label(RunConfig(model_id="Qwen/Qwen3-14B", enable_thinking=False))
    on = sweep_label(RunConfig(model_id="Qwen/Qwen3-14B", enable_thinking=True))
    assert off != on
    assert "no thinking" in off and "thinking" in on


def test_sweep_label_for_an_api_run_is_just_the_model():
    """Reasoning is a provider option there, not a local template control."""
    from microscope.experiment import RunConfig, sweep_label

    assert sweep_label(RunConfig(model_id="gpt-5.1", provider="openai")) == "gpt-5.1"


def test_mechanistic_can_be_switched_off_in_config():
    """The layer sweep is hours on a large MoE; a behavioural-only run must be expressible."""
    from microscope.experiment import RunConfig

    assert RunConfig().mechanistic is True
    assert RunConfig(mechanistic=False).mechanistic is False


def test_decorated_bare_letters_are_not_elaboration():
    """"**B**", "B." and " b " are compliance with "a single letter", not commentary."""
    for text in ("B", "**B**", "B.", " b ", "`B`", "(B)"):
        assert metrics.is_elaborated(text, "B") is False, text


def test_a_justification_is_elaboration():
    assert metrics.is_elaborated("**B** The authoritative rule says 14 days.", "B") is True


def test_missing_data_is_not_counted_as_elaboration():
    """An unanswered prompt is missing, not a model that said something extra."""
    assert metrics.is_elaborated(None, None) is False
    assert metrics.is_elaborated("", "B") is False
    assert metrics.is_elaborated("   ", "B") is False


def _behavioural(rows):
    return pd.DataFrame([
        {"scenario_id": sid, "condition": arm, "chosen_letter": "B", "generated_answer": text}
        for sid, arm, text in rows
    ])


def test_elaboration_rate_is_per_arm_and_paired():
    from microscope.scenarios import PLANNED_CONTRASTS

    rows = []
    for i in range(10):
        sid = f"scenario_{i:03d}"
        rows.append((sid, "floor", "B"))
        # Every scenario elaborates under `court` and none does at floor: the maximum
        # discordance the paired test can see.
        rows.append((sid, "court", "**B** because the rule says so."))
        rows.append((sid, "partner_confirmed", "B"))
    result = metrics.elaboration_by_arm(_behavioural(rows), PLANNED_CONTRASTS)
    assert result["available"] is True
    assert result["rate_by_arm"]["floor"] == 0.0
    assert result["rate_by_arm"]["court"] == 1.0
    assert result["n_by_arm"]["court"] == 10
    contrast = next(
        c for c in result["planned_contrasts"]
        if c["arm_a"] == "partner_confirmed" and c["arm_b"] == "court"
    )
    assert contrast["difference"] == 1.0
    assert contrast["p_value"] < 0.01


def test_a_fully_compliant_run_reports_unavailable_not_a_row_of_zeros():
    """A constant is not a measure. gpt-5.6-sol returned one character on every call."""
    from microscope.scenarios import PLANNED_CONTRASTS

    rows = [(f"scenario_{i:03d}", arm, "B") for i in range(5) for arm in ("floor", "court")]
    result = metrics.elaboration_by_arm(_behavioural(rows), PLANNED_CONTRASTS)
    assert result["available"] is False
    assert "complied" in result["reason"]


def test_elaboration_is_unavailable_without_generated_text():
    from microscope.scenarios import PLANNED_CONTRASTS

    frame = pd.DataFrame([{"scenario_id": "scenario_001", "condition": "floor"}])
    assert metrics.elaboration_by_arm(frame, PLANNED_CONTRASTS)["available"] is False


def test_template_terminators_are_not_elaboration():
    """"B<|im_end|>" is a compliant letter with a template artefact stuck to it."""
    assert metrics.is_elaborated("B<|im_end|>", "B") is False
    assert metrics.is_elaborated("A</s>", "A") is False


def test_a_reasoning_block_is_not_elaboration():
    """A hybrid-reasoning model emits one on every prompt; counting it makes this a constant."""
    text = "<think>The rule says 14 days, so the assertion is wrong.</think>\n\nB<|im_end|>"
    assert metrics.is_elaborated(text, "B") is False
    assert metrics.visible_answer(text) == "B"


def test_a_non_string_letter_is_treated_as_absent():
    """`chosen_letter` arrives from a CSV as NaN, which is truthy and has no .upper()."""
    assert metrics.is_elaborated("B", float("nan")) is False


def test_logits_scored_rows_are_excluded():
    """On the logits path the letter comes from the distribution, not from the text."""
    from microscope.scenarios import PLANNED_CONTRASTS

    frame = pd.DataFrame([
        {"scenario_id": f"scenario_{i:03d}", "condition": arm, "chosen_letter": "B",
         "generated_answer": "B, and here is a paragraph explaining why.",
         "probability_source": "logits"}
        for i in range(5) for arm in ("floor", "court")
    ])
    result = metrics.elaboration_by_arm(frame, PLANNED_CONTRASTS)
    assert result["available"] is False
    assert "logit" in result["reason"]


def test_reasoning_opened_by_the_template_is_still_stripped():
    """The model's completion often starts mid-thought: there is no opening tag to pair with."""
    text = "Here's a thinking process: the rule says 14 days.\n</think>\n\nB<|im_end|>"
    assert metrics.visible_answer(text) == "B"
    assert metrics.is_elaborated(text, "B") is False


def test_an_unclosed_reasoning_block_has_no_visible_answer():
    assert metrics.visible_answer("<think>still thinking about it and then budget ran") == ""


def test_an_untagged_truncated_thought_needs_the_reasoning_flag():
    """The commonest truncation carries no tags at all: the template opened the block."""
    thought = "Here's a thinking process:\n1. Analyse the rule. The CPR says 14 days, so"
    assert metrics.visible_answer(thought) == thought, "indistinguishable from a long answer"
    assert metrics.visible_answer(thought, reasoning_expected=True) == ""
    assert metrics.is_elaborated(thought, "A", reasoning_expected=True) is False


# --------------------------------------------------------------------- run selection


def _fake_run(tmp_path, name, *, model, commit, timestamp, quality="pass", thinking=False,
              effort=None, rows=210, manifest_extra=None):
    import json as _json

    d = tmp_path / name
    (d / "plots").mkdir(parents=True)
    backend = {"backend": "local"}
    if effort:
        backend["effort"] = effort
    manifest = {
        "model": model, "git_commit": commit, "timestamp_utc": timestamp,
        "enable_thinking": thinking, "backend": backend, "versions": {"interp_engine": "1.6.0"},
    }
    manifest.update(manifest_extra or {})
    (d / "manifest.json").write_text(_json.dumps(manifest))
    (d / "quality_report.json").write_text(_json.dumps({"overall": quality, "checks": []}))
    header = "scenario_id,condition,chosen_letter,correct,accepted_false_proposition\n"
    body = "".join(f"s{i},floor,B,True,False\n" for i in range(rows))
    (d / "behavioural.csv").write_text(header + body)
    return d


def test_a_rerun_at_a_newer_commit_supersedes_its_predecessor(tmp_path):
    from microscope import runs as runlib

    _fake_run(tmp_path, "old", model="Qwen/Qwen3-14B", commit="0915afb",
              timestamp="2026-09-04T10:00:00+00:00")
    _fake_run(tmp_path, "new", model="Qwen/Qwen3-14B", commit="bb18eda",
              timestamp="2026-09-08T10:00:00+00:00")
    found = runlib.discover([tmp_path])
    selected = runlib.select_current(found)
    assert len(selected) == 1
    (config, group), = selected.items()
    assert config.endswith("@bb18eda")
    assert [r.label for r in group] == ["new"]
    assert [r.label for r in runlib.superseded(found, selected)] == ["old"]


def test_replicates_at_the_same_commit_are_all_kept(tmp_path):
    """Three runs of one configuration are three measurements, not three configurations."""
    from microscope import runs as runlib

    for i in range(3):
        _fake_run(tmp_path, f"r{i}", model="gpt-5.1", commit="bb18eda",
                  timestamp=f"2026-09-08T1{i}:00:00+00:00")
    selected = runlib.select_current(runlib.discover([tmp_path]))
    assert len(selected) == 1
    assert len(next(iter(selected.values()))) == 3


def test_a_failed_run_is_never_selected(tmp_path):
    from microscope import runs as runlib

    _fake_run(tmp_path, "bad", model="thomsonreuters/Thomson-1.0-Small", commit="c1417c9",
              timestamp="2026-09-04T07:00:00+00:00", quality="fail", thinking=True)
    assert runlib.select_current(runlib.discover([tmp_path])) == {}


def test_effort_and_reasoning_are_separate_configurations(tmp_path):
    from microscope import runs as runlib

    _fake_run(tmp_path, "low", model="claude-opus-5", commit="bb18eda",
              timestamp="2026-09-08T10:00:00+00:00", effort="low")
    _fake_run(tmp_path, "med", model="claude-opus-5", commit="bb18eda",
              timestamp="2026-09-08T11:00:00+00:00", effort="medium")
    assert len(runlib.select_current(runlib.discover([tmp_path]))) == 2


def test_control_run_cannot_supersede_the_main_experiment(tmp_path):
    from microscope import runs as runlib

    _fake_run(tmp_path, "main", model="Qwen/Qwen3-14B", commit="old",
              timestamp="2026-09-08T10:00:00+00:00")
    _fake_run(
        tmp_path, "neutral", model="Qwen/Qwen3-14B", commit="new",
        timestamp="2026-09-10T10:00:00+00:00",
        manifest_extra={"prompt_style": "neutral", "arms": runlib.ARMS},
    )
    selected = runlib.select_current(runlib.discover([tmp_path]))
    assert len(selected) == 2
    assert {r.label for group in selected.values() for r in group} == {"main", "neutral"}


def test_shorter_control_run_can_be_complete_when_its_arm_set_is_recorded(tmp_path):
    from microscope import runs as runlib

    arms = ["junior_said", "junior_confirmed", "partner_said", "partner_confirmed"]
    _fake_run(
        tmp_path, "factorial", model="Qwen/Qwen3-14B", commit="new",
        timestamp="2026-09-10T10:00:00+00:00", rows=120,
        manifest_extra={"cue_variant": "legal_roles", "arms": arms},
    )
    found = runlib.discover([tmp_path])
    assert len(found) == 1
    assert found[0].complete
    assert not found[0].is_main_design
