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
