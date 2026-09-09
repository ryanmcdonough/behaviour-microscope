"""The quality gate itself: does it actually catch what it claims to catch?"""

import numpy as np
import pandas as pd

from microscope import quality


def _behavioural(n=25, letter_mass=0.9, control_accuracy=0.8, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        control_correct = rng.random() < control_accuracy
        for condition in ("junior_said", "partner_said"):
            correct = control_correct if condition == "control" else bool(rng.random() < control_accuracy)
            rows.append(
                {
                    "scenario_id": f"scenario_{i:03d}",
                    "condition": condition,
                    "correct": correct,
                    "accepted_false_proposition": not correct,
                    "letter_mass": letter_mass,
                    "p_false_normalised": 0.2 if correct else 0.8,
                }
            )
    return pd.DataFrame(rows)


def _interventions(behavioural, layers=(3, 4, 5), real_shift=0.3, random_shift=0.02):
    rows = []
    base = {sid: 0.6 for sid in behavioural["scenario_id"].unique()}
    for sid, value in base.items():
        rows.append({"scenario_id": sid, "arm": "baseline", "condition": "partner_said", "layer": -1,
                     "p_false_normalised": value})
        rows.append({"scenario_id": sid, "arm": "baseline", "condition": "control", "layer": -1,
                     "p_false_normalised": value})
        for layer in layers:
            rows.append({"scenario_id": sid, "arm": "patch_forward", "condition": "partner_said",
                         "layer": layer, "p_false_normalised": value - real_shift})
            rows.append({"scenario_id": sid, "arm": "patch_reverse", "condition": "junior_said",
                         "layer": layer, "p_false_normalised": value + real_shift})
            rows.append({"scenario_id": sid, "arm": "control_random", "condition": "partner_said",
                         "layer": layer, "p_false_normalised": value - random_shift})
            rows.append({"scenario_id": sid, "arm": "control_zero", "condition": "partner_said",
                         "layer": layer, "p_false_normalised": value})
    return pd.DataFrame(rows)


def _divergence(n_layers=8):
    return pd.DataFrame({"layer": range(n_layers), "mean_relative_l2": np.linspace(0.01, 0.1, n_layers)})


def _summary(interventions):
    zero = interventions[interventions.arm == "control_zero"]
    baseline = interventions[(interventions.arm == "baseline") & (interventions.condition == "partner_said")]
    merged = zero.merge(baseline, on="scenario_id", suffixes=("_zero", "_base"))
    deltas = (merged["p_false_normalised_zero"] - merged["p_false_normalised_base"]).abs()
    return {
        "intervention_controls": {
            "control_zero": {"n": int(len(deltas)), "max_abs_change_in_p_false": float(deltas.max())}
        }
    }


def _manifest(stale=None, logit_ok=True):
    return {
        "logit_path_check": {"checked": True, "within_tolerance": logit_ok, "max_abs_prob_difference": 1e-5,
                              "argmax_agrees": True},
        "stale_scenarios": stale or [],
    }


def test_a_healthy_run_passes_every_check():
    behavioural = _behavioural(n=30, letter_mass=0.9, control_accuracy=0.85)
    interventions = _interventions(behavioural, real_shift=0.3, random_shift=0.01)
    report = quality.run_checks(behavioural, _divergence(), interventions, _summary(interventions), _manifest())
    assert report["overall"] == "pass", quality.format_report(report)


def test_control_accuracy_at_chance_fails():
    behavioural = _behavioural(n=30, control_accuracy=0.5, seed=1)
    interventions = _interventions(behavioural)
    report = quality.run_checks(behavioural, _divergence(), interventions, _summary(interventions), _manifest())
    statuses = {c["name"]: c["status"] for c in report["checks"]}
    assert statuses["control_accuracy"] == "fail"
    assert report["overall"] == "fail"


def test_low_letter_mass_fails():
    behavioural = _behavioural(n=30, letter_mass=0.01, control_accuracy=0.9)
    interventions = _interventions(behavioural)
    report = quality.run_checks(behavioural, _divergence(), interventions, _summary(interventions), _manifest())
    statuses = {c["name"]: c["status"] for c in report["checks"]}
    assert statuses["letter_mass"] == "fail"


def test_a_nonzero_zero_patch_fails():
    behavioural = _behavioural(n=30, control_accuracy=0.9)
    interventions = _interventions(behavioural)
    summary = _summary(interventions)
    summary["intervention_controls"]["control_zero"]["max_abs_change_in_p_false"] = 0.05
    report = quality.run_checks(behavioural, _divergence(), interventions, summary, _manifest())
    statuses = {c["name"]: c["status"] for c in report["checks"]}
    assert statuses["zero_patch_noop"] == "fail"


def test_random_control_as_big_as_the_real_effect_is_flagged():
    behavioural = _behavioural(n=30, control_accuracy=0.9, seed=2)
    interventions = _interventions(behavioural, real_shift=0.1, random_shift=0.09)
    report = quality.run_checks(behavioural, _divergence(), interventions, _summary(interventions), _manifest())
    statuses = {c["name"]: c["status"] for c in report["checks"]}
    assert statuses["random_control"] in {"warn", "fail"}


def test_small_sample_warns_not_fails():
    behavioural = _behavioural(n=12, control_accuracy=0.9, seed=3)
    interventions = _interventions(behavioural)
    report = quality.run_checks(behavioural, _divergence(), interventions, _summary(interventions), _manifest())
    statuses = {c["name"]: c["status"] for c in report["checks"]}
    assert statuses["sample_size"] == "warn"
    assert report["overall"] != "pass"


def test_stale_scenarios_are_surfaced_as_a_warning_not_silently_dropped():
    behavioural = _behavioural(n=30, control_accuracy=0.9, seed=4)
    interventions = _interventions(behavioural)
    manifest = _manifest(stale=["scenario_019"])
    report = quality.run_checks(behavioural, _divergence(), interventions, _summary(interventions), manifest)
    statuses = {c["name"]: c["status"] for c in report["checks"]}
    assert statuses["stale_ground_truth"] == "warn"


def test_nan_in_any_frame_fails():
    behavioural = _behavioural(n=30, control_accuracy=0.9, seed=5)
    interventions = _interventions(behavioural)
    interventions.loc[0, "p_false_normalised"] = float("nan")
    report = quality.run_checks(behavioural, _divergence(), interventions, _summary(interventions), _manifest())
    statuses = {c["name"]: c["status"] for c in report["checks"]}
    assert statuses["finite_values"] == "fail"


def test_review_reads_back_a_written_run_directory(tmp_path):
    behavioural = _behavioural(n=30, control_accuracy=0.9, seed=6)
    interventions = _interventions(behavioural)
    divergence = _divergence()
    summary = _summary(interventions)
    manifest = _manifest()

    import json
    behavioural.to_csv(tmp_path / "behavioural.csv", index=False)
    divergence.to_csv(tmp_path / "activation_analysis.csv", index=False)
    interventions.to_csv(tmp_path / "interventions.csv", index=False)
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    report = quality.review(tmp_path)
    assert report["overall"] == "pass"
