"""The figures render from run-shaped frames, including degenerate ones."""

import numpy as np
import pandas as pd

from microscope import plots


def _behavioural(n=6):
    rng = np.random.default_rng(1)
    rows = []
    for i in range(n):
        for condition in ("junior_said", "partner_said"):
            p = float(rng.uniform(0.2, 0.8))
            rows.append(
                {
                    "scenario_id": f"scenario_{i:03d}",
                    "condition": condition,
                    "accepted_false_proposition": p > 0.5,
                    "p_false_normalised": p,
                }
            )
    return pd.DataFrame(rows)


def _interventions(n_layers=6):
    rows = []
    for sid in range(4):
        for condition in ("junior_said", "partner_said"):
            rows.append({"scenario_id": f"s{sid}", "arm": "baseline", "condition": condition,
                         "layer": -1, "p_false_normalised": 0.6})
        for layer in range(n_layers):
            rows.append({"scenario_id": f"s{sid}", "arm": "patch_forward", "condition": "partner_said",
                         "layer": layer, "p_false_normalised": 0.6 - 0.05 * layer})
            rows.append({"scenario_id": f"s{sid}", "arm": "patch_reverse", "condition": "junior_said",
                         "layer": layer, "p_false_normalised": 0.6 + 0.03 * layer})
            rows.append({"scenario_id": f"s{sid}", "arm": "control_random", "condition": "partner_said",
                         "layer": layer, "p_false_normalised": 0.6})
    return pd.DataFrame(rows)


def _divergence(n_layers=6):
    return pd.DataFrame(
        {
            "layer": range(n_layers),
            "mean_relative_l2": np.linspace(0.01, 0.2, n_layers),
            "ci_low": np.linspace(0.005, 0.18, n_layers),
            "ci_high": np.linspace(0.02, 0.22, n_layers),
        }
    )


class _Cfg:
    contrast = ("junior_said", "partner_said")


def test_every_figure_is_written_for_a_full_run(tmp_path):
    paths = plots.write_all(_behavioural(), _divergence(), _interventions(), tmp_path, _Cfg())
    assert len(paths) == 5
    for path in paths:
        assert path.exists() and path.stat().st_size > 1000


def test_figures_render_when_the_control_arms_are_absent(tmp_path):
    """A run configured without the random control must still produce every plot."""
    interventions = _interventions()
    interventions = interventions[interventions.arm != "control_random"]
    paths = plots.write_all(_behavioural(), _divergence(), interventions, tmp_path, _Cfg())
    assert all(p.exists() for p in paths)


def test_a_behavioural_only_run_skips_the_mechanistic_figures(tmp_path):
    """An API backend produces no activations; the run must still plot what it has."""
    paths = plots.write_all(_behavioural(), None, None, tmp_path, _Cfg())
    assert len(paths) == 2
    assert all(p.exists() for p in paths)
