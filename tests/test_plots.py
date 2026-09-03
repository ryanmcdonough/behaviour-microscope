"""The figures render from run-shaped frames, including degenerate ones."""

import numpy as np
import pandas as pd

from microscope import plots


def _behavioural(n=6):
    rng = np.random.default_rng(1)
    rows = []
    for i in range(n):
        for condition in ("control", "partner"):
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
        for condition in ("control", "partner"):
            rows.append({"scenario_id": f"s{sid}", "arm": "baseline", "condition": condition,
                         "layer": -1, "p_false_normalised": 0.6})
        for layer in range(n_layers):
            rows.append({"scenario_id": f"s{sid}", "arm": "patch_forward", "condition": "partner",
                         "layer": layer, "p_false_normalised": 0.6 - 0.05 * layer})
            rows.append({"scenario_id": f"s{sid}", "arm": "patch_reverse", "condition": "control",
                         "layer": layer, "p_false_normalised": 0.6 + 0.03 * layer})
            rows.append({"scenario_id": f"s{sid}", "arm": "control_random", "condition": "partner",
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


def test_all_four_figures_are_written(tmp_path):
    paths = plots.write_all(_behavioural(), _divergence(), _interventions(), tmp_path)
    assert len(paths) == 4
    for path in paths:
        assert path.exists() and path.stat().st_size > 1000


def test_figures_render_when_the_control_arms_are_absent(tmp_path):
    """A run configured without the random control must still produce every plot."""
    interventions = _interventions()
    interventions = interventions[interventions.arm != "control_random"]
    paths = plots.write_all(_behavioural(), _divergence(), interventions, tmp_path)
    assert all(p.exists() for p in paths)
