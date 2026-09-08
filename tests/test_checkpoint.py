"""Incremental saves, resume, STOP, and finalize-without-a-model."""

from pathlib import Path

import pandas as pd

from microscope.backends import Measurement
from microscope.experiment import (
    RunConfig,
    finalize_run,
    run_behavioural,
    write_run_artifacts,
)
from microscope.scenarios import load_scenarios


class FakeBackend:
    name = "fake"
    model_id = "fake"
    supports_mechanistic = False
    concurrency_safe = False

    def __init__(self, stop_dir: Path | None = None, stop_after: int | None = None):
        self.calls = 0
        self.stop_dir = stop_dir
        self.stop_after = stop_after
        self.prompts: list[str] = []

    def measure(self, prompt: str) -> Measurement:
        self.calls += 1
        self.prompts.append(prompt)
        if (
            self.stop_after is not None
            and self.calls >= self.stop_after
            and self.stop_dir is not None
        ):
            (self.stop_dir / "STOP").write_text("test\n")
        return Measurement(
            chosen_letter="B", generated="B", parse_ok=True, probability_source="text",
        )

    def describe(self) -> dict:
        return {"backend": "fake", "model": "fake"}

    def shutdown(self) -> None:
        return None


def _cfg(tmp_path: Path, **kwargs) -> RunConfig:
    return RunConfig(
        model_id="fake",
        provider="openai",
        mechanistic=False,
        arms=("floor", "adverse"),
        limit=2,
        results_root=tmp_path,
        run_name="ckpt",
        **kwargs,
    )


def test_each_measurement_is_on_disk_before_the_next(tmp_path: Path):
    scenarios = load_scenarios()[:2]
    cfg = _cfg(tmp_path)
    run_dir = tmp_path / "ckpt"
    run_dir.mkdir()
    backend = FakeBackend()
    frame = run_behavioural(backend, scenarios, cfg, run_dir=run_dir)
    assert backend.calls == 4
    assert len(frame) == 4
    saved = pd.read_csv(run_dir / "behavioural.csv")
    assert len(saved) == 4
    assert list(saved["condition"]) == ["floor", "adverse", "floor", "adverse"]


def test_stop_file_halts_after_the_current_item(tmp_path: Path):
    scenarios = load_scenarios()[:2]
    cfg = _cfg(tmp_path)
    run_dir = tmp_path / "ckpt"
    run_dir.mkdir()
    backend = FakeBackend(stop_dir=run_dir, stop_after=2)
    frame = run_behavioural(backend, scenarios, cfg, run_dir=run_dir)
    assert backend.calls == 2
    assert len(frame) == 2
    assert len(pd.read_csv(run_dir / "behavioural.csv")) == 2


def test_resume_skips_rows_already_on_disk(tmp_path: Path):
    scenarios = load_scenarios()[:2]
    cfg = _cfg(tmp_path)
    run_dir = tmp_path / "ckpt"
    run_dir.mkdir()
    first = FakeBackend(stop_dir=run_dir, stop_after=2)
    run_behavioural(first, scenarios, cfg, run_dir=run_dir)
    (run_dir / "STOP").unlink()

    second = FakeBackend()
    frame = run_behavioural(second, scenarios, cfg, run_dir=run_dir)
    assert second.calls == 2
    assert len(frame) == 4
    assert len(pd.read_csv(run_dir / "behavioural.csv")) == 4


def test_start_at_redoes_from_that_index(tmp_path: Path):
    scenarios = load_scenarios()[:2]
    cfg = _cfg(tmp_path)
    run_dir = tmp_path / "ckpt"
    run_dir.mkdir()
    run_behavioural(FakeBackend(), scenarios, cfg, run_dir=run_dir)

    redo = FakeBackend()
    cfg.start_at = 3
    frame = run_behavioural(redo, scenarios, cfg, run_dir=run_dir)
    assert redo.calls == 2
    assert len(frame) == 4


def test_finalize_writes_plots_without_measuring(tmp_path: Path):
    scenarios = load_scenarios()[:2]
    cfg = _cfg(tmp_path)
    run_dir = tmp_path / "ckpt"
    run_dir.mkdir()
    run_behavioural(FakeBackend(), scenarios, cfg, run_dir=run_dir)

    backend = FakeBackend()
    write_run_artifacts(run_dir, cfg, backend=backend, verbose=False)
    assert backend.calls == 0
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "quality_report.json").exists()
    assert (run_dir / "plots" / "1_behaviour.png").exists()

    again = FakeBackend()
    out = finalize_run(run_dir, cfg, verbose=False)
    assert out == run_dir
    assert again.calls == 0
