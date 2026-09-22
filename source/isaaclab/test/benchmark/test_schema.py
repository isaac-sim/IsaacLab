# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the benchmark bundle schema and its JSON serialization."""

import dataclasses
import json

import pytest

from isaaclab.benchmark.schema import SCHEMA_VERSION, Learning, LearningCurve, MeanStd, RunConfig, RunIdentity
from isaaclab.benchmark.serialize import write_bundle_file

pytestmark = pytest.mark.benchmark


def _round_trip(bundle, path) -> dict:
    write_bundle_file(bundle, str(path))
    with open(path) as f:
        return json.load(f)


def test_training_bundle_round_trip(tmp_path, training_bundle, serialized_step_timing):
    bundle = dataclasses.replace(
        training_bundle,
        run=dataclasses.replace(
            training_bundle.run,
            config=RunConfig(physics_backend="newton_mjwarp", rendering_backend="ovrtx", presets=["rgb", "ovrtx"]),
        ),
        runtime=dataclasses.replace(training_bundle.runtime, environment_step_timing=serialized_step_timing),
        extra={"grad_norm": 0.42, "note": "warmup", "stable": True, "restarts": 2},
    )

    data = _round_trip(bundle, tmp_path / "training.json")

    assert data["schema_version"] == SCHEMA_VERSION
    assert data["run"]["framework"] == "rsl_rl"
    assert data["run"]["config"] == {
        "physics_backend": "newton_mjwarp",
        "rendering_backend": "ovrtx",
        "presets": ["rgb", "ovrtx"],
    }
    timing = data["runtime"]["environment_step_timing"]
    assert timing["warmup_steps"] == 0
    assert timing["outside_simulation_step_fraction"] == pytest.approx(0.375)
    assert timing["measurement_mode"] == "serialized_synchronized"
    assert data["resources"]["gpu_util_pct"]["peak"] is None
    assert data["resources"]["ram_gb"]["peak"] == pytest.approx(24.0)
    assert data["learning"]["success_rate"]["series_per_iter"] == pytest.approx([0.1, 0.5, 0.95])
    assert data["success_rate"] == pytest.approx(0.75)
    assert data["checkpoint_path"].endswith("model_499.pt")
    assert data["video_path"] is None
    assert data["versions"]["sb3"] is None
    assert data["extra"] == {"grad_norm": 0.42, "note": "warmup", "stable": True, "restarts": 2}


def test_training_bundle_without_series(tmp_path, training_bundle):
    curve = LearningCurve(final_raw=1.0, final_ema=1.0, series_per_iter=None)
    bundle = dataclasses.replace(
        training_bundle, learning=Learning(ema_alpha=0.05, reward=curve, ep_length=curve, success_rate=curve)
    )

    learning = _round_trip(bundle, tmp_path / "training.json")["learning"]

    assert learning["reward"]["series_per_iter"] is None
    assert learning["ep_length"]["series_per_iter"] is None
    assert learning["success_rate"]["series_per_iter"] is None


def test_runtime_bundle_round_trip(tmp_path, runtime_bundle):
    data = _round_trip(runtime_bundle, tmp_path / "runtime.json")

    assert data["run"]["framework"] is None
    assert data["run"]["max_iterations"] is None
    assert data["run"]["config"]["presets"] == []
    assert data["extra"] is None
    assert "learning" not in data
    assert data["resources"]["gpu_mem_gb"]["peak"] == pytest.approx(12.0)


def test_play_bundle_round_trip(tmp_path, play_bundle):
    data = _round_trip(play_bundle, tmp_path / "play.json")

    assert data["run"]["framework"] == "rsl_rl"
    assert data["success_rate"] == pytest.approx(0.75)
    assert data["reward"]["peak"] == pytest.approx(5.0)
    assert data["ep_length"]["mean"] == pytest.approx(20.0)
    assert data["checkpoint_path"] == "model.pt"
    assert "learning" not in data


def test_startup_bundle_round_trip(tmp_path, startup_bundle):
    data = _round_trip(startup_bundle, tmp_path / "startup.json")

    assert data["run"]["num_envs"] is None
    assert data["phases"]["python_imports"]["top_functions"][0]["calls"] == 2
    assert data["phases"]["first_step"]["top_functions"] == []
    assert data["config"] == {"top_n": 1, "whitelist": None}


def test_environment_step_timing_validates_measurement_modes(serialized_step_timing):
    timing = serialized_step_timing

    with pytest.raises(ValueError, match="host_return timing cannot contain"):
        dataclasses.replace(timing, measurement_mode="host_return")
    with pytest.raises(ValueError, match="requires a complete simulation breakdown"):
        dataclasses.replace(timing, simulation_step_time_s=None)
    with pytest.raises(ValueError, match="must equal simulation plus outside-simulation time"):
        dataclasses.replace(
            timing,
            outside_simulation_step_time_s=MeanStd(mean=0.02, std=0.005, peak=0.04),
            outside_simulation_step_fraction=0.25,
        )
    with pytest.raises(ValueError, match="must match the aggregate timing ratio"):
        dataclasses.replace(timing, outside_simulation_step_fraction=0.3)
    host_return = dataclasses.replace(
        timing,
        measurement_mode="host_return",
        simulation_step_time_s=None,
        outside_simulation_step_time_s=None,
        outside_simulation_step_fraction=None,
        simulation_step_calls=None,
    )
    with pytest.raises(ValueError, match="environment step time and FPS must be greater than zero"):
        dataclasses.replace(host_return, environment_step_fps=MeanStd(mean=-1.0, std=0.0))


def test_field_validation():
    with pytest.raises(ValueError, match="peak"):
        MeanStd(mean=10.0, std=1.0, peak=5.0)
    with pytest.raises(ValueError, match="duration_s"):
        RunIdentity(
            run_id="x",
            framework=None,
            config=RunConfig(physics_backend="physx"),
            task="t",
            seed=0,
            start_time_utc="a",
            end_time_utc="b",
            duration_s=-1.0,
            status="crashed",
        )


def test_package_reexports_match_schema_module():
    import isaaclab.benchmark as pkg
    from isaaclab.benchmark import schema

    schema_names = {name for name in dir(schema) if not name.startswith("_")}
    checked = [name for name in pkg.__all__ if name in schema_names]
    assert checked
    for name in checked:
        assert getattr(pkg, name) is getattr(schema, name), name


def test_write_bundle_file_is_atomic(tmp_path, monkeypatch, runtime_bundle):
    """A failure mid-serialization must not clobber an existing good file."""
    import isaaclab.benchmark.serialize as serialize

    path = tmp_path / "runtime.json"
    write_bundle_file(runtime_bundle, str(path))
    good = path.read_text()

    def fail(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(serialize.json, "dump", fail)
    with pytest.raises(RuntimeError):
        write_bundle_file(runtime_bundle, str(path))

    assert path.read_text() == good
    assert not path.with_name("runtime.json.tmp").exists()
