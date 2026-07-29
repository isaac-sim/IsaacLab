# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deriving per-task metadata from a completed dispatch's bundles."""

import json
from pathlib import Path

import pytest
import yaml

from tools.odin.harvest import HarvestError, harvest_dispatch, write_task_metadata


def _write_bundle(
    dispatch_dir: Path,
    row_key: str,
    *,
    task: str = "Isaac-Ant",
    framework: str = "rsl_rl",
    physics: str = "physx",
    seed: int = 42,
    num_envs: int = 4096,
    max_iterations: int = 500,
    duration_s: float = 300.0,
    reward: float | None = 12.5,
    status: str = "completed",
) -> Path:
    row_dir = dispatch_dir / row_key
    row_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.2",
        "run": {
            "run_id": row_key,
            "framework": framework,
            "config": {"physics_backend": physics, "rendering_backend": "none", "presets": []},
            "task": task,
            "seed": seed,
            "duration_s": duration_s,
            "status": status,
            "num_envs": num_envs,
            "max_iterations": max_iterations,
        },
        "runtime": {"total_wall_time_s": duration_s},
    }
    if reward is not None:
        payload["learning"] = {"ema_alpha": 0.1, "reward": {"final_raw": reward, "final_ema": reward}}
    (row_dir / f"benchmark_training_{task}_2026-07-28_12-00-00.json").write_text(json.dumps(payload))
    return row_dir


def test_harvest_groups_seeds_of_one_task(tmp_path: Path) -> None:
    _write_bundle(tmp_path, "row42", seed=42, duration_s=300.0, reward=10.0)
    _write_bundle(tmp_path, "row43", seed=43, duration_s=400.0, reward=20.0)

    entries = harvest_dispatch(tmp_path)

    assert len(entries) == 1
    assert entries[0].seeds == [42, 43]


def test_timeout_uses_the_slowest_seed_with_headroom(tmp_path: Path) -> None:
    _write_bundle(tmp_path, "row42", seed=42, duration_s=300.0)
    _write_bundle(tmp_path, "row43", seed=43, duration_s=400.0)

    entry = harvest_dispatch(tmp_path, timeout_headroom=2.0)[0]

    assert entry.duration_s_max == 400.0
    assert entry.timeout_s == 800


def test_reward_baseline_spans_seeds(tmp_path: Path) -> None:
    _write_bundle(tmp_path, "row42", seed=42, reward=10.0)
    _write_bundle(tmp_path, "row43", seed=43, reward=20.0)

    entry = harvest_dispatch(tmp_path)[0]

    assert entry.reward_final_ema_mean == 15.0
    assert entry.reward_final_ema_min == 10.0


def test_resolved_sizing_is_recorded(tmp_path: Path) -> None:
    # This is the whole point: the seed list omitted sizing, so the only place
    # the real values exist is the emitted bundle.
    _write_bundle(tmp_path, "row", num_envs=2048, max_iterations=750)
    entry = harvest_dispatch(tmp_path)[0]
    assert entry.num_envs == 2048
    assert entry.max_iterations == 750


def test_distinct_physics_backends_stay_separate(tmp_path: Path) -> None:
    _write_bundle(tmp_path, "physx_row", physics="physx")
    _write_bundle(tmp_path, "newton_row", physics="newton_mjwarp")

    entries = harvest_dispatch(tmp_path)

    assert {entry.physics for entry in entries} == {"physx", "newton_mjwarp"}


def test_crashed_runs_are_excluded(tmp_path: Path) -> None:
    # A crashed run's duration describes the failure, not the task.
    _write_bundle(tmp_path, "ok", seed=42, duration_s=300.0, status="completed")
    _write_bundle(tmp_path, "bad", seed=43, duration_s=9999.0, status="crashed")

    entry = harvest_dispatch(tmp_path)[0]

    assert entry.seeds == [42]
    assert entry.duration_s_max == 300.0


def test_runs_without_a_reward_still_yield_sizing(tmp_path: Path) -> None:
    _write_bundle(tmp_path, "row", reward=None)
    entry = harvest_dispatch(tmp_path)[0]
    assert entry.reward_final_ema_mean is None
    assert entry.num_envs == 4096


def test_unreadable_rows_are_skipped(tmp_path: Path) -> None:
    _write_bundle(tmp_path, "good")
    (tmp_path / "junk").mkdir()
    (tmp_path / "junk" / "benchmark_training_x.json").write_text("{broken")

    assert len(harvest_dispatch(tmp_path)) == 1


def test_empty_dispatch_yields_nothing(tmp_path: Path) -> None:
    assert harvest_dispatch(tmp_path) == []


def test_non_positive_headroom_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(HarvestError, match="timeout_headroom"):
        harvest_dispatch(tmp_path, timeout_headroom=0)


def test_unreadable_dispatch_state_is_fatal(tmp_path: Path) -> None:
    # Swallowing it empties side_b_rows, which averages an A/B dispatch's two
    # commits into one baseline -- exactly what excluding side B prevents.
    _write_bundle(tmp_path, "row_a")
    (tmp_path / "dispatch.json").write_text("{not json")

    with pytest.raises(HarvestError, match="dispatch.json"):
        harvest_dispatch(tmp_path)


def test_written_metadata_is_a_valid_seed_list_overlay(tmp_path: Path) -> None:
    _write_bundle(tmp_path, "row")
    out = tmp_path / "out" / "task_metadata.yaml"

    write_task_metadata(out, harvest_dispatch(tmp_path))

    payload = yaml.safe_load(out.read_text())
    entry = payload["tasks"][0]
    # The overlay must carry every field plan_rows() reads.
    assert {"task_id", "rl_library", "physics", "num_envs", "max_iterations", "timeout_s"} <= set(entry)
    assert entry["observed"]["seeds"] == [42]


def test_ab_side_b_rows_are_excluded(tmp_path: Path) -> None:
    # Harvesting both commits into one baseline would silently average them.
    _write_bundle(tmp_path, "row_a", seed=42, duration_s=100.0, reward=10.0)
    _write_bundle(tmp_path, "row_b", seed=42, duration_s=900.0, reward=99.0)
    (tmp_path / "dispatch.json").write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "dispatch_id": "d",
                "started_at": "x",
                "ended_at": None,
                "seeds": [42],
                "images": {},
                "osmo_workflow_ids": [],
                "jobs": [{"row_key": "row_a", "side": "a"}, {"row_key": "row_b", "side": "b"}],
            }
        )
    )

    entries = harvest_dispatch(tmp_path)

    assert len(entries) == 1
    assert entries[0].duration_s_max == 100.0
    assert entries[0].reward_final_ema_mean == 10.0


def test_renderer_and_presets_are_not_averaged_together(tmp_path: Path) -> None:
    # A depth rollout is not the same workload as an RGB one.
    for name, renderer, presets in (("d", "ovrtx", ["depth"]), ("r", "ovrtx", ["rgb"])):
        row = tmp_path / name
        row.mkdir()
        (row / f"benchmark_training_x_2026-07-29_12-00-0{name}.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.2",
                    "run": {
                        "task": "Isaac-Cam",
                        "framework": "rsl_rl",
                        "seed": 42,
                        "status": "completed",
                        "duration_s": 10.0,
                        "num_envs": 8,
                        "max_iterations": 5,
                        "config": {
                            "physics_backend": "newton_mjwarp",
                            "rendering_backend": renderer,
                            "presets": presets,
                        },
                    },
                    "runtime": {"total_wall_time_s": 10.0},
                }
            )
        )

    entries = harvest_dispatch(tmp_path)

    assert len(entries) == 2
    assert {e.presets for e in entries} == {"depth", "rgb"}
