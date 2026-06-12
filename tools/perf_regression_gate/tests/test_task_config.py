# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the migrated tasks.json schema + relative floor."""

import pytest
from task_config import get_task, load_tasks


def test_per_backend_excluded_frames_differ():
    # PhysX is steady fast ([[0,1]]); Newton needs more JIT warm-up ([[0,4]]).
    physx = get_task("Isaac-Cartpole", "physx")
    newton = get_task("Isaac-Cartpole", "newton")
    assert sorted(physx.excluded_frames) == [0, 1]
    assert sorted(newton.excluded_frames) == [0, 1, 2, 3, 4]


def test_relative_floor_is_fraction_of_ref():
    t = get_task("Isaac-Cartpole", "physx")
    assert t.fps_floor_pct == 40.0
    assert t.fps_floor("NVIDIA L40S") == pytest.approx(0.40 * 276401.7, rel=1e-6)


def test_floor_gpu_substring_match():
    t = get_task("Isaac-Cartpole", "physx")
    # "L40S" should still match the "NVIDIA L40S" ref key.
    assert t.fps_floor("L40S") == pytest.approx(0.40 * 276401.7, rel=1e-6)


def test_floor_disabled_when_gpu_unknown():
    t = get_task("Isaac-Cartpole", "physx")
    assert t.fps_floor("RTX-9999") == 0.0


def test_camera_task_flags():
    t = get_task("Isaac-Repose-Cube-Shadow-Vision-Direct-v0", "physx_isaacsim_rtx_renderer")
    assert t.enable_cameras is True
    assert t.camera_resolution == (64, 64)
    assert "camera" in t.tags
    assert sorted(t.excluded_frames)[:1] == [0] and max(t.excluded_frames) == 59


def test_get_task_unknown_raises():
    with pytest.raises(KeyError):
        get_task("Does-Not-Exist", "physx")


def test_all_tasks_load():
    tasks = load_tasks()
    keys = {(t.task_id, t.backend_key) for t in tasks}
    assert ("Isaac-Cartpole", "physx") in keys
    assert ("Isaac-Cartpole", "newton") in keys
    assert ("Isaac-Velocity-Flat-G1-v0", "newton") in keys
    assert ("Isaac-Factory-GearMesh-Direct-v0", "physx") in keys
