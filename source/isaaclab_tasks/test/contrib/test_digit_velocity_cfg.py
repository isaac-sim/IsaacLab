# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.preset_cli import enumerate_task_presets
from isaaclab_tasks.utils.preset_target import PresetTarget

_DIGIT_TASK_IDS = (
    "IsaacContrib-Velocity-Flat-Digit",
    "IsaacContrib-Velocity-Flat-Digit-Play",
    "IsaacContrib-Velocity-Rough-Digit",
    "IsaacContrib-Velocity-Rough-Digit-Play",
)

_OLD_DIGIT_TASK_IDS = (
    "Isaac-Velocity-Flat-Digit",
    "Isaac-Velocity-Flat-Digit-Play",
    "Isaac-Velocity-Rough-Digit",
    "Isaac-Velocity-Rough-Digit-Play",
)

_DIGIT_PHYSX_ONLY_TASK_IDS = _DIGIT_TASK_IDS + (
    "IsaacContrib-Tracking-LocoManip-Digit",
    "IsaacContrib-Tracking-LocoManip-Digit-Play",
)


def test_digit_velocity_tasks_are_registered_under_contrib() -> None:
    assert set(_DIGIT_TASK_IDS) <= set(gym.registry)
    assert set(_OLD_DIGIT_TASK_IDS).isdisjoint(gym.registry)

    for task_id in _DIGIT_TASK_IDS:
        spec = gym.spec(task_id)
        assert spec.kwargs["env_cfg_entry_point"].startswith("isaaclab_tasks.contrib.velocity.config.digit.")
        assert spec.kwargs["rsl_rl_cfg_entry_point"].startswith("isaaclab_tasks.contrib.velocity.config.digit.agents.")


def test_digit_tasks_only_expose_physx() -> None:
    for task_id in _DIGIT_PHYSX_ONLY_TASK_IDS:
        presets = enumerate_task_presets(task_id)
        assert presets is not None
        assert presets[PresetTarget.PHYSICS] == ["physx"]
