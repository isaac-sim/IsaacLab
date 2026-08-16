# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for preset resolution and Kit decision logic.

These tests verify that given presets (e.g. ``presets=newton_mjwarp,ovrtx``),
the config-based logic correctly decides whether Isaac Sim Kit is needed.
No Kit/GPU required — safe for CI and beginners.
"""

import sys

import gymnasium as gym
import pytest
from isaaclab_physx.physics import PhysxCfg

from isaaclab.physics import PhysicsCfg, PhysxAutoCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config
from isaaclab_tasks.utils.hydra import collect_presets, resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_tasks.utils.preset_cli import enumerate_task_presets
from isaaclab_tasks.utils.preset_target import PresetTarget

_CAMERA_PRESETS_TASK = "Isaac-Cartpole-Camera-Direct"


def _physics_cfg(value):
    return value if isinstance(value, PhysicsCfg) else getattr(value, "physics", None)


def _resolve_with_presets(presets: str):
    """Resolve env_cfg with given presets. Modifies sys.argv temporarily."""
    return _resolve_with_args(f"presets={presets}")


def _resolve_with_args(*args: str):
    """Resolve env_cfg with the given Hydra-style args. Modifies sys.argv temporarily."""
    old_argv = sys.argv.copy()
    try:
        sys.argv = [sys.argv[0], *args]
        env_cfg, _ = resolve_task_config(_CAMERA_PRESETS_TASK, "rl_games_cfg_entry_point")
        return env_cfg
    finally:
        sys.argv = old_argv


def test_resolve_task_config_applies_plain_scalar_override():
    """Plain ``env.*=value`` overrides should resolve without requiring Hydra composition."""
    old_argv = sys.argv.copy()
    try:
        sys.argv = [sys.argv[0], "env.scene.num_envs=123"]
        env_cfg, _ = resolve_task_config(_CAMERA_PRESETS_TASK, "rl_games_cfg_entry_point")
    finally:
        sys.argv = old_argv

    assert env_cfg.scene.num_envs == 123


def test_rtx_is_renderer_selector():
    """The automatic RTX selector is exposed as ``renderer=rtx``."""
    preset_map = enumerate_task_presets(_CAMERA_PRESETS_TASK)

    assert preset_map is not None
    assert "rtx" in preset_map[PresetTarget.RENDERER]


def test_isaacsim_physx_is_physics_selector():
    """The concrete Isaac Sim PhysX selector is exposed as ``physics=isaacsim_physx``."""
    preset_map = enumerate_task_presets(_CAMERA_PRESETS_TASK)

    assert preset_map is not None
    assert "isaacsim_physx" in preset_map[PresetTarget.PHYSICS]


@pytest.mark.parametrize(
    ("task_id", "presets", "camera_paths"),
    [
        ("Isaac-Cartpole-Camera-Direct", (), ("tiled_camera",)),
        ("Isaac-Cartpole-Camera", (), ("scene.tiled_camera",)),
        ("Isaac-Reorient-KukaAllegro-Camera", (), ("scene.base_camera",)),
        ("Isaac-Reorient-KukaAllegro-Camera", ("duo_camera",), ("scene.base_camera", "scene.wrist_camera")),
        ("Isaac-Lift-KukaAllegro-Camera", (), ("scene.base_camera",)),
        ("Isaac-Lift-KukaAllegro-Camera", ("duo_camera",), ("scene.base_camera", "scene.wrist_camera")),
        ("Isaac-Reorient-Cube-Shadow-Camera-Direct", (), ("tiled_camera",)),
        ("Isaac-Reorient-Cube-Shadow-Camera", (), ("scene.tiled_camera",)),
        ("Isaac-Lift-Soft-Franka-Camera", (), ("scene.base_camera",)),
        ("Isaac-Lift-Cable-Franka-Camera", (), ("scene.base_camera",)),
        ("Isaac-Lift-Cloth-Franka-Camera", (), ("scene.base_camera",)),
    ],
)
def test_core_camera_tasks_default_to_newton_renderer(
    task_id: str, presets: tuple[str, ...], camera_paths: tuple[str, ...]
):
    """Core camera tasks should resolve their default renderer to Newton."""
    from isaaclab_newton.renderers import NewtonWarpRendererCfg

    env_cfg = resolve_presets(load_cfg_from_registry(task_id, "env_cfg_entry_point"), selected=presets)
    for camera_path in camera_paths:
        camera_cfg = env_cfg
        for attr in camera_path.split("."):
            camera_cfg = getattr(camera_cfg, attr)
        assert isinstance(camera_cfg.renderer_cfg, NewtonWarpRendererCfg), f"{task_id}:{camera_path}"


def test_registered_task_physx_presets_keep_auto_selection_explicit():
    """PhysX defaults are concrete while ``physx`` remains the automatic selector."""

    for task_id, task_spec in gym.registry.items():
        if not task_id.startswith(("Isaac-", "IsaacContrib-")) or "env_cfg_entry_point" not in task_spec.kwargs:
            continue
        env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
        presets = collect_presets(env_cfg)
        has_auto_physx = any(isinstance(_physics_cfg(fields.get("physx")), PhysxAutoCfg) for fields in presets.values())
        for path, fields in presets.items():
            location = f"{task_id}:{path}"
            physics_fields = {name: _physics_cfg(value) for name, value in fields.items()}
            if any(isinstance(value, PhysxCfg) for value in physics_fields.values()):
                auto_cfg = physics_fields.get("physx")
                isaacsim_cfg = physics_fields.get("isaacsim_physx")
                default_cfg = physics_fields.get("default")
                assert isinstance(auto_cfg, PhysxAutoCfg), location
                assert isinstance(isaacsim_cfg, PhysxCfg), location
                assert not isinstance(default_cfg, PhysxAutoCfg), location
                if isinstance(default_cfg, PhysxCfg):
                    assert default_cfg == isaacsim_cfg, location
                assert auto_cfg.isaacsim_physx == isaacsim_cfg, location
                assert auto_cfg.ovphysx == physics_fields.get("ovphysx"), location
            elif has_auto_physx and "physx" in fields:
                assert fields.get("isaacsim_physx") == fields["physx"], location


def test_physx_and_isaacsim_physx_presets_conflict():
    """``physx`` and ``isaacsim_physx`` are distinct choices even when both use PhysxCfg."""
    with pytest.raises(ValueError, match="Conflicting global presets"):
        _resolve_with_presets("physx,isaacsim_physx")
