# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for preset resolution and Kit decision logic.

These tests verify that given presets (e.g. ``presets=newton,ovrtx_renderer``),
the config-based logic correctly decides whether Isaac Sim Kit is needed.
No Kit/GPU required — safe for CI and beginners.
"""

import sys

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import compute_kit_requirements, resolve_task_config

_CAMERA_PRESETS_TASK = "Isaac-Cartpole-Camera-Presets-Direct-v0"


def _resolve_with_presets(presets: str):
    """Resolve env_cfg with given presets. Modifies sys.argv temporarily."""
    old_argv = sys.argv.copy()
    try:
        sys.argv = [sys.argv[0], f"presets={presets}"]
        env_cfg, agent_cfg = resolve_task_config(_CAMERA_PRESETS_TASK, "rl_games_cfg_entry_point")
        return env_cfg
    finally:
        sys.argv = old_argv


def test_preset_newton_ovrtx_does_not_need_kit():
    """Newton + OVRTX renderer is kitless — no AppLauncher required."""
    env_cfg = _resolve_with_presets("newton,ovrtx_renderer")
    needs_kit, _, _ = compute_kit_requirements(env_cfg)
    assert needs_kit is False


def test_preset_newton_newton_renderer_does_not_need_kit():
    """Newton + Newton Warp renderer is kitless."""
    env_cfg = _resolve_with_presets("newton,newton_renderer")
    needs_kit, _, _ = compute_kit_requirements(env_cfg)
    assert needs_kit is False


def test_preset_physx_needs_kit():
    """PhysX physics requires Kit."""
    env_cfg = _resolve_with_presets("physx")
    needs_kit, _, _ = compute_kit_requirements(env_cfg)
    assert needs_kit is True


def test_preset_default_needs_kit():
    """Default (PhysX + Isaac RTX) requires Kit."""
    env_cfg = _resolve_with_presets("default")
    needs_kit, _, _ = compute_kit_requirements(env_cfg)
    assert needs_kit is True


def test_preset_newton_isaac_rtx_needs_kit():
    """Newton + Isaac RTX renderer requires Kit (RTX runs in Kit)."""
    env_cfg = _resolve_with_presets("newton,isaacsim_rtx_renderer")
    needs_kit, _, _ = compute_kit_requirements(env_cfg)
    assert needs_kit is True
