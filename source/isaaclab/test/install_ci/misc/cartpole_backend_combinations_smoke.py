# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera Cartpole training on every physics backend with every renderer.

The caller owns environment preparation and needs the ``isaacsim`` and ``ov`` extras. Each pairing
trains in its own process. Isaac Sim / Kit cannot share a process with OvPhysX or OVRTX
(:func:`isaaclab.app.sim_launcher.validate_launch`), so those pairings are left out.
"""

from __future__ import annotations

import itertools
import subprocess
import sys
from pathlib import Path

import pytest

_PHYSICS = ("newton_mjwarp", "newton_kamino", "isaacsim_physx", "ovphysx")
_RENDERERS = ("newton_renderer", "ovrtx", "isaacsim_rtx")
_KIT = {"isaacsim_physx", "isaacsim_rtx"}
_KITLESS_ONLY = {"ovphysx", "ovrtx"}

_COMBINATIONS = [
    pytest.param(physics, renderer, id=f"{physics}-{renderer}")
    for physics, renderer in itertools.product(_PHYSICS, _RENDERERS)
    if not ({physics, renderer} & _KIT and {physics, renderer} & _KITLESS_ONLY)
]


@pytest.mark.parametrize(("physics", "renderer"), _COMBINATIONS)
def test_train_cartpole_camera_completes(physics: str, renderer: str) -> None:
    """Verify that camera Cartpole completes short RSL-RL training on one backend pairing."""
    isaaclab_root = next(parent for parent in Path(__file__).resolve().parents if (parent / "uv.lock").is_file())
    command = [
        *(sys.executable, "-m", "isaaclab", "train", "--rl_library", "rsl_rl"),
        *("--task", "Isaac-Cartpole-Camera-Direct", "--num_envs", "16"),
        *(f"presets={physics},{renderer}", "--max_iterations", "2"),
    ]
    # A cold run compiles kernels and shaders before training begins.
    result = subprocess.run(command, cwd=isaaclab_root, text=True, capture_output=True, timeout=1800, check=False)
    output = result.stdout + (result.stderr or "")
    assert result.returncode == 0, f"Training failed (rc={result.returncode}):\n{output}"
    assert "Traceback (most recent call last):" not in output, f"Training produced a traceback:\n{output}"
    assert "Training time:" in output, f"Training did not report completion:\n{output}"
