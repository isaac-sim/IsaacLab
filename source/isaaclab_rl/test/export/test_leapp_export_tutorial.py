# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One scene-backed check for the direct RL deployment tutorial."""

import subprocess
import sys
from pathlib import Path

import pytest
from leapp_direct_env import DIRECT_TASK
from leapp_initialized_checkpoints import _CheckpointEnv, _create_rsl_rl_checkpoint

from isaaclab.test.utils import resolve_test_sim_device

from isaaclab_tasks.utils import load_cfg_from_registry

_REPO_ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.integration
def test_direct_tutorial_scene_export(tmp_path: Path):
    """Launch the tutorial scene and export an initialized native RSL-RL policy."""
    agent_cfg = load_cfg_from_registry(DIRECT_TASK, "rsl_rl_cfg_entry_point")
    checkpoint = _create_rsl_rl_checkpoint(_CheckpointEnv(DIRECT_TASK, "rsl_rl"), agent_cfg, tmp_path)
    command = [
        sys.executable,
        str(Path(__file__).with_name("leapp_direct_env.py")),
        "--device",
        resolve_test_sim_device(),
        "--task",
        DIRECT_TASK,
        "--checkpoint",
        str(checkpoint),
        "--export_save_path",
        str(tmp_path),
        "--disable_graph_visualization",
    ]
    result = subprocess.run(command, cwd=_REPO_ROOT, capture_output=True, text=True, timeout=600)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output[-5000:]
    artifact_dir = tmp_path / DIRECT_TASK
    assert (artifact_dir / f"{DIRECT_TASK}.yaml").is_file(), output[-5000:]
    assert (artifact_dir / f"{DIRECT_TASK}.onnx").is_file(), output[-5000:]
