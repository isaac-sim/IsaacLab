# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cross-backend LEAPP export integration tests."""

import subprocess
import sys
from pathlib import Path

import pytest

_EXPORT_TEST_DIR = Path(__file__).resolve().parent
if str(_EXPORT_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPORT_TEST_DIR))

from leapp_export_flow_helpers import (  # noqa: E402
    REPO_ROOT,
    ExportFlowBackend,
    export_flow_batch_main,
    fail_on_process_error,
    run_export_flow_subprocess,
)

_THIS_SCRIPT = Path(__file__).resolve()
_LEAPP_ROOT = REPO_ROOT / "scripts" / "reinforcement_learning" / "leapp"

_COMMON_TASKS = ("Isaac-Cartpole", "Isaac-Reach-Franka", "Isaac-Lift-Cube-Franka")
_SKIP_PRETRAINED = (
    ("Unfortunately a pre-trained checkpoint is currently unavailable", "No pretrained checkpoint available"),
)

_EXPORT_BACKENDS = (
    ExportFlowBackend(
        id="rsl_rl",
        rl_library="rsl_rl",
        export_script=_LEAPP_ROOT / "rsl_rl" / "export.py",
        module_name="_isaaclab_rsl_rl_leapp_export",
        export_fn_name="export_rsl_rl_agent",
        cache_marker="ensure_actor_hidden_state_initialized",
        tasks=(
            "Isaac-Cartpole",
            "Isaac-Ant",
            "IsaacContrib-Navigation-Flat-AnymalC",
            "Isaac-Velocity-Rough-AnymalD",
            "Isaac-Reach-Franka",
            "Isaac-Lift-Cube-Franka",
            "Isaac-Open-Drawer-Franka",
            "Isaac-Reorient-KukaAllegro",
        ),
        include_headless_arg=False,
        strip_play_suffix=False,
        allow_missing_export=True,
    ),
    ExportFlowBackend(
        id="rl_games",
        rl_library="rl_games",
        export_script=_LEAPP_ROOT / "rl_games" / "export.py",
        module_name="_isaaclab_rl_games_leapp_export",
        export_fn_name="export_rl_games_agent",
        cache_marker="export_rl_games_agent",
        tasks=_COMMON_TASKS,
        skip_output_patterns=_SKIP_PRETRAINED,
    ),
    ExportFlowBackend(
        id="skrl",
        rl_library="skrl",
        export_script=_LEAPP_ROOT / "skrl" / "export.py",
        module_name="_isaaclab_skrl_leapp_export",
        export_fn_name="export_skrl_agent",
        cache_marker="export_skrl_agent",
        tasks=_COMMON_TASKS,
        use_skrl_agent_entry_point=True,
        skip_output_patterns=_SKIP_PRETRAINED,
    ),
    ExportFlowBackend(
        id="sb3",
        rl_library="sb3",
        export_script=_LEAPP_ROOT / "sb3" / "export.py",
        module_name="_isaaclab_sb3_leapp_export",
        export_fn_name="export_sb3_agent",
        cache_marker="export_sb3_agent",
        tasks=("Isaac-Cartpole",),
        stub_isaaclab_on_load=False,
        subprocess_env={"ACCEPT_EULA": "Y", "OMNI_KIT_ACCEPT_EULA": "Y"},
        skip_output_patterns=_SKIP_PRETRAINED,
    ),
)


def test_export_flow_fails_on_sim_traceback():
    """Catch simulator failures even when the process reports success."""
    result = subprocess.CompletedProcess(
        args=["export-flow"],
        returncode=0,
        stdout="Traceback (most recent call last):\nFileNotFoundError: missing asset",
        stderr="",
    )

    with pytest.raises(pytest.fail.Exception):
        fail_on_process_error(result, ["Isaac-Reach-Franka"])


@pytest.mark.parametrize("backend", _EXPORT_BACKENDS, ids=lambda backend: backend.id)
def test_leapp_export_flow(backend: ExportFlowBackend):
    """Run each backend export script and assert LEAPP artifacts are created."""
    run_export_flow_subprocess(_THIS_SCRIPT, backend)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--export-flow-batch":
    export_flow_batch_main(_EXPORT_BACKENDS, sys.argv)
