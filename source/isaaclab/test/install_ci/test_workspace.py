# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The committed uv workspace installs and launches its supported runtimes."""

import platform
from collections.abc import Callable
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


def test_workspace_install(
    workspace: tuple[str, Path, dict[str, str]], checkout: Path, run: Callable[..., str]
) -> None:
    runtime, _, env = workspace
    probe = "import isaaclab, isaaclab_tasks; from isaaclab.sim import SimulationContext"
    if runtime == "isaacsim":
        probe += "; from isaaclab_physx.app import KitLauncher"
    else:
        probe += (
            "; import importlib.metadata as m; "
            "assert not any(d.metadata['Name'] == 'isaacsim' for d in m.distributions())"
        )
    run("uv", "run", "--no-sync", "python", "-c", probe, cwd=checkout, env=env)
    output = run("uv", "run", "--no-sync", "isaaclab", "--help", cwd=checkout, env=env)
    assert "train" in output


@pytest.mark.gpu
def test_workspace_trains(workspace: tuple[str, Path, dict[str, str]], checkout: Path, run: Callable[..., str]) -> None:
    runtime, directory, env = workspace
    if runtime == "isaacsim":
        if platform.system() == "Linux":
            shim = directory / "venv/lib/python3.12/site-packages/isaacsim/kit/kernel/plugins/libcarb.env.shim.so"
            assert shim.is_file()
            env = {**env, "LD_PRELOAD": str(shim)}
    physics = "isaacsim_physx" if runtime == "isaacsim" else "newton_mjwarp"
    output = run(
        "uv",
        "run",
        "--no-sync",
        "isaaclab",
        "train",
        "--rl_library",
        "rsl_rl",
        "--task",
        "Isaac-Cartpole-Direct",
        "--num_envs",
        "16",
        "--max_iterations",
        "5",
        f"physics={physics}",
        cwd=checkout,
        env=env,
        timeout=900,
    )
    assert "Training time:" in output
    assert "Traceback (most recent call last):" not in output
