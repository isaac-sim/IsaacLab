# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installed wheel resources and CLI work without importing from the checkout."""

from collections.abc import Callable
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


def test_installed_wheel(installed_wheel: tuple[str, Path, Path], wheel: Path, run: Callable[..., str]) -> None:
    runtime, directory, python = installed_wheel
    output = run(str(python), "-c", "import isaaclab; print(isaaclab.__version__)", cwd=directory)
    assert output.strip() == wheel.name.split("-")[1]
    run(
        str(python),
        "-c",
        """
import importlib.util
from pathlib import Path
import isaaclab
from isaaclab import _deprioritize_prebundle_paths
from isaaclab.app import launch_simulation
from isaaclab.envs import VideoRecorderCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG
from isaaclab.cli.utils import ISAACLAB_ROOT
from isaaclab.programs import DEMOS, EXAMPLES
import isaaclab_rl
import isaaclab_tasks

assert list(isaaclab.__path__) == [str(Path(isaaclab.__file__).parent)]
assert all(program.path.is_file() for program in (*DEMOS, *EXAMPLES))
assert (ISAACLAB_ROOT / 'tools/template/cli.py').is_file()
assert all((ISAACLAB_ROOT / script).is_file() for script in (
    'scripts/reinforcement_learning/train.py',
    'scripts/reinforcement_learning/train_multigpu.py',
    'scripts/reinforcement_learning/play.py',
    'scripts/environments/teleoperation/teleop_se3_agent.py',
    'scripts/tools/record_demos.py',
    'scripts/tools/replay_demos.py',
))
assert (ISAACLAB_ROOT / 'apps/isaaclab.python.kit').is_file()
assert (Path(isaaclab.__file__).parent / 'examples/assets/nvidia_logo_domino_poses.pth').is_file()
assert importlib.util.find_spec('pytetwild') is None
""",
        cwd=directory,
    )
    if runtime == "isaacsim":
        run(
            str(python),
            "-c",
            "from isaaclab_physx.app import KitLauncher; from isaaclab.sim import SimulationContext",
            cwd=directory,
        )
    else:
        run(
            str(python),
            "-c",
            "import importlib.metadata as m; "
            "assert not any(d.metadata['Name'] == 'isaacsim' for d in m.distributions())",
            cwd=directory,
        )
    output = run(str(python), "-m", "isaaclab", "--help", cwd=directory)
    assert "train" in output


@pytest.mark.gpu
@pytest.mark.parametrize("installed_wheel", ["base"], indirect=True)
def test_wheel_trains(installed_wheel: tuple[str, Path, Path], run: Callable[..., str]) -> None:
    runtime, directory, python = installed_wheel
    output = run(
        str(python),
        "-m",
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
        "physics=newton_mjwarp",
        cwd=directory,
        timeout=900,
    )
    assert "Training time:" in output
    assert "Traceback (most recent call last):" not in output
