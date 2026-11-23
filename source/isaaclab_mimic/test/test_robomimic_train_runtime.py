# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Runtime selection of the robomimic training script."""

import runpy
import sys
from pathlib import Path
from types import ModuleType

import gymnasium as gym
import pytest

pytestmark = pytest.mark.unit

# Repository root: this file lives at source/isaaclab_mimic/test/<file>.py
ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("isaac_sim_installed", [True, False])
def test_robomimic_train_launches_isaac_sim_only_when_it_is_installed(monkeypatch, isaac_sim_installed):
    """An installed Isaac Sim is launched as upstream does; without one, training runs without a runtime."""
    from isaaclab_isaacsim.app import KitLauncher

    import isaaclab.app

    class RuntimeLaunched(Exception):
        pass

    def launch_simulation(cfg):
        assert cfg is None
        raise RuntimeLaunched

    # robomimic is an optional dependency; the script only needs its names to reach the launch.
    for name, attributes in {
        "robomimic": (),
        "robomimic.utils": (),
        "robomimic.utils.env_utils": (),
        "robomimic.utils.file_utils": (),
        "robomimic.utils.obs_utils": (),
        "robomimic.utils.torch_utils": (),
        "robomimic.utils.train_utils": (),
        "robomimic.utils.log_utils": ("DataLogger", "PrintLogger"),
        "robomimic.algo": ("algo_factory",),
        "robomimic.config": ("Config", "config_factory"),
    }.items():
        module = ModuleType(name)
        for attribute in attributes:
            setattr(module, attribute, object)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(isaaclab.app, "launch_simulation", launch_simulation)
    monkeypatch.setattr(KitLauncher, "is_available", classmethod(lambda cls: isaac_sim_installed))
    monkeypatch.setattr(sys, "argv", ["train.py", "--task", "Not-A-Registered-Task-v0", "--algo", "bc"])
    # Without a runtime the script goes straight to training, which stops at the unregistered task.
    expected = RuntimeLaunched if isaac_sim_installed else gym.error.Error
    with pytest.raises(expected):
        runpy.run_path(str(ROOT / "scripts/imitation_learning/robomimic/train.py"), run_name="__main__")
