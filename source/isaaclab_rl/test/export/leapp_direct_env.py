# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LEAPP export entrypoint for the direct-environment tutorial."""

import importlib.util
import runpy
import sys
from pathlib import Path

import gymnasium as gym

_REPO_ROOT = Path(__file__).resolve().parents[4]
DIRECT_TASK = "IsaacContrib-Velocity-Flat-AnymalC-Direct"


def _make_tutorial_env(**kwargs):
    path = _REPO_ROOT / "scripts/tutorials/06_deploy/anymal_c_env.py"
    spec = importlib.util.spec_from_file_location("isaaclab_tasks.contrib.anymal_c_direct._leapp_tutorial", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load tutorial environment from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AnymalCEnv(**kwargs)


def _main() -> None:
    import isaaclab_tasks.contrib.anymal_c_direct  # noqa: F401

    original = gym.spec(DIRECT_TASK)
    gym.registry.pop(DIRECT_TASK)
    gym.register(
        id=DIRECT_TASK,
        entry_point=_make_tutorial_env,
        disable_env_checker=original.disable_env_checker,
        kwargs=dict(original.kwargs),
    )
    script = _REPO_ROOT / "scripts/reinforcement_learning/leapp/rsl_rl/export.py"
    sys.argv[0] = str(script)
    sys.path.insert(0, str(script.parent))
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    _main()
