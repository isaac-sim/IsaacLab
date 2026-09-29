# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import subprocess
import sys
import textwrap

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from isaaclab_tasks.utils.hydra import resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_TASK = "Isaac-Pendulum-MARL-Direct"


def test_pendulum_marl_package_exports_are_discoverable_without_eager_runtime_imports():
    """Public task classes should be discoverable without importing the environment implementation."""
    script = textwrap.dedent(
        """\
        import sys

        import isaaclab_tasks.core.pendulum as pendulum

        assert "PendulumMARLEnv" in dir(pendulum)
        assert "PendulumMARLEnvCfg" in dir(pendulum)
        assert "isaaclab_tasks.core.pendulum.pendulum_marl_env" not in sys.modules
        """
    )

    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)

    assert result.returncode == 0, result.stdout + result.stderr


def test_pendulum_marl_defaults_to_mjwarp_newton_physics():
    """The unqualified task should resolve to the validated MJWarp Newton preset."""
    cfg = resolve_presets(load_cfg_from_registry(_TASK, "env_cfg_entry_point"))

    assert isinstance(cfg.sim.physics, NewtonCfg)
    assert isinstance(cfg.sim.physics.solver_cfg, MJWarpSolverCfg)
