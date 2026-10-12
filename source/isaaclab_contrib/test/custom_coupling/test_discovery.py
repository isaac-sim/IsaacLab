# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test opt-in custom coupling discovery."""

import subprocess
import sys
import textwrap


def test_custom_coupling_is_opt_in() -> None:
    """Only importing the registration module may register the example task."""
    script = textwrap.dedent(
        """
        import gymnasium as gym
        import isaaclab_tasks

        custom_id = "IsaacContrib-Lift-Soft-Franka-Custom-Coupling"
        assert "Isaac-Lift-Soft-Franka" in gym.registry
        assert custom_id not in gym.registry

        # Importing the library modules must not register the task.
        import isaaclab_contrib.custom_coupling
        import isaaclab_contrib.custom_coupling.coupled_mjwarp_vbd_manager

        assert custom_id not in gym.registry

        import isaaclab_contrib.custom_coupling.tasks

        assert custom_id in gym.registry
        spec = gym.spec(custom_id)
        assert spec.kwargs["env_cfg_entry_point"] == (
            "isaaclab_contrib.custom_coupling.franka_soft_env_cfg:FrankaSoftCustomCouplingEnvCfg"
        )
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
