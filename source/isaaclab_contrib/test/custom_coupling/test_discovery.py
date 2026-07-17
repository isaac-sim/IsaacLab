# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for opt-in custom coupling discovery."""

import subprocess
import sys
import textwrap


def test_custom_coupling_is_opt_in() -> None:
    """Verify core and custom task registration in a fresh process."""
    script = textwrap.dedent(
        """
        import gymnasium as gym
        import warnings

        import isaaclab_contrib.deformable as deformable
        from isaaclab_contrib.coupling import CouplerProxyCfg

        with warnings.catch_warnings(record=True) as import_warnings:
            warnings.simplefilter("always")
            import isaaclab_tasks
            from isaaclab_tasks.core.lift.config.franka_soft.franka_cloth_env_cfg import FrankaClothEnvCfg
            from isaaclab_tasks.core.lift.config.franka_soft.franka_soft_env_cfg import FrankaSoftEnvCfg
        assert not any(
            "CoupledMJWarpVBDSolverCfg is deprecated" in str(item.message) for item in import_warnings
        )

        custom_id = "IsaacContrib-Lift-Soft-Franka-Custom-Coupling"
        assert custom_id not in gym.registry
        assert "Isaac-Lift-Soft-Franka" in gym.registry
        assert "Isaac-Lift-Cloth-Franka" in gym.registry

        assert hasattr(deformable, "CoupledMJWarpVBDSolverCfg")
        assert hasattr(deformable, "CoupledFeatherstoneVBDSolverCfg")
        assert not hasattr(deformable, "NewtonCoupledMJWarpVBDManager")
        assert not hasattr(deformable, "NewtonCoupledFeatherstoneVBDManager")

        from isaaclab_contrib.deformable.coupled_mjwarp_vbd_manager import NewtonCoupledMJWarpVBDManager
        from isaaclab_contrib.deformable.coupled_featherstone_vbd_manager import NewtonCoupledFeatherstoneVBDManager

        assert NewtonCoupledMJWarpVBDManager.__name__ == "NewtonCoupledMJWarpVBDManager"
        assert NewtonCoupledFeatherstoneVBDManager.__name__ == "NewtonCoupledFeatherstoneVBDManager"
        assert custom_id not in gym.registry

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            deformable.CoupledMJWarpVBDSolverCfg()
            deformable.CoupledFeatherstoneVBDSolverCfg()
        assert len(caught) == 2
        assert all(item.category is DeprecationWarning for item in caught)

        soft_cfg = FrankaSoftEnvCfg()
        cloth_cfg = FrankaClothEnvCfg()
        assert isinstance(soft_cfg.sim.physics.default.solver_cfg, CouplerProxyCfg)
        assert isinstance(cloth_cfg.sim.physics.default.solver_cfg, CouplerProxyCfg)

        import isaaclab_contrib.custom_coupling as custom_coupling
        from isaaclab_contrib.custom_coupling.franka_soft_env_cfg import FrankaSoftCustomCouplingEnvCfg

        assert custom_id in gym.registry
        spec = gym.spec(custom_id)
        assert spec.kwargs["env_cfg_entry_point"].startswith("isaaclab_contrib.custom_coupling.")

        assert not hasattr(custom_coupling, "FrankaSoftCustomCouplingEnvCfg")
        assert not hasattr(custom_coupling, "NewtonCoupledMJWarpVBDManager")
        custom_cfg = FrankaSoftCustomCouplingEnvCfg()
        assert isinstance(custom_cfg.sim.physics.default.solver_cfg, custom_coupling.CoupledMJWarpVBDSolverCfg)
        assert custom_cfg.sim.physics.default.solver_cfg.model_cfg == soft_cfg.sim.physics.default.solver_cfg.model_cfg
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
