# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Agibot place environment backend compatibility."""

import pytest
from isaaclab_newton.physics import NewtonCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

from isaaclab_tasks.contrib.place.config.agibot.place_toy2box_rmp_rel_env_cfg import RmpFlowAgibotPlaceToy2BoxEnvCfg


@pytest.mark.parametrize(
    "physics_cfg",
    (PhysxCfg(), OvPhysxCfg()),
    ids=("isaacsim_physx", "ovphysx"),
)
def test_agibot_accepts_newton_visualizer_with_physx_backends(physics_cfg) -> None:
    """NewtonGL visualization should remain available with either supported PhysX backend."""
    env_cfg = RmpFlowAgibotPlaceToy2BoxEnvCfg()
    env_cfg.sim.physics = physics_cfg
    env_cfg.sim.visualizer_cfgs = NewtonGLVisualizerCfg(headless=True)

    env_cfg.validate_config()


def test_agibot_rejects_newton_physics_for_incompatible_collision_mesh() -> None:
    """Newton physics should report the remaining asset collision-mesh limitation."""
    env_cfg = RmpFlowAgibotPlaceToy2BoxEnvCfg()
    env_cfg.sim.physics = NewtonCfg()
    env_cfg.sim.visualizer_cfgs = NewtonGLVisualizerCfg(headless=True)

    with pytest.raises(ValueError, match=r"generated convex collision mesh.*physics=isaacsim_physx"):
        env_cfg.validate_config()
