# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise the maintained PhysX soft and cloth scenes through the required clone path."""

from isaaclab.test.utils import launch_test_simulation

launch_test_simulation()

import pytest
import torch

from isaaclab.assets import DeformableObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import build_simulation_context
from isaaclab.utils import configclass

from isaaclab_tasks.core.lift.config.franka_soft.franka_cloth_env_cfg import FrankaClothEnvCfg
from isaaclab_tasks.utils import resolve_presets

pytestmark = [pytest.mark.integration, pytest.mark.isaacsim_ci]


def test_physx_cloth_scene_replicates_two_live_environments():
    cfg = resolve_presets(FrankaClothEnvCfg(), selected=("isaacsim_physx",))

    @configclass
    class DeformableSceneCfg(InteractiveSceneCfg):
        deformable: DeformableObjectCfg = cfg.scene.deformable

    with build_simulation_context(device="cuda:0", sim_cfg=cfg.sim) as sim:
        scene = InteractiveScene(DeformableSceneCfg(num_envs=2, env_spacing=2.0))
        sim.reset()
        deformable = scene["deformable"]
        assert deformable.is_initialized
        assert deformable.num_instances == 2
        positions = deformable.data.nodal_pos_w.torch
        assert positions.shape[0] == 2
        assert torch.isfinite(positions).all()
        assert not torch.allclose(positions[0], positions[1])
        sim.step()
        scene.update(sim.get_physics_dt())
        assert torch.isfinite(deformable.data.nodal_pos_w.torch).all()
