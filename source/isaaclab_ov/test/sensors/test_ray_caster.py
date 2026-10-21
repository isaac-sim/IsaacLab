# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVPhysX ray-caster bindings and multi-mesh tracking."""

from __future__ import annotations

import torch
from isaaclab_ov.physics import OvPhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg, patterns
from isaaclab.utils import configclass, instantiate


def test_multi_mesh_tracks_multiple_body_types():
    """Cloned sensor offsets and distinct targets follow native bodies without destination USD prims."""

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        left = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Left",
            spawn=sim_utils.CuboidCfg(
                size=(0.5, 0.5, 0.5),
                rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
                mass_props=sim_utils.MassCfg(mass=1.0),
                collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.75, 0.0, 0.0)),
        )
        right = left.replace(
            prim_path="{ENV_REGEX_NS}/Right", init_state=RigidObjectCfg.InitialStateCfg(pos=(0.75, 0.0, 0.0))
        )
        scanner = MultiMeshRayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Left/Mount",
            offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.65, -0.2, 1.7)),
            mesh_prim_paths=[
                MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr=f"{{ENV_REGEX_NS}}/{body}")
                for body in ("Left", "Right")
            ],
            pattern_cfg=patterns.GridPatternCfg(resolution=1.5, size=(1.5, 0.0)),
            ray_alignment="world",
        )

    cfg = sim_utils.SimulationCfg(physics=OvPhysxCfg(), gravity=(0.0, 0.0, 0.0))
    with sim_utils.build_simulation_context(sim_cfg=cfg) as sim:
        scene = instantiate(SceneCfg(num_envs=2, env_spacing=4.0))
        sim_utils.create_prim(
            "/World/envs/env_0/Left/Mount", translation=(0.1, 0.2, 0.3), orientation=(0.0, 0.0, 1.0, 0.0)
        )
        sim.reset()
        sensor = scene["scanner"]
        assert not sim.stage.GetPrimAtPath("/World/envs/env_1/Left/Mount")
        torch.testing.assert_close(
            sensor.data.pos_w.torch, scene.env_origins + torch.tensor([-0.65, 0.2, 0.3], device=sim.device)
        )
        torch.testing.assert_close(
            sensor.data.quat_w.torch, torch.tensor([[0.0, 0.0, 1.0, 0.0]], device=sim.device).expand(2, -1)
        )
        expected = torch.tensor([[[-0.75, 0.0, 0.25], [0.75, 0.0, 0.25]]], device=sim.device)
        expected = expected + scene.env_origins[:, None, :]
        torch.testing.assert_close(sensor.data.ray_hits_w.torch, expected)

        pose = scene["right"].data.root_link_pose_w.torch.clone()
        pose[:, 2] += torch.tensor([0.25, 0.5], device=sim.device)
        scene["right"].write_root_link_pose_to_sim_index(root_pose=pose)
        sensor.reset()
        expected[:, 1, 2] += torch.tensor([0.25, 0.5], device=sim.device)
        torch.testing.assert_close(sensor.data.ray_hits_w.torch, expected)
