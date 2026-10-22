# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVPhysX ray-caster bindings and multi-mesh tracking."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

from isaaclab_ov.physics import OvPhysxCfg

from pxr import UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.cloner import CloneCfg, InclusionSet
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg, patterns
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils import configclass, instantiate

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_multi_mesh_tracks_multiple_body_types(device):
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
        right = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Right",
            spawn=sim_utils.MultiUsdFileCfg(
                usd_path=[str(Path(__file__).parents[1] / "assets/data/articulation_ordering_branching.usda")] * 2
            ),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.75, 0.0, 0.0)),
            actuators={},
        )
        scanner = MultiMeshRayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Right/left_tip/Mount",
            offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(-0.85, -0.2, 1.2)),
            mesh_prim_paths=[
                MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr=f"{{ENV_REGEX_NS}}/{body}")
                for body in ("Left", "Right/[^/]*/visuals")
            ],
            pattern_cfg=patterns.GridPatternCfg(resolution=1.5, size=(1.5, 0.0)),
            ray_alignment="world",
        )

    cfg = sim_utils.SimulationCfg(physics=OvPhysxCfg(), device=device, gravity=(0.0, 0.0, 0.0))
    with sim_utils.build_simulation_context(sim_cfg=cfg) as sim:
        scene = instantiate(
            SceneCfg(
                num_envs=16,
                env_spacing=4.0,
                clone_cfg=CloneCfg(clone_strategy=lambda weights, count: np.arange(count) % 2),
            )
        )
        for env_id in (0, 1):
            link_path = f"/World/envs/env_{env_id}/Right/left_tip"
            UsdGeom.Xformable(sim.stage.GetPrimAtPath(link_path)).AddTranslateOp().Set((0.0, 0.0, 0.5))
            UsdPhysics.Joint.Get(sim.stage, f"/World/envs/env_{env_id}/Right/left_elbow").CreateLocalPos0Attr(
                (0.0, 0.0, 0.5)
            )
            UsdGeom.Cube.Define(sim.stage, link_path + "/visuals").CreateSizeAttr(0.5)
            sim_utils.create_prim(link_path + "/Mount", translation=(0.1, 0.2, 0.3), orientation=(0.0, 0.0, 1.0, 0.0))
        sim.reset()
        sensor = scene["scanner"]
        assert not sim.stage.GetPrimAtPath("/World/envs/env_2/Right/left_tip/Mount")
        torch.testing.assert_close(
            sensor.data.pos_w.torch, scene.env_origins + torch.tensor([0.85, 0.2, 0.8], device=sim.device)
        )
        torch.testing.assert_close(
            sensor.data.quat_w.torch, torch.tensor([[0.0, 0.0, 1.0, 0.0]], device=sim.device).expand(16, -1)
        )
        expected = torch.tensor([[[-0.75, 0.0, 0.25], [0.75, 0.0, 0.75]]], device=sim.device)
        expected = expected + scene.env_origins[:, None, :]
        torch.testing.assert_close(sensor.data.ray_hits_w.torch, expected)

        pose = scene["right"].data.root_link_pose_w.torch.clone()
        heights = torch.linspace(0.25, 0.5, scene.num_envs, device=sim.device)
        pose[:, 2] += heights
        scene["right"].write_root_link_pose_to_sim_index(root_pose=pose)
        sensor.reset()
        expected[:, 1, 2] += heights
        torch.testing.assert_close(sensor.data.ray_hits_w.torch, expected)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_ray_caster_rejects_partial_native_mounts(device):
    """A native mount missing from some cloned worlds raises a sensor-level count error."""
    scene_cfg = InteractiveSceneCfg(
        num_envs=4,
        env_spacing=2.0,
        clone_cfg=CloneCfg(clone_combinations=[InclusionSet(assets=["mount"]), InclusionSet(assets=[])]),
    )
    scene_cfg.mount = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Mount",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=1.0),
        ),
    )
    scene_cfg.scanner = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Mount",
        mesh_prim_paths=[MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Mount")],
        pattern_cfg=patterns.GridPatternCfg(resolution=1.0, size=(0.0, 0.0)),
    )
    cfg = sim_utils.SimulationCfg(physics=OvPhysxCfg(), device=device)
    with sim_utils.build_simulation_context(sim_cfg=cfg) as sim:
        scene = instantiate(scene_cfg)
        with pytest.raises(RuntimeError, match="found 2 instances in 4 environments"):
            sim.reset()
        assert not scene["scanner"].is_initialized
