# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checks USD-level scale randomization of rigid objects in a manager-based environment."""

from __future__ import annotations

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import pytest
import torch
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

from pxr import Sdf

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import ActionTerm, ActionTermCfg, SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

pytestmark = pytest.mark.integration


class CubeActionTerm(ActionTerm):
    """PD velocity controller that drives the cube towards the commanded position in the environment frame."""

    _asset: RigidObject

    def __init__(self, cfg: CubeActionTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(env.num_envs, 3, device=self.device)
        self._vel_command = torch.zeros(self.num_envs, 6, device=self.device)
        self.p_gain = cfg.p_gain
        self.d_gain = cfg.d_gain

    @property
    def action_dim(self) -> int:
        return self._raw_actions.shape[1]

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

    def apply_actions(self):
        pos_error = self._raw_actions - (self._asset.data.root_pos_w.torch - self._env.scene.env_origins)
        vel_error = -self._asset.data.root_lin_vel_w.torch
        self._vel_command[:, :3] = self.p_gain * pos_error + self.d_gain * vel_error
        self._asset.write_root_velocity_to_sim_index(root_velocity=self._vel_command)


@configclass
class CubeActionTermCfg(ActionTermCfg):
    class_type: type = CubeActionTerm
    p_gain: float = 5.0
    d_gain: float = 0.5


def base_position(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Root position of the asset in the environment frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w.torch - env.scene.env_origins


def _cube_cfg(name: str) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_[^/]+/{name}",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=PhysxRigidBodyCfg(max_depenetration_velocity=1.0, disable_gravity=True),
            mass_props=sim_utils.MassCfg(mass=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 5)),
    )


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Ground plane and two floating cubes: one with randomized scale and one with a fixed scale."""

    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", debug_vis=False)
    cube1: RigidObjectCfg = _cube_cfg("cube1")
    cube2: RigidObjectCfg = _cube_cfg("cube2")


@configclass
class ActionsCfg:
    joint_pos = CubeActionTermCfg(asset_name="cube1")


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        position = ObsTerm(func=base_position, params={"asset_cfg": SceneEntityCfg("cube1")})

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5)},
            "asset_cfg": SceneEntityCfg("cube1"),
        },
    )
    randomize_cube1_scale = EventTerm(
        func=mdp.randomize_rigid_body_scale,
        mode="prestartup",
        params={
            "scale_range": {"x": (0.5, 1.5), "y": (0.5, 1.5), "z": (0.5, 1.5)},
            "asset_cfg": SceneEntityCfg("cube1"),
        },
    )
    randomize_cube2_scale = EventTerm(
        func=mdp.randomize_rigid_body_scale,
        mode="prestartup",
        params={
            "scale_range": {"x": (1.0, 1.0), "y": (1.0, 1.0), "z": (1.0, 1.0)},
            "asset_cfg": SceneEntityCfg("cube2"),
        },
    )


@configclass
class CubeEnvCfg(ManagerBasedEnvCfg):
    # prestartup USD randomization requires individually parsed assets
    scene: MySceneCfg = MySceneCfg(num_envs=4, env_spacing=2.5, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 2
        self.sim.dt = 0.01
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.render_interval = self.decimation


def _read_scales(pattern: str) -> list[tuple[float, float, float]]:
    """Return the authored ``xformOp:scale`` of every prim matching ``pattern``."""
    root_layer = sim_utils.get_current_stage().GetRootLayer()
    scales = []
    for prim_path in sim_utils.find_matching_prim_paths(pattern):
        prim_spec = Sdf.CreatePrimInLayer(root_layer, prim_path)
        scales.append(tuple(prim_spec.GetAttributeAtPath(prim_path + ".xformOp:scale").default))
    return scales


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_scale_randomization(device):
    """Randomized cubes receive distinct scales, fixed cubes keep unit scale, and the environment still steps."""
    sim_utils.create_new_stage()
    env_cfg = CubeEnvCfg()
    env_cfg.sim.device = device
    env = ManagerBasedEnv(cfg=env_cfg)

    assert len(sim_utils.find_matching_prim_paths("/World/envs/env_[^/]+/cube[^/]*/[^/]*")) == env.num_envs * 2
    randomized_scales = _read_scales("/World/envs/env_[^/]+/cube1")
    assert len(set(randomized_scales)) == env.num_envs, "repeated scale values indicate randomization is not applied"
    assert _read_scales("/World/envs/env_[^/]+/cube2") == [(1.0, 1.0, 1.0)] * env.num_envs

    target_position = torch.rand(env.num_envs, 3, device=env.device) * 2 - env.scene.env_origins
    with torch.inference_mode():
        for count in range(20):
            if count % 10 == 0:
                env.reset()
            env.step(target_position)
    env.close()


def test_scale_randomization_rejects_replicate_physics():
    sim_utils.create_new_stage()
    cfg_failure = CubeEnvCfg()
    cfg_failure.scene.replicate_physics = True
    with pytest.raises(RuntimeError):
        ManagerBasedEnv(cfg_failure)
