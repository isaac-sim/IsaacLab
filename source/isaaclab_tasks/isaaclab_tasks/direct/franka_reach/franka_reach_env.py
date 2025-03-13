# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane




@configclass
class FrankaReachEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 2
    action_space = 9
    observation_space = 23
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)

    # robot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/franka_instanceable.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 1.157,
                "panda_joint2": -1.066,
                "panda_joint3": -0.155,
                "panda_joint4": -2.239,
                "panda_joint5": -1.841,
                "panda_joint6": 1.003,
                "panda_joint7": 0.469,
                "panda_finger_joint.*": 0.035,
            },
            pos=(1.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )

    # # table
    # table = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Table",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)),
    # )

    # # # ground plane
    # ground = AssetBaseCfg(
    #     prim_path="/World/ground",
    #     spawn=sim_utils.GroundPlaneCfg(),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    # )

    # reset
    max_target_dist = 0.2
    initial_joint_pos_range = [-0.1, 0.1]

    # reward scales
    dist_reward_scale = 1.5
    rot_reward_scale = 1.5
    reach_reward_scale = 10.0
    action_penalty_scale = 0.05
    finger_reward_scale = 2.0

    action_scale = 7.5
    dof_velocity_scale = 0.1

class FrankaReachEnv(DirectRLEnv):

    cfg: FrankaReachEnvCfg

    def __init__(self, cfg: FrankaReachEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # self._setup_scene()

    def _setup_scene(self):
        """Initialize simulation scene."""
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -0.4))

        # spawn a usd file of a table into the scene
        cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
        cfg.func(
            "/World/envs/env_.*/Table", cfg, translation=(0.55, 0.0, 0.0), orientation=(0.70711, 0.0, 0.0, 0.70711)
        )

        self._robot = Articulation(self.cfg.robot)

        self.scene.clone_environments(copy_from_source=False)

        self.scene.articulations["robot"] = self._robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _get_rewards(self) -> torch.Tensor:
        target_pos = torch.tensor([0.5, 0.0, 0.5], device=self.device)
        end_effector_pos = self.robot.data.body_pos_w[:, self.robot.find_bodies("panda_link7")[0][0]]
        distance = torch.norm(end_effector_pos - target_pos, dim=-1)
        dist_reward = self.cfg.dist_reward_scale / (1.0 + distance)
        action_penalty = torch.sum(self.actions**2, dim=-1) * self.cfg.action_penalty_scale
        return dist_reward - action_penalty

    def _get_observations(self) -> dict:
        dof_pos_scaled = (2.0 * (self.robot.data.joint_pos - self.robot.data.default_joint_pos))
        obs = torch.cat((dof_pos_scaled, self.robot.data.joint_vel), dim=-1)
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        target_pos = torch.tensor([0.5, 0.0, 0.5], device=self.device)
        end_effector_pos = self.robot.data.body_pos_w[:, self.robot.find_bodies("panda_link7")[0][0]]
        distance = torch.norm(end_effector_pos - target_pos, dim=-1)
        terminated = distance < self.cfg.max_target_dist
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        joint_pos = self.robot.data.default_joint_pos[env_ids] + sample_uniform(
            self.cfg.initial_joint_pos_range[0],
            self.cfg.initial_joint_pos_range[1],
            joint_pos.shape,
            self.device,
        )
        self.robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.actions)
