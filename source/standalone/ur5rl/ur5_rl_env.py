from __future__ import annotations

import math
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
import torch
from collections.abc import Sequence

from omni.isaac.lab_assets.cartpole import CARTPOLE_CFG

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import sample_uniform


@configclass
class HawUr5EnvCfg(DirectRLEnvCfg):
    # env
    num_actions = 12
    num_observations = 4
    num_states = 5
    reward_scale_example = 1.0
    decimation = 2
    action_scale = 1.0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # Gripper parameters

    # robot
    robot_cfg: ArticulationCfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path="omniverse://localhost/MyAssets/haw_ur5_assembled/haw_u5_with_gripper.usd"
        ),
        prim_path="/World/envs/env_.*/ur5",
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit=None,
                velocity_limit=None,
                stiffness=None,
                damping=None,
            ),
        },
    )

    arm_dof_name = [
        "shoulder_pan_joint",  # 0
        "shoulder_lift_joint",  # -110
        "elbow_joint",  # 110
        "wrist_1_joint",  # -180
        "wrist_2_joint",  # -90
        "wrist_3_joint",  # 0
    ]
    gripper_dof_name = [
        "left_outer_knuckle_joint",
        "left_inner_finger_joint",
        "left_inner_knuckle_joint",
        "right_inner_knuckle_joint",
        "right_outer_knuckle_joint",
        "right_inner_finger_joint",
    ]

    haw_ur5_dof_name = arm_dof_name + gripper_dof_name

    action_dim = len(arm_dof_name) + len(gripper_dof_name)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=10, env_spacing=2.0, replicate_physics=True
    )

    # reset conditions
    # ...

    # reward scales
    # ...


class HawUr5Env(DirectRLEnv):
    cfg: HawUr5EnvCfg

    def __init__(self, cfg: HawUr5EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._arm_dof_idx, _ = self.robot.find_joints(self.cfg.arm_dof_name)
        self._gripper_dof_idx, _ = self.robot.find_joints(self.cfg.gripper_dof_name)
        self.haw_ur5_dof_idx, _ = self.robot.find_joints(self.cfg.haw_ur5_dof_name)
        self.action_scale = self.cfg.action_scale

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        self.action_dim = len(self._arm_dof_idx) + len(self._gripper_dof_idx)

    def _setup_scene(self):
        # add Articulation
        self.robot = Articulation(self.cfg.robot_cfg)

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        # add articultion to scene
        self.scene.articulations["ur5"] = self.robot

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Separate the main joint actions (first 6) and the gripper action (last one)
        main_joint_actions = actions[:, :6]  # Shape: (num_envs, 6)
        gripper_actions = actions[:, 6]  # Shape: (num_envs)
        main_joint_actions = self.cfg.action_scale * main_joint_actions.clone()

        # Convert each gripper action to the corresponding 6 gripper joint positions (min max 36 joint limit)
        gripper_joint_positions = torch.stack(
            [
                36 * gripper_actions,  # "left_outer_knuckle_joint"
                -36 * gripper_actions,  # "left_inner_finger_joint"
                -36 * gripper_actions,  # "left_inner_knuckle_joint"
                -36 * gripper_actions,  # "right_inner_knuckle_joint"
                36 * gripper_actions,  # "right_outer_knuckle_joint"
                36 * gripper_actions,  # "right_inner_finger_joint"
            ],
            dim=1,
        )  # Shape: (num_envs, 6)

        # Concatenate the main joint actions with the gripper joint positions
        full_joint_actions = torch.cat(
            (main_joint_actions, gripper_joint_positions), dim=1
        )  # Shape: (num_envs, 12)

        # Apply action scaling and assign to self.actions
        self.actions = full_joint_actions

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(
            self.actions, joint_ids=self.haw_ur5_dof_idx
        )

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.joint_pos[self._arm_dof_idx].unsqueeze(dim=1),
                self.joint_vel[self._arm_dof_idx].unsqueeze(dim=1),
                self.joint_pos[self._gripper_dof_idx].unsqueeze(dim=1),
                self.joint_vel[self._gripper_dof_idx].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = torch.zeros(1, device=self.device)
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        out_of_bounds = torch.zeros(1, device=self.device)
        time_out = torch.zeros(1, device=self.device)
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES  # type: ignore

        # General resetting tasks (timers etc.)
        super()._reset_idx(env_ids)  # type: ignore

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        # TODO Add random noise to joint positions for domain randomization

        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel
        # self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
