# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

import omni.isaac.core.utils.torch as torch_utils
from omni.isaac.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate
from gymnasium import spaces

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObjectCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
import numpy as np
from omni.isaac.lab.utils import configclass
import omni.isaac.lab.utils.math as math_utils
import omni.isaac.lab.envs.mdp as mdp

from omni.isaac.lab.sensors import ContactSensor, ContactSensorCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg, PhysxCfg, RenderCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab_assets import SIGMABAN_CFG


@configclass
class EventCfg:
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    robot_joint_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    robot_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    randomize_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


@configclass
class SigmabanEnvCfg(DirectRLEnvCfg):

    # env
    # Period for the agent to apply control [s]
    dt = 0.05
    dt_sim = 1 / 500  # 1/500
    episode_length_s = 7.0
    decimation = round(dt / dt_sim)  # 25

    events: EventCfg = EventCfg()

    # simulation
    physx: PhysxCfg = PhysxCfg(gpu_max_rigid_patch_count=4096 * 4096)
    sim: SimulationCfg = SimulationCfg(dt=dt_sim, render_interval=decimation, physx=physx)

    # Ground
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = SIGMABAN_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/.*", debug_vis=False)
    # contact_sensor: ContactSensorCfg = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/.*/collisions/.*", history_length=3, track_air_time=True, debug_vis=True)
    ##########################################################################################################
    # Duration of the stabilization pre-simulation (waiting for the gravity to stabilize the robot) [s]
    stabilization_time = 2.0
    # Maximum command angular velocity [rad/s]
    vmax = 2 * np.pi
    # Is the render done in "realtime"
    render_realtime = True
    # Probability of seeding the robot in finale position
    reset_final_p = 0.1
    # Termination conditions
    terminate_upside_down = True
    terminate_gyro = True
    terminate_shock = False
    # Randomization
    random_angles = 1.5  # [+- deg]
    random_time_ratio = [0.98, 1.02]  # [+- ratio]
    random_friction = [0.25, 1.5]  # [friction]
    random_body_mass = [0.8, 1.2]  # [+- ratio]
    random_body_com_pos = 5e-3  # [+- m]
    random_damping = [0.8, 1.2]  # [+- ratio]
    random_frictionloss = [0.8, 1.2]  # [+- ratio]
    random_v = [13.8, 16.8]  # [volts]
    nominal_v = 15  # [volts]
    # Control type (position, velocity or error)
    control = "velocity"
    dofs_ctrled = ["elbow", "shoulder_pitch", "hip_pitch", "knee", "ankle_pitch"]
    interpolate = True
    # Delay for velocity [s]
    qdot_delay = 0.030
    tilt_delay = 0.050
    dtilt_delay = 0.050
    # Previous actions
    previous_actions_size = 1
    # Cost scales
    action_cost_scale = 1e-1
    variation_cost_scale = 5e-2
    self_collision_cost_scale = 1e-1

    # Spaces
    observation_space = 22
    state_space = 0
    action_space = 5
    if control == "velocity":
        action_space = spaces.Box(
            np.array([-vmax] * len(dofs_ctrled), dtype=np.float32),
            np.array([vmax] * len(dofs_ctrled), dtype=np.float32),
            dtype=np.float32,
        )


def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


class StandUpEnv(DirectRLEnv):
    cfg: SigmabanEnvCfg

    def __init__(self, cfg: DirectRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dofs_ctrled_idx, self.names = self.robot.find_joints([f"left_{dof}" for dof in self.cfg.dofs_ctrled])
        self.dof_limits = [self.robot.data.soft_joint_pos_limits[0, dof, :] for dof in self.dofs_ctrled_idx]

        # Pre-fetching indexes and sites for faster evaluation
        self.right_dofs_ctrled_idx = torch.tensor(
            [self.robot.find_joints(f"right_{dof}")[0][0] for dof in self.cfg.dofs_ctrled],
            dtype=torch.int,
            device=self.sim.device,
        )
        self.left_dofs_ctrled_idx = torch.tensor(
            [self.robot.find_joints(f"left_{dof}")[0][0] for dof in self.cfg.dofs_ctrled],
            dtype=torch.int,
            device=self.sim.device,
        )

        self.range_low = torch.tensor([r[0] for r in self.dof_limits], dtype=torch.float32, device=self.sim.device)
        self.range_high = torch.tensor([r[1] for r in self.dof_limits], dtype=torch.float32, device=self.sim.device)

        max_variation = self.cfg.dt * self.cfg.vmax
        self.delta_max = torch.tensor(
            [max_variation] * len(self.cfg.dofs_ctrled), dtype=torch.float32, device=self.sim.device
        )

        # Target robot state (q_motors, tilt) [rad^6]
        # [elbow, shoulder_pitch, hip_pitch, knee, ankle_pitch, IMU_pitch]
        # in rad : [-0.864, 0.340, -0.907, 1.378, -0.638, -0.148]
        self.desired_state = torch.deg2rad(
            torch.tensor([-49.5, 19.5, -52, 79, -36.5, 8.5], dtype=torch.float32, device=self.sim.device)
        )

        # Window for qdot delay simulation
        self.q_history_size = max(1, round(self.cfg.qdot_delay / self.sim.cfg.dt))
        self.q_history = torch.zeros(
            size=(self.num_envs, self.q_history_size, len(self.cfg.dofs_ctrled)), device=self.sim.device
        )

        # Window for tilt delay simulation
        self.tilt_history_size = max(1, round(self.cfg.tilt_delay / self.sim.cfg.dt))
        self.tilt_history = torch.zeros(size=(self.num_envs, self.tilt_history_size), device=self.sim.device)
        # print("Tilt History shape: ", self.tilt_history.shape)

        self.dtilt_history_size = max(1, round(self.cfg.dtilt_delay / self.sim.cfg.dt))
        self.dtilt_history = torch.zeros(size=(self.num_envs, self.dtilt_history_size), device=self.sim.device)
        # print("Dtilt History shape: ", self.dtilt_history.shape)

        # Values for randomization
        self.trunk_body_idx, _ = self.robot.find_bodies("torso")
        self.trunk_mass = self.robot.data.default_mass[0, self.trunk_body_idx].detach().clone()
        # self.trunk_com = ?
        self.stiffness_original = self.robot.data.joint_stiffness[0, :].detach().clone()

        self.damping_original = self.robot.data.joint_damping[0, :].detach().clone()
        self.frictionloss_original = self.robot.data.joint_friction[0, :].detach().clone()

        self.body_quat_original = self.robot.data.body_quat_w[0, :].detach().clone()

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)

        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)

        # add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):

        self.actions = actions.clone()

        self.start_ctrl = self.robot.data.joint_pos_target[:, self.left_dofs_ctrled_idx].detach().clone()
        # print(f"Start Ctrl: {self.start_ctrl}")

        if self.cfg.control == "velocity":
            target_ctrl_unclipped = self.start_ctrl + self.actions * self.cfg.dt
            # print(f"Target Ctrl Unclipped: {target_ctrl_unclipped}")

        # Clipping the actions
        self.target_ctrl = torch.clip(
            target_ctrl_unclipped, self.start_ctrl - self.delta_max, self.start_ctrl + self.delta_max
        )

        self.target_ctrl = torch.clip(self.target_ctrl, self.range_low, self.range_high)

    def _apply_action(self):
        # action_variation = torch.abs(self.actions - self.previous_actions[:, -1, :])
        k = self._sim_step_counter % self.cfg.sim.render_interval

        # print(f"Joint Pos Target: {self.robot.data.joint_pos_target[0, self.left_dofs_ctrled_idx]}")

        if self.cfg.interpolate:
            alpha = (k + 1) / self.cfg.decimation
            self.ctrl_to_apply = self.start_ctrl + alpha * (self.target_ctrl - self.start_ctrl)

            self.robot.set_joint_position_target(self.ctrl_to_apply, joint_ids=self.right_dofs_ctrled_idx)
            self.robot.set_joint_position_target(self.ctrl_to_apply, joint_ids=self.left_dofs_ctrled_idx)

        # print("Ctrl to apply: ", self.ctrl_to_apply)
        # print("Joint pos target", self.robot.data.joint_pos_target[:, self.left_dofs_ctrled_idx].detach().clone())

        self.robot.set_joint_effort_target(torch.zeros(size=(self.num_envs, 20), device=self.sim.device))

        self.q = self.robot.data.joint_pos[:, self.left_dofs_ctrled_idx].detach().clone()
        self.q_history = torch.cat((self.q_history, self.q.unsqueeze(1)), dim=1)

        self.tilt = get_tilt(self.robot.data.root_quat_w)
        self.tilt_history = torch.cat((self.tilt_history, self.tilt.unsqueeze(1)), dim=1)

        self.dtilt = self.robot.data.root_ang_vel_b[:, 1].detach().clone()

        pass

    def _compute_intermediate_values(self):
        (
            self.q,
            self.q_dot,
            # self.ctrl_to_apply,
        ) = compute_intermediate_values(
            self.q_history,
            self.q_history_size,
            self.cfg.sim.dt,
            # self.ctrl_to_apply,
            self.left_dofs_ctrled_idx,
        )

    def _get_observations(self) -> dict:
        self._compute_intermediate_values()
        # print(f"Q: {self.q[-5: , :]}")
        # print(f"Q_dot: {self.q_dot[-5: , :]}")
        # print(f"Control to apply : {self.ctrl_to_apply}")
        # print(f"Joint position command: {self.robot.data.joint_pos_target[: , self.left_dofs_ctrled_idx]}")
        # print(f"Tilt History: {self.tilt_history[-5:, 0]}")
        # print(f"Dtilt History: {self.dtilt_history[-5:, 0]}")
        # print(f"Previous Actions: {self.previous_actions[-5:, :, :]}")

        obs = torch.cat(
            (
                self.q,
                self.q_dot,
                self.robot.data.joint_pos_target[:, self.left_dofs_ctrled_idx],
                self.tilt_history[:, 0].unsqueeze(-1),
                self.dtilt_history[:, 0].unsqueeze(-1),
                self.previous_actions.flatten(start_dim=1, end_dim=2),
            ),
            dim=-1,
        )

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward, self.previous_actions = compute_rewards(
            self.num_envs,
            self.desired_state,
            self.q_history,
            self.tilt_history,
            self.actions,
            self.previous_actions,
            self.cfg.previous_actions_size,
            self.cfg.action_cost_scale,
            self.cfg.variation_cost_scale,
            self.cfg.self_collision_cost_scale,
        )

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        self.q_history = self.q_history[:, -self.q_history_size :]
        self.tilt_history = self.tilt_history[:, -self.tilt_history_size :]
        self.dtilt_history = self.dtilt_history[:, -self.dtilt_history_size :]

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        died = torch.tensor([False] * self.num_envs, device=self.sim.device)

        if self.cfg.terminate_upside_down:
            tilt = get_tilt(self.robot.data.root_quat_w)
            upside_down = torch.rad2deg(torch.abs(tilt)) > 135  # 135deg
            died = died | upside_down

        if self.cfg.terminate_gyro:
            gyro = torch.abs(self.robot.data.root_ang_vel_b[:, 1]) > 5
            died = died | gyro

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        initial_q, initial_tilt = randomize_fall(
            len(env_ids),
            self.cfg.reset_final_p,
            self.right_dofs_ctrled_idx,
            self.range_low,
            self.range_high,
            self.desired_state,
        )

        default_joint_pos = self.robot.data.default_joint_pos[env_ids]
        default_joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        rotation_to_apply = math_utils.quat_mul(
            default_root_state[:, 3:7],
            math_utils.quat_from_angle_axis(
                initial_tilt, torch.tensor([0, 1, 0], dtype=torch.float32, device=self.sim.device)
            ),
        )

        default_root_state[:, 3:7] = rotation_to_apply


        self.robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)

        self.robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, None, env_ids)

        # print("Initial Q/Velocity: ", initial_q, default_joint_vel[:, self.left_dofs_ctrled_idx])

        self.robot.write_joint_state_to_sim(
            initial_q, default_joint_vel[:, self.left_dofs_ctrled_idx], self.left_dofs_ctrled_idx, env_ids
        )
        self.robot.write_joint_state_to_sim(
            initial_q, default_joint_vel[:, self.right_dofs_ctrled_idx], self.right_dofs_ctrled_idx, env_ids
        )

        self.robot.set_joint_position_target(initial_q, self.right_dofs_ctrled_idx, env_ids)
        self.robot.set_joint_position_target(initial_q, self.left_dofs_ctrled_idx, env_ids)

        # for _ in range(round(self.cfg.stabilization_time / self.sim.cfg.dt)):
        #     self.sim.step(render=False)

        # Initializing Q history
        q = self.robot.data.joint_pos[env_ids][:, self.left_dofs_ctrled_idx]

        self.q_history[env_ids, :, :] = q.unsqueeze(1).repeat(1, self.q_history_size, 1)

        # Initializing Tilt history
        self.tilt_history[env_ids, -1] = get_tilt(self.robot.data.root_quat_w[env_ids, :])
        # print("Tilt history: ", self.tilt_history)

        # Initializing Dtilt history
        self.dtilt_history = torch.zeros(size=(self.num_envs, self.dtilt_history_size), device=self.sim.device)

        # Initializing previous actions
        self.previous_actions = torch.zeros(
            size=(self.num_envs, self.cfg.previous_actions_size, len(self.cfg.dofs_ctrled)), device=self.sim.device
        )

# @torch.jit.script
# def apply_angular_offset(self, joint: str, offset: float):
#     """Apply an angular offset to a joint

#     Args:
#         joint: name of the joint to apply the offset to
#         offset: angular offset to apply
#     """
#     joint_idx = self.robot.find_joints(joint)
#     self.robot.date.body_quat_w[:, joint_idx] += offset

#     self.robot.data.joint_pos[:, joint_idx] += offset


@torch.jit.script
def get_tilt(quat) -> torch.Tensor:
    """Get the tilt of the robot along the y axis

    Args:
        quat: quaternion representing the rotation of the torso

    Returns:
        tilt: tilt of the torso in the x-z plane
    """

    q_w, q_x, q_y, q_z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    sin_pitch = 2.0 * (q_w * q_y - q_z * q_x)

    pitch = torch.asin(torch.clamp(sin_pitch, -1.0, 1.0))
    # equivalent to
    # pitch = torch.where(torch.abs(sin_pitch) >= 1, copysign(torch.pi / 2.0, sin_pitch), torch.asin(sin_pitch))

    tilt = normalize_angle(pitch)  # arctan2(sin_pitch, cos_pitch)

    return tilt


@torch.jit.script
def randomize_fall(
    num_envs: int,
    reset_final_p: float,
    dofs_ctrled: torch.Tensor,
    range_low: torch.Tensor,
    range_high: torch.Tensor,
    desired_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = dofs_ctrled.device
    target = torch.rand(num_envs) < reset_final_p
    initial_q = torch.empty((num_envs, len(dofs_ctrled)), device=device).uniform_(-torch.pi, torch.pi)

    initial_q[target] = desired_target[: len(dofs_ctrled)]
    offset = torch.empty((num_envs, len(dofs_ctrled)), device=device).uniform_(-0.1, 0.1)
    initial_q[target, 2] -= offset[target, 2]
    initial_q[target, 3] += offset[target, 3] * 2
    initial_q[target, 4] -= offset[target, 4]

    initial_q = torch.clip(initial_q, range_low, range_high)

    initial_tilt = torch.empty((num_envs), device=device).uniform_(-torch.pi / 2, torch.pi / 2)
    initial_tilt[target] = desired_target[-1]

    return initial_q, initial_tilt


@torch.jit.script
def compute_rewards(
    num_envs: int,
    desired_target: torch.Tensor,
    q_history: torch.Tensor,
    tilt_history: torch.Tensor,
    actions: torch.Tensor,
    previous_actions_old: torch.Tensor,
    previous_actions_size: int,
    action_cost_scale: float,
    variation_cost_scale: float,
    self_collision_cost_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = actions.device

    desired_target = desired_target.unsqueeze(0).repeat(num_envs, 1)

    current_state = torch.cat((q_history[:, -1, :], tilt_history[:, -1].unsqueeze(-1)), dim=-1)

    action_variation = torch.abs(actions - previous_actions_old[:, -1, :])

    previous_actions = torch.cat((previous_actions_old, actions.unsqueeze(1)), dim=1)
    previous_actions = previous_actions[:, -previous_actions_size:, :]

    state_reward = torch.exp(-20 * (torch.linalg.norm(current_state - desired_target, ord=2, dim=-1) ** 2))

    action_cost = torch.exp(-torch.linalg.norm(actions, ord=2, dim=-1))

    variation_cost = torch.exp(-torch.linalg.norm(action_variation, ord=2, dim=-1))

    # Self collision computation
    force_torque = torch.zeros((num_envs, 6), device=device)

    # self_collision_cost = torch.exp(-torch.linalg.norm(force_torque, ord=2, dim=-1) ** 2)

    total_reward = (
        state_reward
        + action_cost * action_cost_scale
        + variation_cost * variation_cost_scale
        # + self_collision_cost * self_collision_cost_scale
    )

    return total_reward, previous_actions


@torch.jit.script
def compute_intermediate_values(
    q_history: torch.Tensor,
    q_history_size: int,
    sim_dt: float,
    # computed_torque: torch.Tensor,
    dofs_ctrled_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:

    q = q_history[:, -1, :]
    q_dot = (q - q_history[:, 0, :]) / (q_history_size * sim_dt)

    # ctrl = computed_torque[:, dofs_ctrled_idx]

    return (q, q_dot)
