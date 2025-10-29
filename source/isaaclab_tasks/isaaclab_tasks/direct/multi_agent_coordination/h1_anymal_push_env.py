# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import torch

import isaacsim.core.utils.torch as torch_utils
from isaacsim.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_from_angle_axis

##
# Pre-defined configs
##
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip
from isaaclab_assets.robots.unitree import H1_CFG  # isort: skip


def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material_0 = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot_0", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    physics_material_1 = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot_1", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass_0 = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot_0", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    add_base_mass_1 = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot_1", body_names="pelvis"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )


@configclass
class HeterogeneousPushMultiAgentEnvCfg(DirectMARLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 4
    anymal_action_scale = 0.5
    action_space = 12
    action_spaces = {"robot_0": 12, "robot_1": 19}

    observation_spaces = {"robot_0": 48, "robot_1": 72}
    state_space = 0
    state_spaces = {f"robot_{i}": 0 for i in range(2)}
    possible_agents = ["robot_0", "robot_1"]

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=6.0, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()

    # robot
    robot_0: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Robot_0")
    contact_sensor_0: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_0/.*", history_length=3, update_period=0.005, track_air_time=True
    )
    robot_0.init_state.rot = (1.0, 0.0, 0.0, 1.0)
    robot_0.init_state.pos = (-1.0, 0.0, 0.5)

    robot_1: ArticulationCfg = H1_CFG.replace(prim_path="/World/envs/env_.*/Robot_1")
    robot_1.init_state.rot = (1.0, 0.0, 0.0, 1.0)
    robot_1.init_state.pos = (1.0, 0.0, 1.0)

    # rec prism
    cfg_rec_prism = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.CuboidCfg(
            size=(3, 2, 2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=10),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            # visual_material=sim_utils.GlassMdlCfg(glass_color=(0.0, 1.0, 0.0), frosting_roughness=0.7),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 2, 0.6), rot=(1.0, 0.0, 0.0, 0.0)),  # started the bar lower
    )

    # reward scales (override from flat config)
    flat_orientation_reward_scale = 0.0

    # reward scales
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.5
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -2.5e-5
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.01
    feet_air_time_reward_scale = 0.5
    undersired_contact_reward_scale = -1.0
    flat_orientation_reward_scale = -5.0
    flat_bar_roll_angle_reward_scale = -1.0
    angular_velocity_scale: float = 0.25
    dof_vel_scale: float = 0.1
    h1_action_scale = 1.0
    termination_height: float = 0.8
    anymal_min_z_pos = 0.3
    h1_min_z_pos = 0.8

    joint_gears: list = [
        50.0,  # left_hip_yaw
        50.0,  # right_hip_yaw
        50.0,  # torso
        50.0,  # left_hip_roll
        50.0,  # right_hip_roll
        50.0,  # left_shoulder_pitch
        50.0,  # right_shoulder_pitch
        50.0,  # left_hip_pitch
        50.0,  # right_hip_pitch
        50.0,  # left_shoulder_roll
        50.0,  # right_shoulder_roll
        50.0,  # left_knee
        50.0,  # right_knee
        50.0,  # left_shoulder_yaw
        50.0,  # right_shoulder_yaw
        50.0,  # left_ankle
        50.0,  # right_ankle
        50.0,  # left_elbow
        50.0,  # right_elbow
    ]


def define_markers() -> VisualizationMarkers:
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            "sphere1": sim_utils.SphereCfg(
                radius=0.15,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.0, 0.0)),
            ),
            "sphere2": sim_utils.SphereCfg(
                radius=0.15,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
            ),
            "arrow1": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.1, 0.1, 1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.0, 0.0)),
            ),
            "arrow2": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.1, 0.1, 1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)


class HeterogeneousPushMultiAgentEnv(DirectMARLEnv):
    cfg: HeterogeneousPushMultiAgentEnvCfg

    def __init__(
        self,
        cfg: HeterogeneousPushMultiAgentEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)
        # Joint position command (deviation from default joint positions)
        self.actions = {
            agent: torch.zeros(self.num_envs, action_space, device=self.device)
            for agent, action_space in self.cfg.action_spaces.items()
        }
        self.previous_actions = {
            agent: torch.zeros(self.num_envs, action_space, device=self.device)
            for agent, action_space in self.cfg.action_spaces.items()
        }
        self.base_bodies = ["base", "pelvis"]
        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._joint_dof_idx, _ = self.robots["robot_1"].find_joints(".*")

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
            ]
        }

        self.base_ids = {}
        self.feet_ids = {}
        self.undesired_body_contact_ids = {}

        for idx, (robot_id, contact_sensor) in enumerate(self.contact_sensors.items()):
            _base_id, _ = contact_sensor.find_bodies(self.base_bodies[idx])
            _feet_ids, _ = contact_sensor.find_bodies(".*FOOT")
            _undesired_contact_body_ids, _ = contact_sensor.find_bodies(".*THIGH")
            self.base_ids[robot_id] = _base_id
            self.feet_ids[robot_id] = _feet_ids
            self.undesired_body_contact_ids[robot_id] = _undesired_contact_body_ids

        self.potentials = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)
        self.prev_potentials = torch.zeros_like(self.potentials)
        self.targets = torch.tensor([1000, 0, 0], dtype=torch.float32, device=self.sim.device).repeat(
            (self.num_envs, 1)
        )
        self.joint_gears = torch.tensor(self.cfg.joint_gears, dtype=torch.float32, device=self.sim.device)
        self.targets = torch.tensor([1000, 0, 0], dtype=torch.float32, device=self.sim.device).repeat(
            (self.num_envs, 1)
        )
        self.targets += self.scene.env_origins
        self.heading_vec = torch.tensor([1, 0, 0], dtype=torch.float32, device=self.sim.device).repeat(
            (self.num_envs, 1)
        )
        self.up_vec = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        self.start_rotation = torch.tensor([1, 0, 0, 0], device=self.sim.device, dtype=torch.float32)
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))
        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

    def _setup_scene(self):
        self.num_robots = sum(1 for key in self.cfg.__dict__.keys() if "robot_" in key)
        self.robots = {}
        self.contact_sensors = {}
        self.height_scanners = {}
        self.my_visualizer = define_markers()
        self.object = RigidObject(self.cfg.cfg_rec_prism)

        self.scene.rigid_objects["object"] = self.object

        for i in range(self.num_robots):
            robot_id = f"robot_{i}"
            if robot_id in self.cfg.__dict__:
                self.robots[f"robot_{i}"] = Articulation(self.cfg.__dict__["robot_" + str(i)])
                self.scene.articulations[f"robot_{i}"] = self.robots[f"robot_{i}"]

            contact_sensor_id = "contact_sensor_" + str(i)

            if contact_sensor_id in self.cfg.__dict__:
                self.contact_sensors[f"robot_{i}"] = ContactSensor(self.cfg.__dict__["contact_sensor_" + str(i)])
                self.scene.sensors[f"robot_{i}"] = self.contact_sensors[f"robot_{i}"]

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict):
        # We need to process the actions for each scene independently
        self.processed_actions = copy.deepcopy(actions)

        robot_id = "robot_0"
        robot_action_space = self.action_spaces[robot_id].shape[0]
        self.actions[robot_id] = actions[robot_id][:, :robot_action_space].clone()
        self.processed_actions[robot_id] = (
            self.cfg.anymal_action_scale * self.actions[robot_id] + self.robots[robot_id].data.default_joint_pos
        )

        robot_id = "robot_1"
        self.actions[robot_id] = actions[robot_id].clone()

    def _get_anymal_fallen(self):
        agent_dones = []
        robot = self.robots["robot_0"]
        died = robot.data.body_com_pos_w[:, 0, 2].view(-1) < self.cfg.anymal_min_z_pos
        agent_dones.append(died)

        robot = self.robots["robot_1"]
        died = robot.data.body_com_pos_w[:, 0, 2].view(-1) < self.cfg.h1_min_z_pos
        agent_dones.append(died)

        return torch.any(torch.stack(agent_dones), dim=0)

    def _apply_action(self):

        robot_id = "robot_0"
        self.robots[robot_id].set_joint_position_target(self.processed_actions[robot_id])

        robot_id = "robot_1"
        forces = self.cfg.h1_action_scale * self.joint_gears * self.actions[robot_id]
        self.robots[robot_id].set_joint_effort_target(forces, joint_ids=self._joint_dof_idx)

    def _compute_intermediate_values(self):
        self.torso_position, self.torso_rotation = (
            self.robots["robot_1"].data.root_link_pos_w,
            self.robots["robot_1"].data.root_link_quat_w,
        )
        self.velocity, self.ang_velocity = (
            self.robots["robot_1"].data.root_com_lin_vel_w,
            self.robots["robot_1"].data.root_com_ang_vel_w,
        )
        self.dof_pos, self.dof_vel = self.robots["robot_1"].data.joint_pos, self.robots["robot_1"].data.joint_vel

        (
            self.up_proj,
            self.heading_proj,
            self.up_vec,
            self.heading_vec,
            self.vel_loc,
            self.angvel_loc,
            self.roll,
            self.pitch,
            self.yaw,
            self.angle_to_target,
            self.dof_pos_scaled,
            self.prev_potentials,
            self.potentials,
        ) = compute_intermediate_values(
            self.targets,
            self.torso_position,
            self.torso_rotation,
            self.velocity,
            self.ang_velocity,
            self.dof_pos,
            self.robots["robot_1"].data.soft_joint_pos_limits[0, :, 0],
            self.robots["robot_1"].data.soft_joint_pos_limits[0, :, 1],
            self.inv_start_rot,
            self.basis_vec0,
            self.basis_vec1,
            self.potentials,
            self.prev_potentials,
            self.cfg.sim.dt,
        )

    def _get_observations(self) -> dict:
        self.previous_actions = copy.deepcopy(self.actions)

        obs = {}

        robot_id = "robot_0"
        robot = self.robots[robot_id]
        # anymal_commands = torch.zeros_like(self._commands)
        obs[robot_id] = torch.cat(
            [
                tensor
                for tensor in (
                    robot.data.root_com_lin_vel_b,
                    robot.data.root_com_ang_vel_b,
                    robot.data.projected_gravity_b,
                    self._commands,
                    robot.data.joint_pos - robot.data.default_joint_pos,
                    robot.data.joint_vel,
                    None,
                    self.actions[robot_id],
                )
                if tensor is not None
            ],
            dim=-1,
        )

        robot_id = "robot_1"
        robot = self.robots[robot_id]
        obs[robot_id] = torch.cat(
            (
                self.torso_position[:, 2].view(-1, 1),
                self.vel_loc,
                self.angvel_loc * self.cfg.angular_velocity_scale,
                normalize_angle(self.yaw).unsqueeze(-1),
                normalize_angle(self.roll).unsqueeze(-1),
                normalize_angle(self.angle_to_target).unsqueeze(-1),
                self.up_proj.unsqueeze(-1),
                self.heading_proj.unsqueeze(-1),
                self.dof_pos_scaled,
                self.dof_vel * self.cfg.dof_vel_scale,
                self.actions[robot_id],
                self._commands,
            ),
            dim=-1,
        )
        return obs

    def get_y_euler_from_quat(self, quaternion):
        w, x, y, z = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]
        y_euler_angle = torch.arcsin(2 * (w * y - z * x))
        return y_euler_angle

    def _draw_markers(self, command):
        xy_commands = command.clone()
        z_commands = xy_commands[:, 2].clone()
        xy_commands[:, 2] = 0

        marker_ids = torch.concat(
            [
                0 * torch.zeros(2 * self._commands.shape[0]),
                1 * torch.ones(self._commands.shape[0]),
                2 * torch.ones(self._commands.shape[0]),
                3 * torch.ones(self._commands.shape[0]),
            ],
            dim=0,
        )

        bar_pos = self.object.data.body_com_pos_w.squeeze(1).clone()
        bar_yaw = self.object.data.root_com_ang_vel_b[:, 2].clone()

        scale1 = torch.ones((self._commands.shape[0], 3), device=self.device)
        scale1[:, 0] = torch.abs(z_commands)

        scale2 = torch.ones((self._commands.shape[0], 3), device=self.device)
        scale2[:, 0] = torch.abs(bar_yaw)

        offset1 = torch.zeros((self._commands.shape[0], 3), device=self.device)
        offset1[:, 1] = 0

        offset2 = torch.zeros((self._commands.shape[0], 3), device=self.device)
        offset2[:, 1] = 0

        _90 = (-3.14 / 2) * torch.ones(self._commands.shape[0]).to(self.device)

        marker_orientations = quat_from_angle_axis(
            torch.concat(
                [
                    torch.zeros(3 * self._commands.shape[0]).to(self.device),
                    torch.sign(z_commands) * _90,
                    torch.sign(bar_yaw) * _90,
                ],
                dim=0,
            ),
            torch.tensor([0.0, 1.0, 0.0], device=self.device),
        )

        marker_scales = torch.concat(
            [torch.ones((3 * self._commands.shape[0], 3), device=self.device), scale1, scale2], dim=0
        )

        # obj_vel = self.object.data.root_com_lin_vel_b.clone()
        # obj_vel[:, 2] = 0

        obj_vel = self.object.data.root_com_lin_vel_b
        obj_vel[:, 2] = 0

        offset = torch.tensor([0, 2, 0], device=self.device)

        marker_locations = torch.concat(
            [
                bar_pos + offset,
                bar_pos + xy_commands + offset,
                bar_pos + obj_vel + offset,
                bar_pos + offset1 + offset,
                bar_pos + offset2 + offset,
            ],
            dim=0,
        )

        self.my_visualizer.visualize(
            marker_locations, marker_orientations, scales=marker_scales, marker_indices=marker_ids
        )

    def _get_rewards(self) -> dict:
        reward = {}

        bar_commands = torch.stack([-self._commands[:, 1], self._commands[:, 0], self._commands[:, 2]]).t()
        self._draw_markers(bar_commands)
        obj_xy_vel = self.object.data.root_com_lin_vel_b[:, :2]

        # linear velocity tracking
        lin_vel_error = torch.sum(torch.square(bar_commands[:, :2] - obj_xy_vel), dim=1)  # changing this to the bar
        lin_vel_error_mapped = torch.exp(-lin_vel_error)

        # angular velocity tracking
        yaw_rate_error = torch.square(self._commands[:, 2] - self.object.data.root_com_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error)

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        for key, val in rewards.items():
            self._episode_sums[key] += val

        return {"robot_0": reward, "robot_1": reward}

    def _get_too_far_away(self):
        anymal_pos = self.robots["robot_0"].data.body_com_pos_w[:, 0, :]
        h1_pos = self.robots["robot_1"].data.body_com_pos_w[:, 0, :]

        box_pos = self.object.data.body_com_pos_w[:, 0, :]

        anymal_too_far = (
            torch.sqrt(torch.square(anymal_pos[:, 0] - box_pos[:, 0]) + torch.square(anymal_pos[:, 1] - box_pos[:, 1]))
            > 3
        )
        h1_too_far = (
            torch.sqrt(torch.square(h1_pos[:, 0] - box_pos[:, 0]) + torch.square(h1_pos[:, 1] - box_pos[:, 1])) > 3
        )

        return torch.logical_or(anymal_too_far, h1_too_far)

    def _get_dones(self) -> tuple[dict, dict]:
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        h1_died = self.torso_position[:, 2] < self.cfg.termination_height
        anymal_fallen = self._get_anymal_fallen()
        too_far = self._get_too_far_away()
        dones = torch.logical_or(h1_died, anymal_fallen)
        dones = torch.logical_or(dones, too_far)
        return {key: time_out for key in self.robots.keys()}, {key: dones for key in self.robots.keys()}

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        object_default_state[:, 0:3] = object_default_state[:, 0:3] + self.scene.env_origins[env_ids]
        self.object.write_root_state_to_sim(object_default_state, env_ids)
        self.object.reset(env_ids)

        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # Joint position command (deviation from default joint positions)
        for agent, action_space in self.cfg.action_spaces.items():
            self.actions[agent][env_ids] = torch.zeros(env_ids.shape[0], action_space, device=self.device)
            self.previous_actions[agent][env_ids] = torch.zeros(env_ids.shape[0], action_space, device=self.device)

        # X/Y linear velocity and yaw angular velocity commands
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)
        self._commands[env_ids, 0] = torch.zeros_like(self._commands[env_ids, 0]).uniform_(0.5, 1.0)

        # reset idx for anymal #
        robot = self.robots["robot_0"]
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = robot._ALL_INDICES
        robot.reset(env_ids)

        joint_pos = robot.data.default_joint_pos[env_ids]
        joint_vel = robot.data.default_joint_vel[env_ids]
        default_root_state = robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        robot = self.robots["robot_1"]
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = robot._ALL_INDICES
        robot.reset(env_ids)

        # Reset robot state
        to_target = self.targets[env_ids] - default_root_state[:, :3]
        to_target[:, 2] = 0.0
        self.potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.cfg.sim.dt
        self._compute_intermediate_values()
        joint_pos = robot.data.default_joint_pos[env_ids]
        joint_vel = robot.data.default_joint_vel[env_ids]
        default_root_state = robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()


@torch.jit.script
def compute_intermediate_values(
    targets: torch.Tensor,
    torso_position: torch.Tensor,
    torso_rotation: torch.Tensor,
    velocity: torch.Tensor,
    ang_velocity: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_lower_limits: torch.Tensor,
    dof_upper_limits: torch.Tensor,
    inv_start_rot: torch.Tensor,
    basis_vec0: torch.Tensor,
    basis_vec1: torch.Tensor,
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor,
    dt: float,
):
    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2
    )

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, targets, torso_position
    )

    dof_pos_scaled = torch_utils.maths.unscale(dof_pos, dof_lower_limits, dof_upper_limits)

    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    prev_potentials[:] = potentials
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    return (
        up_proj,
        heading_proj,
        up_vec,
        heading_vec,
        vel_loc,
        angvel_loc,
        roll,
        pitch,
        yaw,
        angle_to_target,
        dof_pos_scaled,
        prev_potentials,
        potentials,
    )
