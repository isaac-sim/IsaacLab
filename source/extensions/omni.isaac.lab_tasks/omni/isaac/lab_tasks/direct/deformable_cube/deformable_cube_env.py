from __future__ import annotations
import math
import torch
from collections.abc import Sequence

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.markers import VisualizationMarkersCfg, VisualizationMarkers
from omni.isaac.lab.assets import Articulation, DeformableObject, ArticulationCfg, DeformableObjectCfg, RigidObjectCfg, RigidObject
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane 
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import subtract_frame_transforms

from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

CUBE_SIZE = 0.1
SPAWN_X_BOUNDS = (0.05, 0.90)
SPAWN_Y_BOUNDS = (-0.42, 0.42)
SPAWN_Z_LEVEL = 1.0

# TODO: Edit GPU settings for softbody contact buffer size
# TODO: mess with deformable object settings
# TODO: Add some way to detect if cube is in container

@configclass
class DeformableCubeEnvCfg(DirectRLEnvCfg):
	num_envs = 4
	env_spacing = 3.0
	dt = 1 / 120
	observation_space = 15
	action_space = 9
	state_space = 0
	action_scale = 7.5

	# env
	decimation = 1
	episode_length_s = 5.0

	# simulation
	sim: SimulationCfg = SimulationCfg(dt=dt, render_interval=decimation)
	scene: InteractiveSceneCfg = InteractiveSceneCfg(
		num_envs=num_envs, 
		env_spacing=env_spacing, 
		replicate_physics=False,
	)

	# entities
	table_cfg: RigidObjectCfg = RigidObjectCfg(
		prim_path="/World/envs/env_.*/Table",
		spawn=sim_utils.UsdFileCfg(
			usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
			rigid_props=sim_utils.RigidBodyPropertiesCfg(
				disable_gravity=False,
			),
			collision_props=sim_utils.CollisionPropertiesCfg(
				collision_enabled=True,
			),
		),
		init_state=RigidObjectCfg.InitialStateCfg(
			pos=(0.5, 0.0, 1.0),
			rot=(0.707, 0.0, 0.0, 0.707)
		),
	)

	robot_cfg: ArticulationCfg = ArticulationCfg(
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
			pos=(-0.05, 0.0, 1.0),
			rot=(1.0, 0.0, 0.0, 0.0)
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

	object_cfg: DeformableObjectCfg = DeformableObjectCfg(
		prim_path="/World/envs/env_.*/Cube",
		spawn=sim_utils.MeshCuboidCfg(
			size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
			deformable_props=sim_utils.DeformableBodyPropertiesCfg(
				rest_offset=0.0,
				contact_offset=0.001
			),
			visual_material=sim_utils.PreviewSurfaceCfg(
				diffuse_color=(0.5, 0.1, 0.0)
			),
			physics_material=sim_utils.DeformableBodyMaterialCfg(
				poissons_ratio=0.4,
				youngs_modulus=1e5
			),
		),
		init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, SPAWN_Z_LEVEL + CUBE_SIZE / 1.9)),
		debug_vis=True,
	)

	container_cfg: RigidObjectCfg = RigidObjectCfg(
		prim_path="/World/envs/env_.*/Container",
		spawn=sim_utils.UsdFileCfg(
			usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/KLT_Bin/small_KLT_visual_collision.usd",
			rigid_props=sim_utils.RigidBodyPropertiesCfg(
				disable_gravity=False,
			),
			mass_props=sim_utils.MassPropertiesCfg(density=5000.0),
			collision_props=sim_utils.CollisionPropertiesCfg(
				collision_enabled=True,
			),
			scale=(2.4, 2.5, 1.2),
		),
		init_state=RigidObjectCfg.InitialStateCfg(
			pos=(0.5, 0.75, 1.02),
			rot=(math.sqrt(2)/2, 0.0, 0.0, math.sqrt(2)/2)
		),
	)


	# markers
	table_markers_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
		prim_path="/Visuals/TableMarkers",
		markers={
			"table_bottom_left": sim_utils.UsdFileCfg(
				usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
				scale=(0.15, 0.15, 0.15),
			),
			"table_bottom_right": sim_utils.UsdFileCfg(
				usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
				scale=(0.15, 0.15, 0.15),
			),
			"table_top_left": sim_utils.UsdFileCfg(
				usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
				scale=(0.15, 0.15, 0.15),
			),
			"table_top_right": sim_utils.UsdFileCfg(
				usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
				scale=(0.15, 0.15, 0.15),
			),
		}
	)


# draws frame markers to bound the region where objects can spawn
def draw_spawn_bounds(
		markers: VisualizationMarkers,
		origins: torch.Tensor, 
		x_bounds: tuple[float, float], 
		y_bounds: tuple[float, float],
		z_val: float
):
	N_envs = origins.shape[0]

	# spawn corner markers
	bottom_left_T = torch.tensor([x_bounds[0], y_bounds[1], z_val], device=origins.device).reshape(1, -1)
	bottom_right_T = torch.tensor([x_bounds[0], y_bounds[0], z_val], device=origins.device).reshape(1, -1)
	top_left_T = torch.tensor([x_bounds[1], y_bounds[1], z_val], device=origins.device).reshape(1, -1)
	top_right_T = torch.tensor([x_bounds[1], y_bounds[0], z_val], device=origins.device).reshape(1, -1)

	bottom_left, bottom_right, top_left, top_right = origins.clone().repeat(4, 1).reshape(4, -1, 3).unbind(0)
	bottom_left += bottom_left_T
	bottom_right += bottom_right_T
	top_left += top_left_T
	top_right += top_right_T

	r2b2 = math.sqrt(2) / 2
	bottom_left_quat = torch.tensor([-r2b2, 0.0, 0.0, r2b2], device=origins.device).reshape(1, -1).repeat(N_envs, 1)
	bottom_right_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=origins.device).reshape(1, -1).repeat(N_envs, 1)
	top_left_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=origins.device).reshape(1, -1).repeat(N_envs, 1)
	top_right_quat = torch.tensor([r2b2, 0.0, 0.0, r2b2], device=origins.device).reshape(1, -1).repeat(N_envs, 1)

	marker_quats = torch.cat([bottom_left_quat, bottom_right_quat, top_left_quat, top_right_quat], dim=0)
	marker_locs = torch.cat([bottom_left, bottom_right, top_left, top_right], dim=0)
	marker_idxs = torch.tensor([0, 1, 2, 3]).repeat_interleave(N_envs)
	markers.visualize(marker_locs, marker_quats, marker_indices=marker_idxs)


class DeformableCubeEnv(DirectRLEnv):
	cfg: DeformableCubeEnvCfg

	def __init__(self, cfg: DeformableCubeEnvCfg, render_mode: str | None = None, **kwargs):
		super().__init__(cfg, render_mode, **kwargs)
		self.robot_dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
		self.robot_dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

		self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
		self.robot_dof_speed_scales[self.robot.find_joints("panda_finger_joint1")[0]] = 0.1
		self.robot_dof_speed_scales[self.robot.find_joints("panda_finger_joint2")[0]] = 0.1
		self.robot_dof_targets = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)

	def _setup_scene(self):
		self.robot = Articulation(self.cfg.robot_cfg)
		self.scene.articulations["robot"] = self.robot

		self.object = DeformableObject(self.cfg.object_cfg)
		self.container = RigidObject(self.cfg.container_cfg)

		self.table = RigidObject(self.cfg.table_cfg)
		self.table_markers = VisualizationMarkers(self.cfg.table_markers_cfg)
		draw_spawn_bounds(
			self.table_markers, self.scene.env_origins, 
			SPAWN_X_BOUNDS, SPAWN_Y_BOUNDS, SPAWN_Z_LEVEL
		)

		spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

		# clone, filter, and replicate
		self.scene.clone_environments(copy_from_source=False)
		self.scene.filter_collisions(global_prim_paths=[])

		# add lights
		light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
		light_cfg.func("/World/Light", light_cfg)

	def _pre_physics_step(self, actions: torch.Tensor) -> None:
		self.actions = actions.clone().clamp(-1.0, 1.0)
		targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.step_dt * self.actions * self.cfg.action_scale
		self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

	def _apply_action(self) -> None:
		self.robot.set_joint_position_target(self.robot_dof_targets)

	def _get_observations(self) -> dict:
		joint_pos_rel = (self.robot.data.joint_pos - self.robot.data.default_joint_pos).clone()
		joint_vel_rel = (self.robot.data.joint_vel - self.robot.data.default_joint_vel).clone()
		object_pos, object_quat = subtract_frame_transforms(
			self.robot.data.root_link_state_w[:, :3],
			self.robot.data.root_link_state_w[:, 3:7],
			self.object.data.root_pos_w
		)
		container_pos = self.container.data.root_link_pos_w.clone()
		last_action = self.actions.clone()
		print(joint_pos_rel.shape, joint_vel_rel.shape, object_pos.shape, object_quat.shape, container_pos.shape, last_action.shape)

	def _get_rewards(self) -> torch.Tensor:
		pass

	def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
		self.object.update(self.step_dt)
		out_of_bounds = self.object.data.root_pos_w[:, -1] < 0.5 # if object falls
		time_out = self.episode_length_buf >= self.max_episode_length - 1
		return out_of_bounds, time_out

	def _reset_idx(self, env_ids: Sequence[int] | None):
		print(f"[INFO]: Resetting envs {env_ids}")
		self.episode_length_buf[env_ids] = 0.0

		# reset objects
		x_offset, y_offset = torch.rand(2, len(env_ids), 1, device=self.device)
		x_offset = x_offset * (SPAWN_X_BOUNDS[1] - SPAWN_X_BOUNDS[0]) + SPAWN_X_BOUNDS[0]
		y_offset = y_offset * (SPAWN_Y_BOUNDS[1] - SPAWN_Y_BOUNDS[0]) + SPAWN_Y_BOUNDS[0]
		pos_t = torch.cat([x_offset, y_offset, torch.zeros_like(x_offset)], dim=-1)

		quat_t = torch.zeros(len(env_ids), 4, device=self.device)
		quat_t[:, 0], quat_t[:, -1] = torch.rand(2, len(env_ids), device=self.device) * 2 - 1.0
		quat_t = quat_t / quat_t.norm(dim=1, keepdim=True)

		nodal_state = self.object.data.default_nodal_state_w[env_ids, :, :]
		nodal_state[..., :3] = self.object.transform_nodal_pos(nodal_state[..., :3], pos=pos_t, quat=quat_t)
		nodal_state[..., 3:] = 0.0
		self.object.write_nodal_state_to_sim(nodal_state, env_ids)

		# reset robots
		joint_pos = self.robot.data.default_joint_pos[env_ids]
		joint_vel = torch.zeros_like(joint_pos)
		self.robot.write_joint_state_to_sim(position=joint_pos, velocity=joint_vel, env_ids=env_ids)