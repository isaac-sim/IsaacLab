from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from collections.abc import Sequence


import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import RigidObjectCfg, RigidObject, Articulation, DeformableObject
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import quat_from_euler_xyz, euler_xyz_from_quat
from .deformabe_cube_env_cfg import DeformableCubeEnvCfg

class DeformableCubeEnv(DirectRLEnv):
	cfg: DeformableCubeEnvCfg

	def __init__(self, cfg: DeformableCubeEnvCfg, render_mode: str | None = None, **kwargs):
		super().__init__(cfg, render_mode, **kwargs)
		pass

	def _setup_scene(self):
		self.robot = Articulation(self.cfg.robot_cfg)
		self.object = DeformableObject(self.cfg.object_cfg)
		# TODO: spawn container

		spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

		# clone, filter, and replicate
		self.scene.clone_environments(copy_from_source=False)
		self.scene.filter_collisions(global_prim_paths=[])

		# add lights
		light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
		light_cfg.func("/World/Light", light_cfg)

	def _pre_physics_step(self, actions: torch.Tensor) -> None:
		pass

	def _apply_action(self) -> None:
		pass

	def _get_observations(self) -> dict:
		pass

	def _get_rewards(self) -> torch.Tensor:
		pass

	def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
		pass

	def _reset_idx(self, env_ids: Sequence[int] | None):
		pass