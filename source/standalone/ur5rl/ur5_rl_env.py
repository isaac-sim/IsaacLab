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
class HawUr5Cfg(DirectRLEnvCfg):
    # env
    num_actions = 3
    num_observations = 4
    num_states = 5
    reward_scale_example = 1.0
    decimation = 2
    action_scale = 1.0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

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
    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=10, env_spacing=4.0, replicate_physics=True
    )


class HawUr5Env(DirectRLEnv):
    cfg: HawUr5Cfg

    def __init__(self, cfg: HawUr5Cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.action_scale = self.cfg.action_scale

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
        self.actions = self.cfg.action_scale * actions.clone()

    def _apply_action(self) -> None:
        # self.cartpole.set_joint_effort_target(self.actions, joint_ids=self._cart_dof_idx)
        pass

    def _get_observations(self) -> dict:
        obs = torch.ones(1, device=self.device)
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
        super()._reset_idx(env_ids)
