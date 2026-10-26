# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ANYmal C applied-effort dynamics and locomotion tasks."""

import math

import torch
from isaaclab_newton.envs.mdp.recorders import NeRDRecorderCfg, NeRDRecorderManagerCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg
from isaaclab_newton.physics.nerd_manager_cfg import NeRDSolverCfg

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import ManagerTermBase, RewardTermCfg, TerminationTermCfg
from isaaclab.terrains import FlatPatchSamplingCfg, TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass

from isaaclab_assets.robots.anymal import ANYMAL_C_CFG

from ..anymal_d.flat_env_cfg import AnymalDFlatEnvCfg


class control_period_joint_acc_l2(ManagerTermBase):
    """Difference velocities over the control period in both numerical and learned physics."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        self.previous = self.robot.data.joint_vel.torch.clone()
        self.fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        self.fresh[slice(None) if env_ids is None else env_ids] = True

    def __call__(self, env):
        velocity = self.robot.data.joint_vel.torch
        penalty = torch.where(self.fresh, 0.0, ((velocity - self.previous) / env.step_dt).square().sum(dim=1))
        self.previous.copy_(velocity)
        self.fresh.zero_()
        return penalty


@configclass
class AnymalCNeRDEnvCfg(AnymalDFlatEnvCfg):
    """Flat-ground teacher with 5 ms ANYdrive updates and a 20 ms policy control period.

    The policy objective uses state-derived quantities that the learned solver also produces.
    Fixed probe geometry requires one plane per world. Rough terrain needs another feature
    contract and is not represented by these flat-ground probes.
    """

    recorders: NeRDRecorderManagerCfg = NeRDRecorderManagerCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.sim.physics = NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=60, nconmax=25, cone="elliptic", impratio=100.0, integrator="implicitfast"
            ),
            num_substeps=1,
            use_cuda_graph=True,
            default_shape_cfg=NewtonShapeCfg(margin=0.0, ke=160000.0, kd=1100.0),
        )
        material = self.scene.terrain.physics_material
        self.scene.terrain = None
        self.scene.ground = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Ground",
            spawn=sim_utils.GroundPlaneCfg(
                size=(self.scene.env_spacing, self.scene.env_spacing), physics_material=material
            ),
        )
        self.scene.contact_forces = None
        self.rewards.feet_air_time = self.rewards.undesired_contacts = self.rewards.dof_torques_l2 = None
        self.rewards.dof_acc_l2 = RewardTermCfg(func=control_period_joint_acc_l2, weight=-1.2e-6)
        self.terminations.base_contact = None
        self.terminations.bad_orientation = TerminationTermCfg(
            func=mdp.bad_orientation, params={"limit_angle": math.pi / 2}
        )
        self.events.physics_material = self.events.add_base_mass = self.events.base_com = None
        self.events.base_external_force_torque = self.events.push_robot = None


@configclass
class AnymalCNeRDExplorationEnvCfg(AnymalCNeRDEnvCfg):
    """Random poses and targets, retaining falls for task-independent dynamics collection."""

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 1
        self.episode_length_s = 2.0
        self.terminations.bad_orientation = None
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.1, 0.1),
                "roll": (-0.3, 0.3),
                "pitch": (-0.3, 0.3),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (-0.5, 0.5),
                "roll": (-1.0, 1.0),
                "pitch": (-1.0, 1.0),
                "yaw": (-1.0, 1.0),
            },
        }
        self.events.reset_robot_joints.func = mdp.reset_joints_by_offset
        self.events.reset_robot_joints.params = {"position_range": (-0.5, 0.5), "velocity_range": (-1.0, 1.0)}
        self.recorders.transition = NeRDRecorderCfg(
            heading=True,
            contact_bodies=("LF_FOOT", "LH_FOOT", "RF_FOOT", "RH_FOOT"),
            root_body="base",
        )


@configclass
class AnymalCNeRDLearnedEnvCfg(AnymalCNeRDEnvCfg):
    """Keep ANYdrive and run four 5 ms learned physics steps per 20 ms policy action."""

    def __post_init__(self):
        super().__post_init__()
        self.sim.physics = self.sim.physics.replace(solver_cfg=NeRDSolverCfg())


@configclass
class AnymalCNeRDRoughExplorationEnvCfg(AnymalCNeRDExplorationEnvCfg):
    """Physics-step collection across rough surfaces, conditioned on all Newton rigid contacts."""

    def __post_init__(self):
        super().__post_init__()
        material = self.scene.ground.spawn.physics_material
        self.scene.ground = None
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            max_init_terrain_level=None,
            terrain_generator=ROUGH_TERRAINS_CFG.replace(
                seed=42, num_rows=4, num_cols=6, border_width=2.0, curriculum=False
            ),
            physics_material=material,
        )
        self.sim.physics.solver_cfg.use_mujoco_contacts = False
        self.sim.physics.solver_cfg.nconmax = 64
        self.sim.physics.solver_cfg.njmax = 256
        for terrain in self.scene.terrain.terrain_generator.sub_terrains.values():
            terrain.flat_patch_sampling = {
                "init_pos": FlatPatchSamplingCfg(
                    num_patches=64, patch_radius=0.5, max_height_diff=0.12, x_range=(-3.0, 3.0), y_range=(-3.0, 3.0)
                )
            }
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.8)
        self.events.reset_base.func = mdp.reset_root_state_from_terrain
        self.events.reset_base.params["pose_range"] = {
            "roll": (-0.3, 0.3),
            "pitch": (-0.3, 0.3),
            "yaw": (-math.pi, math.pi),
        }
        self.recorders.transition.contact_bodies = ()
        self.recorders.transition.max_contacts = 64


@configclass
class AnymalCNeRDRoughEnvCfg(AnymalCNeRDRoughExplorationEnvCfg):
    """Rough-terrain locomotion with native ANYdrive and proprioceptive policy observations."""

    def __post_init__(self):
        super().__post_init__()
        self.recorders.transition = None
        self.decimation = self.sim.render_interval = 4
        self.episode_length_s = 20.0
        self.terminations.bad_orientation = TerminationTermCfg(
            func=mdp.bad_orientation, params={"limit_angle": math.pi / 2}
        )


@configclass
class AnymalCNeRDRoughLearnedEnvCfg(AnymalCNeRDRoughEnvCfg):
    """Replace the numerical rough-terrain solver while retaining collisions and ANYdrive."""

    def __post_init__(self):
        super().__post_init__()
        self.sim.physics = self.sim.physics.replace(solver_cfg=NeRDSolverCfg())


@configclass
class AnymalCRoughNeRDTrainCfg:
    """Contact-conditioned offline fitting; optimizer defaults belong to the shared learner."""

    max_updates = 60000
    learning_rate_decay_updates = 60000
    batch_size = 4096
    precision = "bfloat16"
    network = {
        "num_heads": 10,
        "head_dim": 64,
        "intermediate_size": 896,
        "head_hidden": 192,
        "temporal_attention_spans": (4, 4, 4, 4),
        "rms_norm_eps": 1.0e-6,
        "linear_probe": True,
        "contact_slot_count": 8,
        "contact_slot_dim": 64,
    }
