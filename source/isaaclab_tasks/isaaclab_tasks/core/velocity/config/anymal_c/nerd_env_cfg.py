# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ANYmal C target-controlled dynamics and locomotion tasks."""

import math

import torch
from isaaclab_newton.envs.mdp.recorders import NeRDRecorderCfg, NeRDRecorderManagerCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg
from isaaclab_newton.physics.nerd_manager_cfg import NeRDSolverCfg

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import ManagerTermBase, RewardTermCfg, TerminationTermCfg
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
            control="joint_target_q",
            heading=True,
            contact_bodies=("LF_FOOT", "LH_FOOT", "RF_FOOT", "RH_FOOT"),
            root_body="base",
        )


@configclass
class AnymalCNeRDLearnedEnvCfg(AnymalCNeRDEnvCfg):
    """Target-conditioned NeRD replaces the complete 20 ms numerical/actuator update."""

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = 0.02
        self.decimation = self.sim.render_interval = 1
        self.sim.physics = self.sim.physics.replace(solver_cfg=NeRDSolverCfg())
        self.scene.robot.actuators = {"legs": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0)}
