# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    StudentObservationsCfg,
    TeacherStudentObservationsCfg,
)

from .rough_env_cfg import G1_29_DOFs_RoughEnvCfg


@configclass
class G1_29_DOFs_FlatEnvCfg(G1_29_DOFs_RoughEnvCfg):
    sim: SimulationCfg = SimulationCfg(
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=210,
                nconmax=35,
                ls_iterations=10,
                ls_parallel=True,
                cone="pyramidal",
                impratio=1,
                integrator="implicitfast",
            ),
            num_substeps=1,
            debug_mode=False,
        )
    )

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        # self.scene.height_scanner = None
        # self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # Rewards
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.feet_air_time.weight = 0.75
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )
        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


class G1_29_DOFs_FlatEnvCfg_PLAY(G1_29_DOFs_FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1_29_DOFs_FlatTeacherStudentEnvCfg(G1_29_DOFs_FlatEnvCfg):
    observations: TeacherStudentObservationsCfg = TeacherStudentObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 256
        self.observations.policy.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.observed_joint_names
        )
        self.observations.policy.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.observed_joint_names
        )
        self.observations.teacher.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.observed_joint_names
        )
        self.observations.teacher.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.observed_joint_names
        )
        # reduce the teacher observation noise during distillation
        self.observations.teacher.base_lin_vel.noise = Unoise(n_min=-0.001, n_max=0.001)
        self.observations.teacher.base_ang_vel.noise = Unoise(n_min=-0.002, n_max=0.002)
        self.observations.teacher.projected_gravity.noise = Unoise(n_min=-0.0005, n_max=0.0005)
        self.observations.teacher.joint_pos.noise = Unoise(n_min=-0.0001, n_max=0.0001)
        self.observations.teacher.joint_vel.noise = Unoise(n_min=-0.0001, n_max=0.0001)


@configclass
class G1_29_DOFs_FlatStudentEnvCfg(G1_29_DOFs_FlatEnvCfg):
    observations: StudentObservationsCfg = StudentObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.observations.policy.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.observed_joint_names
        )
        self.observations.policy.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.observed_joint_names
        )
