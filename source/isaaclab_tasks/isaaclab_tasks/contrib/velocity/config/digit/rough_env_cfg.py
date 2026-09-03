# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg
from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.physics import PhysxAutoCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import isaaclab_tasks.core.velocity.mdp as mdp
from isaaclab_tasks.core.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from isaaclab_tasks.utils import PresetCfg, preset

from isaaclab_assets.robots.agility import ARM_JOINT_NAMES, DIGIT_V4_CFG, LEG_JOINT_NAMES

from . import _strip_visual_colliders

_strip_visual_colliders.install()


# The stability bound is c*h/2 = 0.0716 for Digit's damping and substep; this leaves margin.
_MIN_STABLE_ARMATURE = 0.10


def _raise_low_armature(env, env_ids, floor: float):
    """Clamp joint armature up to ``floor`` wherever the actuator damping would be unstable.

    ``wrist_yaw`` ships at I = 0.01822 [kg m^2] against c = 57.3 and h = 0.0025 s, giving
    c*h/I = 7.86 and a measured 6.9x growth per substep; eight more joints sit at 2.74.

    Args:
        env: The environment instance.
        env_ids: Unused; the clamp applies to every environment at startup.
        floor: Minimum armature [kg m^2] for any joint with non-zero damping.
    """
    import torch

    asset = env.scene["robot"]
    armature = asset.data.joint_armature.torch.clone()
    needs = (asset.data.joint_damping.torch > 0.0) & (armature < floor)
    if not bool(needs.any()):
        return
    asset.write_joint_armature_to_sim_index(armature=torch.where(needs, torch.full_like(armature, floor), armature))


@configclass
class DigitPhysicsCfg(PresetCfg):
    """Physics configuration for the Digit velocity environments."""

    isaacsim_physx = PhysxCfg(
        gpu_max_rigid_patch_count=10 * 2**15,
        gpu_found_lost_pairs_capacity=2**23,
        gpu_total_aggregate_pairs_capacity=2**23,
    )
    physx = PhysxAutoCfg(isaacsim_physx=isaacsim_physx)
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=5000,
            nconmax=2000,
            cone="pyramidal",
            impratio=1.0,
            integrator="implicitfast",
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000),
        num_substeps=2,
        default_shape_cfg=NewtonShapeCfg(margin=0.0, ke=160000.0, kd=1100.0),
    )
    default = isaacsim_physx


@configclass
class DigitRewards:
    termination_penalty = RewardTermCfg(
        func=mdp.is_terminated,
        weight=-100.0,
    )
    track_lin_vel_xy_exp = RewardTermCfg(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewardTermCfg(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
        },
    )
    feet_air_time = RewardTermCfg(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_leg_toe_roll"),
            "threshold": 0.8,
            "command_name": "base_velocity",
        },
    )
    feet_slide = RewardTermCfg(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_leg_toe_roll"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_leg_toe_roll"),
        },
    )
    dof_torques_l2 = RewardTermCfg(
        func=mdp.joint_torques_l2,
        weight=-1.0e-6,
    )
    dof_acc_l2 = RewardTermCfg(
        func=mdp.joint_acc_l2,
        weight=-2.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES + ARM_JOINT_NAMES)},
    )
    action_rate_l2 = RewardTermCfg(
        func=mdp.action_rate_l2,
        weight=-0.008,
    )
    flat_orientation_l2 = RewardTermCfg(
        func=mdp.flat_orientation_l2,
        weight=-2.5,
    )
    stand_still = RewardTermCfg(
        func=mdp.stand_still_joint_deviation_l1,
        weight=-0.4,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES),
        },
    )
    lin_vel_z_l2 = RewardTermCfg(
        func=mdp.lin_vel_z_l2,
        weight=-2.0,
    )
    ang_vel_xy_l2 = RewardTermCfg(
        func=mdp.ang_vel_xy_l2,
        weight=-0.1,
    )
    no_jumps = RewardTermCfg(
        func=mdp.desired_contacts,
        weight=-0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_leg_toe_roll"])},
    )
    dof_pos_limits = RewardTermCfg(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_leg_toe_roll", ".*_leg_toe_pitch", ".*_tarsus"])},
    )
    joint_deviation_hip_roll = RewardTermCfg(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_leg_hip_roll")},
    )
    joint_deviation_hip_yaw = RewardTermCfg(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_leg_hip_yaw")},
    )
    joint_deviation_knee = RewardTermCfg(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_tarsus")},
    )
    joint_deviation_feet = RewardTermCfg(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_toe_a", ".*_toe_b"])},
    )
    joint_deviation_arms = RewardTermCfg(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_arm_.*"),
        },
    )

    undesired_contacts = RewardTermCfg(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_rod", ".*_tarsus"]),
            "threshold": 1.0,
        },
    )


@configclass
class DigitObservations:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        base_lin_vel = ObservationTermCfg(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        base_ang_vel = ObservationTermCfg(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        projected_gravity = ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES + ARM_JOINT_NAMES)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES + ARM_JOINT_NAMES)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        actions = ObservationTermCfg(func=mdp.last_action)
        height_scan = ObservationTermCfg(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # Observation groups:
    policy: PolicyCfg = PolicyCfg()


@configclass
class DigitTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    base_contact = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["torso_base"]),
            "threshold": 1.0,
        },
    )
    base_orientation = TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.7},
    )


@configclass
class DigitActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES + ARM_JOINT_NAMES,
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class DigitRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    sim: SimulationCfg = SimulationCfg(physics=DigitPhysicsCfg())
    rewards: DigitRewards = DigitRewards()
    observations: DigitObservations = DigitObservations()
    terminations: DigitTerminationsCfg = DigitTerminationsCfg()
    actions: DigitActionsCfg = DigitActionsCfg()

    def __post_init__(self):
        super().__post_init__()

        # scene
        self.scene.robot = DIGIT_V4_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_base"
        self.scene.contact_forces.history_length = self.decimation
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        # target only actuated joints explicitly — ".*" mis-indexes Digit's ball-joint DoFs
        self.scene.robot.actuators = {
            "legs_arms": ImplicitActuatorCfg(
                joint_names_expr=LEG_JOINT_NAMES + ARM_JOINT_NAMES,
                stiffness=None,
                damping=None,
            ),
        }
        # commands
        self.commands.base_velocity.ranges.lin_vel_x = (-0.8, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.rel_standing_envs = 0.1
        self.commands.base_velocity.resampling_time_range = (3.0, 8.0)
        # events
        self.events.add_base_mass.params["asset_cfg"].body_names = "torso_base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "torso_base"
        # base_com carries a single preset branch; collapse it so the body name can be set.
        self.events.base_com = self.events.base_com.default
        self.events.base_com.params["asset_cfg"].body_names = "torso_base"
        # MJWarp integrates actuator damping explicitly, so a joint runs away once c*h/I > 2.
        # digit_v4.usd ships ten joints between 2.74 and 7.86; clamp those up at startup. A
        # scalar ``ImplicitActuatorCfg.armature`` cannot express this -- it would also pull the
        # joints that are already fine down to the floor.
        self.events.raise_low_armature = preset(
            default=None,
            newton_mjwarp=EventTermCfg(
                func=_raise_low_armature, mode="startup", params={"floor": _MIN_STABLE_ARMATURE}
            ),
        )
        # The asset authors no articulation_props, so Newton filters every intra-articulation
        # shape pair -- 253 of them, exactly C(23,2) for its 23 colliding shapes -- and the legs
        # pass through each other. Which links carry colliders is left as the asset authored it.
        self.scene.robot.spawn.articulation_props = preset(
            default=None, newton_mjwarp=[NewtonArticulationCfg(self_collision_enabled=True)]
        )
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
