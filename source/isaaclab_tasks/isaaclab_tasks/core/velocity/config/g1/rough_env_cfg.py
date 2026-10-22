# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configurations for the Unitree G1 velocity-tracking environment on rough terrain.

This module holds the two stages of the depth pipeline: a teacher whose actor reads a terrain
height scan and the base linear velocity, and a student that replaces both with a chest-mounted
depth camera and is distilled from the teacher in the teacher's own environment.
"""

import math
from dataclasses import MISSING

import torch
from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import CameraCfg, RayCasterCfg, patterns
from isaaclab.sim import PinholeCameraCfg, SimulationCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path
from isaaclab.utils.modifiers import ModifierCfg
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.utils.string import resolve_matching_names, resolve_matching_names_values

from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

from ... import mdp, velocity_env_cfg
from ...velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)
from .terrains import G1_ROUGH_TERRAINS_CFG

DEPTH_WIDTH = 64
DEPTH_HEIGHT = 38
DEPTH_STACK = 3
DEPTH_MAX_RANGE = 3.0
DEPTH_OFFSET_POS = (0.0576, 0.0175, 0.4299)
DEPTH_OFFSET_ROT = (0.9150, 0.0, -0.4035, 0.0)


@configclass
class G1Rewards(RewardsCfg):
    """Reward terms for the G1 teacher and depth student."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=0.0)
    undesired_contacts = None
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.25e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint"])},
    )
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])},
    )
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint"],
            )
        },
    )
    joint_deviation_fingers = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hand_.*_joint"])},
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l2,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
            )
        },
    )
    pelvis_height = RewTerm(
        func=mdp.pelvis_height_deficit_l2,
        weight=-10.0,
        params={
            "target_height": 0.65,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )
    torso_ang_vel = RewTerm(
        func=mdp.body_ang_vel_xy_l2,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"])},
    )
    root_downward_speed = RewTerm(func=mdp.root_downward_speed_l2, weight=-1.0, params={"speed_limit": 0.5})
    feet_support_edge = RewTerm(
        func=mdp.feet_support_edge_distance,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "sole_size": MISSING,
            "sole_offset": MISSING,
        },
    )


def process_depth_image(data: torch.Tensor) -> torch.Tensor:
    """Fill missing depth and rotate the channel-first frames to the G1 camera convention."""
    return torch.nan_to_num(data, nan=DEPTH_MAX_RANGE, posinf=DEPTH_MAX_RANGE).flip((-2, -1))


def get_action_parameters(robot_cfg: ArticulationCfg) -> tuple[dict[str, float], tuple, tuple]:
    """Read authored G1 drives and sole geometry before simulation actuator overrides."""
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(retrieve_file_path(robot_cfg.spawn.usd_path))
    if stage.GetCompositionErrors():
        raise ValueError(f"Could not compose G1 asset: {stage.GetCompositionErrors()}")
    root = stage.GetDefaultPrim()
    joints = {
        prim.GetName(): UsdPhysics.DriveAPI.Get(prim, "angular").GetMaxForceAttr().Get()
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
        if prim.IsA(UsdPhysics.RevoluteJoint)
    }
    action_scale = {}
    for actuator in robot_cfg.actuators.values():
        _, names = resolve_matching_names(actuator.joint_names_expr, list(joints))
        if isinstance(actuator.stiffness, dict):
            _, names, stiffnesses = resolve_matching_names_values(actuator.stiffness, names)
        else:
            stiffnesses = [actuator.stiffness] * len(names)
        for name, stiffness in zip(names, stiffnesses):
            # Use task-specific reference torques for hip roll and finger action scales.
            torque = joints[name]
            if name.endswith("_hip_roll_joint"):
                torque = 88.0
            elif "_hand_" in name:
                torque = 2.45
            # Preserve the policy's four-decimal, half-up action mapping.
            action_scale[name] = math.floor(0.25 * torque * 10000 / stiffness + 0.5) / 10000

    body = root.GetChild("left_ankle_roll_link")
    plate = UsdGeom.Cube(body.GetChild("foot_plate"))
    transform, _ = UsdGeom.XformCache().ComputeRelativeTransform(plate.GetPrim(), body)
    sole_size = tuple(plate.GetSizeAttr().Get() * transform.GetRow3(axis).GetLength() for axis in range(3))
    sole_offset = tuple(transform.ExtractTranslation())
    return action_scale, sole_size, sole_offset


@configclass
class G1EventsCfg(velocity_env_cfg.EventsCfg):
    """G1 startup and reset events."""

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (1 / 1.25, 1.25),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    base_com = None
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )


@configclass
class G1CurriculumCfg(velocity_env_cfg.CurriculumCfg):
    """G1 terrain progression and root-clearance warmup."""

    base_height = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "terminations.base_height.params.minimum_height",
            "modify_fn": mdp.set_value_after_steps,
            "modify_params": {"value": 0.4, "num_steps": 12000},
        },
    )


@configclass
class G1TerminationsCfg(velocity_env_cfg.TerminationsCfg):
    """G1 torso-contact and terrain-clearance terminations."""

    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"), "threshold": 1.0},
    )
    base_height = DoneTerm(
        func=mdp.pelvis_below_terrain_clearance,
        params={
            "minimum_height": -math.inf,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )


@configclass
class G1SceneCfg(velocity_env_cfg.MySceneCfg):
    """G1 robot, mesh terrain and locomotion sensors."""

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1_29dof.usd",
            activate_contact_sensors=True,
            rigid_props=PhysxRigidBodyCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=[
                PhysxArticulationCfg(
                    enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
                ),
                NewtonArticulationCfg(self_collision_enabled=False),
            ],
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.793),
            joint_pos={
                ".*_hip_pitch_joint": -0.20,
                ".*_knee_joint": 0.42,
                ".*_ankle_pitch_joint": -0.23,
                ".*_elbow_joint": 0.87,
                "left_shoulder_pitch_joint": 0.35,
                "left_shoulder_roll_joint": 0.16,
                "right_shoulder_pitch_joint": 0.35,
                "right_shoulder_roll_joint": -0.16,
            },
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_.*", ".*_knee_joint", "waist_yaw_joint"],
                joint_effort_limit=300.0,
                stiffness={
                    ".*_hip_pitch_joint": 200.0,
                    ".*_hip_roll_joint": 150.0,
                    ".*_hip_yaw_joint": 150.0,
                    ".*_knee_joint": 200.0,
                    "waist_yaw_joint": 200.0,
                },
                damping=5.0,
                armature=0.01,
            ),
            "feet": ImplicitActuatorCfg(
                joint_names_expr=[".*_ankle_.*"], joint_effort_limit=20.0, stiffness=20.0, damping=2.0, armature=0.01
            ),
            "waist": ImplicitActuatorCfg(
                joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
                joint_effort_limit=50.0,
                stiffness=200.0,
                damping=5.0,
                armature=0.01,
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[".*_shoulder_.*", ".*_elbow_joint", ".*_wrist_.*"],
                joint_effort_limit=300.0,
                stiffness=40.0,
                damping=10.0,
                armature=0.01,
            ),
            "hands": ImplicitActuatorCfg(
                joint_names_expr=[".*_hand_.*"], joint_effort_limit=300.0, stiffness=40.0, damping=10.0, armature=0.001
            ),
        },
    )

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        global_world_only=True,
    )
    terrain = velocity_env_cfg.MySceneCfg().terrain
    terrain.terrain_generator = G1_ROUGH_TERRAINS_CFG


@configclass
class G1DepthStudentSceneCfg(G1SceneCfg):
    """Teacher scene extended with the student's depth and reflected-depth cameras."""

    @configclass
    class DepthCameraCfg(CameraCfg):
        """Chest-mounted depth camera shared by the student and its reflected view."""

        update_period = 0.0
        update_latest_camera_pose = True
        renderer_cfg = MultiBackendRendererCfg()
        width = DEPTH_WIDTH
        height = DEPTH_HEIGHT
        data_types = ["distance_to_image_plane"]
        spawn = PinholeCameraCfg(
            focal_length=1.93,
            horizontal_aperture=2.0 * 1.93 * math.tan(math.radians(87.0 / 2.0)),
            clipping_range=(0.05, 10.0),
        )
        offset = CameraCfg.OffsetCfg(pos=DEPTH_OFFSET_POS, rot=DEPTH_OFFSET_ROT, convention="world")

    depth_camera = DepthCameraCfg(prim_path="{ENV_REGEX_NS}/Robot/torso_link/depth_camera")
    depth_camera_mirror = DepthCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/depth_camera_mirror",
        offset=CameraCfg.OffsetCfg(
            pos=(DEPTH_OFFSET_POS[0], -DEPTH_OFFSET_POS[1], DEPTH_OFFSET_POS[2]),
            rot=DEPTH_OFFSET_ROT,
            convention="world",
        ),
    )


@configclass
class G1ObservationsCfg:
    """Current proprioception and ray-cast terrain heights."""

    @configclass
    class ProprioCfg(velocity_env_cfg.ObservationsCfg.PolicyCfg):
        """Current robot state and velocity commands."""

        height_scan = None

    @configclass
    class HeightScanCfg(ObsGroup):
        """Ray-cast terrain heights relative to the robot."""

        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )
        enable_corruption = True
        concatenate_terms = True

    proprio: ProprioCfg = ProprioCfg()
    height_scan: HeightScanCfg = HeightScanCfg()


@configclass
class G1StudentObservationsCfg(G1ObservationsCfg):
    """Proprioception history and depth inputs, with noise-free state/height supervision."""

    @configclass
    class ProprioHistoryCfg(G1ObservationsCfg.ProprioCfg):
        """Five proprioceptive frames without privileged velocity or height observations."""

        base_lin_vel = None
        history_length = 5

    @configclass
    class DepthCfg(ObsGroup):
        """Normalized depth-frame stack in the calibrated G1 camera convention."""

        depth = ObsTerm(
            func=mdp.image_depth,
            params={
                "sensor_cfg": SceneEntityCfg("depth_camera"),
                "data_type": "distance_to_image_plane",
                "normalize": False,
                "frame_stack": DEPTH_STACK,
                "channel_first": True,
            },
            modifiers=[ModifierCfg(func=process_depth_image)],
            clip=(0.0, DEPTH_MAX_RANGE),
            scale=1.0 / DEPTH_MAX_RANGE,
        )
        enable_corruption = False
        concatenate_terms = True

    height_scan: G1ObservationsCfg.HeightScanCfg = G1ObservationsCfg.HeightScanCfg(enable_corruption=False)
    proprio_history: ProprioHistoryCfg = ProprioHistoryCfg()
    depth: DepthCfg = DepthCfg()
    depth_mirror: DepthCfg = DepthCfg()

    def __post_init__(self):
        super().__post_init__()
        self.proprio.enable_corruption = False
        self.depth_mirror.depth.params["sensor_cfg"] = SceneEntityCfg("depth_camera_mirror")


@configclass
class G1RoughTeacherEnvCfg(LocomotionVelocityRoughEnvCfg):
    """G1 teacher composed from its scene and manager configurations."""

    seed = 42
    scene: G1SceneCfg = G1SceneCfg(num_envs=4096, env_spacing=2.5)
    rewards: G1Rewards = G1Rewards()
    events: G1EventsCfg = G1EventsCfg()
    curriculum: G1CurriculumCfg = G1CurriculumCfg()
    terminations: G1TerminationsCfg = G1TerminationsCfg()
    observations: G1ObservationsCfg = G1ObservationsCfg()
    sim = SimulationCfg(physics=velocity_env_cfg.RoughPhysicsCfg())
    sim.physics.newton_mjwarp.solver_cfg.njmax = 300
    commands = velocity_env_cfg.CommandsCfg()
    commands.base_velocity.vel_yaw_success_threshold = 0.8
    commands.base_velocity.marker_pos_offset = (0.0, 0.0, 0.75)
    commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
    commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
    commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.scale, sole_size, sole_offset = get_action_parameters(self.scene.robot)
        self.rewards.feet_support_edge.params.update(sole_size=sole_size, sole_offset=sole_offset)


@configclass
class G1RoughDepthStudentEnvCfg(G1RoughTeacherEnvCfg):
    """G1 student with five proprioceptive frames and mirrored depth supervision."""

    scene: G1DepthStudentSceneCfg = G1DepthStudentSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: G1StudentObservationsCfg = G1StudentObservationsCfg()
