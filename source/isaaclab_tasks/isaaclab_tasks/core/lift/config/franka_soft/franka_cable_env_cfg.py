# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka cable lifting environment."""

from __future__ import annotations

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg, VBDSolverCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, CableObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.renderers import isaac_rtx_per_env_scene_partition_enabled

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg

from isaaclab_tasks.utils import PresetCfg

from ... import mdp
from .franka_soft_env_cfg import (
    FRANKA_CAMERA_CFG,
    TABLE_SPAWN_CFG,
    FrankaCameraObservationsCfg,
    FrankaSoftEnvCfg,
    _FrankaSoftSceneCfg,
)
from .franka_soft_env_cfg import EventCfg as FrankaSoftEventCfg

_CABLE_SEGMENT_COUNT = 12
_CABLE_MIDDLE_SEGMENT_INDEX = _CABLE_SEGMENT_COUNT // 2

# Diagonally opposite corners of a box enveloping everything the cable can reach, in env-local
# coordinates [m]. Derived from the ``cable_out_of_bounds`` termination box below, padded to
# absorb the segment overshoot of the step that trips the termination.
_PARTITION_BOUNDS_MIN = (-0.25, -0.75, -0.25)
_PARTITION_BOUNDS_MAX = (1.25, 0.75, 1.25)

# Millimetre-scale marker cubes pinning those corners. See the "Animated curves disappear under
# Isaac RTX scene partitioning" entry in ``docs/source/refs/issues.rst``.
_PARTITION_BOUNDS_MARKER_SPAWN_CFG = sim_utils.CuboidCfg(
    size=(0.001, 0.001, 0.001),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),
)


@configclass
class PhysicsCfg(PresetCfg):
    """Newton proxy physics for rigid-cable coupling."""

    newton_mjwarp_vbd_proxy: NewtonCfg = NewtonCfg(
        solver_cfg=CouplerProxyCfg(
            entries=[
                CouplerEntryCfg(
                    name="rigid",
                    solver_cfg=MJWarpSolverCfg(
                        cone="elliptic",
                        ls_iterations=20,
                        integrator="implicitfast",
                    ),
                    bodies=[r"/World/envs/env_.*/Robot"],
                ),
                CouplerEntryCfg(
                    name="cable",
                    solver_cfg=VBDSolverCfg(iterations=10),
                    bodies=[r"/World/envs/env_.*/Cable"],
                    include_static_shapes=True,
                ),
            ],
            proxies=[
                CouplerProxyMappingCfg(
                    source="rigid",
                    destination="cable",
                    bodies=[
                        r"/World/envs/env_.*/Robot/Geometry/.*panda_hand",
                        r"/World/envs/env_.*/Robot/Geometry/.*panda_(left|right)finger",
                    ],
                    collide_interval=1,
                )
            ],
            iterations=1,
        ),
        default_shape_cfg=NewtonShapeCfg(
            ke=2.5e3,
            kd=100.0,
            mu=10.0,
        ),
        num_substeps=4,
    )

    default = newton_mjwarp_vbd_proxy


@configclass
class FrankaCableSceneCfg(_FrankaSoftSceneCfg):
    """Scene for the Franka cable lifting environment."""

    deformable: None = None

    table: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, -0.525]),
        spawn=TABLE_SPAWN_CFG.replace(
            physics_material=RigidBodyMaterialBaseCfg(static_friction=0.01, dynamic_friction=0.01),
        ),
    )

    cable: CableObjectCfg = CableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cable",
        spawn=sim_utils.CableCfg(
            positions=[(0.03 * index, 0.0, 0.0) for index in range(_CABLE_SEGMENT_COUNT + 1)],
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.85)),
            physics_material=sim_utils.CableMaterialCfg(
                thickness=0.01,
                density=1000.0,
                stretch_stiffness=1.0e6,
                bend_stiffness=1.0e5,
            ),
            collision_props=[sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)],
        ),
        init_state=CableObjectCfg.InitialStateCfg(pos=(0.32, 0.0, 0.011)),
    )

    # Workaround for OMPE-105749: Kit RTX never refreshes the bounding box of an animated
    # ``BasisCurves`` prim, so a partition sized from the cable's initial extent clips the cable
    # once it moves outside that extent. These two static cubes pin the partition to the full
    # workspace volume. FrankaCableEnvCfg drops them when partitioning is off, and they can go
    # altogether once Kit updates animated-curve bounding boxes.
    partition_bounds_marker_min: AssetBaseCfg | None = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PartitionBoundsMarkerMin",
        init_state=AssetBaseCfg.InitialStateCfg(pos=_PARTITION_BOUNDS_MIN),
        spawn=_PARTITION_BOUNDS_MARKER_SPAWN_CFG,
    )

    partition_bounds_marker_max: AssetBaseCfg | None = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PartitionBoundsMarkerMax",
        init_state=AssetBaseCfg.InitialStateCfg(pos=_PARTITION_BOUNDS_MAX),
        spawn=_PARTITION_BOUNDS_MARKER_SPAWN_CFG,
    )


@configclass
class FrankaCableCameraSceneCfg(FrankaCableSceneCfg):
    """Franka cable scene with a base camera."""

    base_camera: CameraCfg = FRANKA_CAMERA_CFG


@configclass
class CommandsCfg:
    """Goal position for cable segment 6 in the robot root frame."""

    cable_pose = mdp.CableUniformPoseCommandCfg(
        asset_name="robot",
        object_name="cable",
        segment_index=_CABLE_MIDDLE_SEGMENT_INDEX,
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.CableUniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6),
            pos_y=(-0.25, 0.25),
            pos_z=(0.25, 0.5),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
        success_vis_asset_name="table",
        success_visualizer_cfg=VisualizationMarkersCfg(
            prim_path="/Visuals/SuccessMarkers",
            markers={
                "failure": TABLE_SPAWN_CFG.replace(
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.5, 0.5)), visible=True
                ),
                "success": TABLE_SPAWN_CFG.replace(
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.8, 0.5)), visible=True
                ),
            },
        ),
    )


@configclass
class ObservationsCfg:
    """Policy observations for cable lifting."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        cable_segment_positions = ObsTerm(
            func=mdp.cable_segment_positions_in_robot_root_frame,
            params={"asset_cfg": SceneEntityCfg("cable")},
        )
        target_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "cable_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class FrankaCableCameraObservationsCfg(FrankaCameraObservationsCfg):
    """Observation groups for visual cable lifting."""

    @configclass
    class PolicyCfg(FrankaCameraObservationsCfg.PolicyCfg):
        target_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "cable_pose"})

    @configclass
    class PerceptionCfg(ObsGroup):
        cable_segment_positions = ObsTerm(
            func=mdp.cable_segment_positions_in_robot_root_frame,
            params={"asset_cfg": SceneEntityCfg("cable")},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    perception: PerceptionCfg = PerceptionCfg()


@configclass
class EventCfg(FrankaSoftEventCfg):
    """Reset events for the Franka cable environment."""

    reset_deformable: EventTerm | None = None

    reset_cable = EventTerm(
        func="isaaclab_tasks.core.lift.mdp.events:reset_cable_state_uniform",
        mode="reset",
        params={
            "position_range": {"x": (-0.15, 0.1), "y": (-0.2, 0.2), "z": (0.0, 0.0)},
            "asset_cfg": SceneEntityCfg("cable"),
        },
    )


@configclass
class RewardsCfg:
    """Cable lifting rewards."""

    reaching_cable = RewTerm(
        func=mdp.cable_ee_distance,
        params={
            "std": 0.1,
            "asset_cfg": SceneEntityCfg("cable"),
        },
        weight=1.0,
    )

    lifting_cable = RewTerm(
        func=mdp.cable_lifting,
        params={
            "std": 0.05,
            "minimal_height": 0.0,
            "asset_cfg": SceneEntityCfg("cable"),
        },
        weight=5.0,
    )

    cable_goal_tracking = RewTerm(
        func=mdp.CableSegmentGoalDistance,
        params={
            "std": 0.3,
            "command_name": "cable_pose",
            "success_threshold": 0.05,
            "segment_index": _CABLE_MIDDLE_SEGMENT_INDEX,
            "asset_cfg": SceneEntityCfg("cable"),
        },
        weight=5.0,
    )

    success_bonus = RewTerm(
        func=mdp.cable_segment_goal_reached,
        params={
            "command_name": "cable_pose",
            "success_threshold": 0.05,
            "segment_index": _CABLE_MIDDLE_SEGMENT_INDEX,
            "asset_cfg": SceneEntityCfg("cable"),
        },
        weight=20.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-3)


@configclass
class TerminationsCfg:
    """Time out and workspace bounds terminations."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    cable_out_of_bounds = DoneTerm(
        func=mdp.cable_outside_bounds,
        params={
            "x_bounds": (0.0, 1.0),
            "y_bounds": (-0.5, 0.5),
            "z_bounds": (-0.02, 1.0),
            "asset_cfg": SceneEntityCfg("cable"),
        },
    )

    ee_below_table = DoneTerm(
        func=mdp.ee_below_minimum,
        params={"minimum_height": 0.0, "ee_frame_cfg": SceneEntityCfg("ee_frame")},
    )

    joint_vel_out_of_limit = DoneTerm(
        func=mdp.joint_vel_out_of_sim_limit,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class FrankaCableEnvCfg(FrankaSoftEnvCfg):
    """Manager-based RL environment for lifting a 12-segment cable."""

    scene: FrankaCableSceneCfg = FrankaCableSceneCfg(num_envs=8192, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.sim.physics = PhysicsCfg()
        # Close the gripper on the thin cable; the shared beam default only closes to 0.01 m.
        self.actions.ik.gripper_action.close_command_expr = {"panda_finger_joint1": 0.0}
        # Only a partitioned Kit RTX render can cull the cable. The check mirrors the default that
        # IsaacRtxRendererCfg.enable_scene_partitioning resolves to; assigning that field explicitly
        # bypasses it, so re-add the markers by hand in that case.
        if not isaac_rtx_per_env_scene_partition_enabled():
            self.scene.partition_bounds_marker_min = None
            self.scene.partition_bounds_marker_max = None


@configclass
class FrankaCableCameraEnvCfg(FrankaCableEnvCfg):
    """Visual Franka cable lifting environment."""

    scene: FrankaCableCameraSceneCfg = FrankaCableCameraSceneCfg(num_envs=128, env_spacing=2.0)
    observations: FrankaCableCameraObservationsCfg = FrankaCableCameraObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.num_rerenders_on_reset = 2
