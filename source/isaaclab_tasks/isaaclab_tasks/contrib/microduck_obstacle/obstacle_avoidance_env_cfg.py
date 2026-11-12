# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Front-facing assisted walking among static obstacles using an 8x8 time-of-flight sensor."""

import math
import os
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.paths import ISAACLAB_ROOT
from isaaclab.sensors import ContactSensorCfg, MultiMeshRayCasterCameraCfg, patterns
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.microduck import mdp as walking_mdp
from isaaclab_tasks.contrib.microduck.flat_env_cfg import (
    ActionsCfg,
    CommandsCfg,
    EventsCfg,
    MicroDuckSceneCfg,
    MicroDuckVelocityFlatEnvCfg,
    ObservationsCfg,
    TerminationsCfg,
)
from isaaclab_tasks.utils import preset

from isaaclab_assets.robots.microduck import MICRODUCK_ALLCOLLISIONS_BACKLASH_CFG, MICRODUCK_ALLCOLLISIONS_CFG

from . import mdp
from .mdp import geometry as navigation
from .mdp.actions import MicroDuckWalkingActionCfg
from .mdp.events import OBSTACLE_CELLS, reset_obstacle_field
from .mdp.observations import head_sensor_age, tof_ranges_and_validity
from .sensors import MicroDuckToFCameraCfg, ToFRandomizationCfg

SIMREADY_ASSET_ROOT = os.environ.get(
    "MICRODUCK_NAVIGATION_PROPS",
    str(Path(ISAACLAB_ROOT) / "assets" / "simready"),
).rstrip("/")
"""Downloaded SimReady packages; MICRODUCK_NAVIGATION_PROPS overrides the local directory."""

MICRODUCK_HEAD_BODY_PATH = "{ENV_REGEX_NS}/Robot/Geometry/trunk_base/neck/neck_pitch_1/yaw_roll_motion/jaw_soft"
"""Prim path of the head body carrying the camera and time-of-flight sensor."""

MICRODUCK_TOF_POS = (-0.0176, 0.0, -0.083)
"""ToF mount [m]: 0.22 m above flat ground at the default 0.125 m root height and home joints."""

MICRODUCK_TOF_ROT = (0.0, math.sin(math.radians(55.0)), 0.0, math.cos(math.radians(55.0)))
"""Head-relative XYZW mount: optical axis 20 degrees down at the home neck/head pitch of +0.3491 rad."""

TOF_FOV_DEG = 45.0
"""Nominal square field of view of the 8x8 zone grid [deg]; not independently calibrated."""

TOF_MAX_RANGE = 4.0
"""Maximum time-of-flight range [m]; farther returns are invalid."""

_TOF_APERTURE = 2.0 * math.tan(math.radians(TOF_FOV_DEG / 2.0))


# One source for primitive collision dimensions and reward footprints. Keep this
# order stable: the critic observes one clearance value per configured obstacle.
OBSTACLE_FOOTPRINTS = {
    "obstacle": navigation.ObstacleFootprintCfg(size=(0.2286, 0.2286)),
    "crate": navigation.ObstacleFootprintCfg(size=(0.6994351148605347, 0.5211517810821533)),
    "tall_box": navigation.ObstacleFootprintCfg(size=(0.4, 0.3)),
    "low_block": navigation.ObstacleFootprintCfg(size=(0.3999999761581421, 0.30000001192092896)),
    "cylinder": navigation.ObstacleFootprintCfg(size=(0.46326370541775974, 0.5750020593644363)),
    "thin_post": navigation.ObstacleFootprintCfg(size=(0.2286, 0.2286)),
    "wall_left": navigation.ObstacleFootprintCfg(size=(0.12, 0.9)),
    "wall_right": navigation.ObstacleFootprintCfg(size=(0.12, 0.9)),
}


@sim_utils.clone
def spawn_fixed_prop(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Load a SimReady prop as one stationary obstacle with a floor-centered origin.

    Furniture can contain separate moving parts, such as drawers. Navigation needs
    their existing shapes, but treats the whole prop as one obstacle. Disable those
    internal joints and bodies, then give the parent one kinematic rigid body:
    resets can reposition it, while gravity and robot contacts cannot move it.

    Center the collision bounds in X/Y and place their lowest point at Z=0. This
    makes layout positions and clearance footprints refer to the same origin.
    All changes are scene-layer overrides; downloaded USD files stay unchanged.
    """
    # Keep the source asset beneath a placement transform so its origin can be adjusted.
    root = sim_utils.create_prim(prim_path, translation=translation, orientation=orientation)
    placement = UsdGeom.Xform.Define(root.GetStage(), f"{prim_path}/Placement")
    asset = sim_utils.spawn_from_usd(f"{prim_path}/Placement/Asset", cfg.replace(rigid_props=None, mass_props=None))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    bounds = Gf.Range3d()
    # Preserve collision geometry, but remove independent motion of the internal parts.
    for prim in Usd.PrimRange(asset):
        if prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            bounds.UnionWith(cache.ComputeRelativeBound(prim, root).ComputeAlignedRange())
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        if prim.HasAPI(UsdPhysics.MassAPI):
            prim.RemoveAPI(UsdPhysics.MassAPI)
        if prim.IsA(UsdPhysics.Joint):
            UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
    if bounds.IsEmpty():
        raise ValueError(f"No enabled collision geometry in {cfg.usd_path}")
    # Ground and center the geometry, then apply the configured body properties to the parent.
    center = (bounds.GetMin() + bounds.GetMax()) * 0.5
    placement.AddTranslateOp().Set(Gf.Vec3d(-center[0], -center[1], -bounds.GetMin()[2]))
    sim_utils.apply_rigid_body_properties(prim_path, [cfg.rigid_props], create_if_missing=True)
    return root


@configclass
class MicroDuckObstacleSceneCfg(MicroDuckSceneCfg):
    """Static object field with a clear central spawn and head ToF sensing."""

    lazy_sensor_update = False
    robot = preset(
        default=MICRODUCK_ALLCOLLISIONS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        backlash=MICRODUCK_ALLCOLLISIONS_BACKLASH_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
    )
    obstacle_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base(/.*)?",
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/" + ("Obstacle" if name == "obstacle" else name) for name in OBSTACLE_CELLS
        ],
        history_length=4,
        update_period=0.005,
    )

    obstacle = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*OBSTACLE_CELLS["obstacle"], 0.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/Props/S_TrafficCone.usd",
            # Original collision bounds [m]; match the 12-inch height and 9-inch base.
            scale=(0.2286 / 0.335697052, 0.2286 / 0.333944370, 0.3048 / 0.462812192),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
        ),
    )
    thin_post = obstacle.replace(
        prim_path="{ENV_REGEX_NS}/thin_post",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*OBSTACLE_CELLS["thin_post"], 0.0)),
    )
    # Adapt the public SimReady packages at spawn time; no prepared local USDs are needed.
    crate = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/crate",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*OBSTACLE_CELLS["crate"], 0.0)),
        spawn=sim_utils.UsdFileCfg(
            func=spawn_fixed_prop,
            usd_path=f"{SIMREADY_ASSET_ROOT}/box_cardboard_a01/simready_usd/sm_box_cardboard_a01_01.usd",
            variants={"Newton": "Enabled", "PhysX": "Disabled", "MuJoCo": "Disabled"},
            make_uninstanceable=True,
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
        ),
    )
    low_block = crate.replace(
        prim_path="{ENV_REGEX_NS}/low_block",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*OBSTACLE_CELLS["low_block"], 0.0)),
        spawn=crate.spawn.replace(
            usd_path=f"{SIMREADY_ASSET_ROOT}/trashCan_office_black_a03/simready_usd/sm_trashCan_office_black_a03_01.usd",
        ),
    )
    cylinder = crate.replace(
        prim_path="{ENV_REGEX_NS}/cylinder",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*OBSTACLE_CELLS["cylinder"], 0.0)),
        spawn=crate.spawn.replace(
            usd_path=f"{SIMREADY_ASSET_ROOT}/table_bedside_b01/simready_usd/sm_table_bedside_b01_01.usd",
        ),
    )
    tall_box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/tall_box",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*OBSTACLE_CELLS["tall_box"], 0.06)),
        spawn=sim_utils.CuboidCfg(
            size=(*OBSTACLE_FOOTPRINTS["tall_box"].size, 0.12),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.45)),
        ),
    )
    wall_left = tall_box.replace(
        prim_path="{ENV_REGEX_NS}/wall_left",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*OBSTACLE_CELLS["wall_left"], 0.325)),
        spawn=tall_box.spawn.replace(
            size=(*OBSTACLE_FOOTPRINTS["wall_left"].size, 0.65),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.65, 0.66, 0.68)),
        ),
    )
    wall_right = wall_left.replace(
        prim_path="{ENV_REGEX_NS}/wall_right",
        spawn=wall_left.spawn.replace(size=(*OBSTACLE_FOOTPRINTS["wall_right"].size, 0.65)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*OBSTACLE_CELLS["wall_right"], 0.325)),
    )
    # One ray per VL53L5CX zone; policy consumes optical-axis depth, as supported by the floor recording.
    head_tof = MicroDuckToFCameraCfg(
        prim_path=MICRODUCK_HEAD_BODY_PATH,
        update_period=1.0 / 15.0,
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(pos=MICRODUCK_TOF_POS, rot=MICRODUCK_TOF_ROT, convention="world"),
        mesh_prim_paths=[
            # Use the finite visual floor; the infinite collision plane loses raycast precision.
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
                prim_expr="/World/ground/terrain/Environment/Geometry", track_mesh_transforms=False
            ),
            *[
                MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
                    prim_expr="{ENV_REGEX_NS}/" + ("Obstacle" if name == "obstacle" else name)
                )
                for name in OBSTACLE_CELLS
            ],
        ],
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=1.0, horizontal_aperture=_TOF_APERTURE, vertical_aperture=_TOF_APERTURE, width=8, height=8
        ),
        randomization=ToFRandomizationCfg(
            mount_pitch_degrees=(-10.0, 10.0),
            mount_height_m=(-0.02, 0.02),
            floor_validity_midpoint_offset_m=(0.0, 0.13),
            near_floor_edge_impairment_probability=0.5,
        ),
        data_types=["distance_to_image_plane", "distance_to_camera"],
        max_distance=TOF_MAX_RANGE,
        depth_clipping_behavior="zero",
    )


@configclass
class MicroDuckObstacleObservationsCfg(ObservationsCfg):
    """Deployable proprioception and ToF history, with privileged critic geometry."""

    @configclass
    class NavigationPolicyCfg(ObservationsCfg.PolicyCfg):
        applied_command = ObsTerm(func=mdp.applied_velocity_commands)

    @configclass
    class ToFCfg(ObsGroup):
        # 200 ms covers about three distinct 15 Hz frames at the 50 Hz control rate.
        history_length = 10
        ranges_and_validity = ObsTerm(func=tof_ranges_and_validity)
        age = ObsTerm(func=head_sensor_age)

    @configclass
    class NavigationCriticCfg(ObservationsCfg.CriticCfg):
        obstacle_clearance = ObsTerm(func=navigation.obstacle_clearances)

    policy: NavigationPolicyCfg = NavigationPolicyCfg()
    critic: NavigationCriticCfg = NavigationCriticCfg()
    tof: ToFCfg = ToFCfg()


@configclass
class MicroDuckObstacleActionsCfg:
    """Keep the native joint_pos term name for MicroDuck encoder-bias events."""

    joint_pos: MicroDuckWalkingActionCfg = MicroDuckWalkingActionCfg(
        asset_name="robot",
        # Native walking playback exports the published checkpoint here.
        policy_path=str(
            Path(ISAACLAB_ROOT)
            / ".pretrained_checkpoints/rsl_rl/https/omniverse-content-production.s3-us-west-2.amazonaws.com"
            / "Assets/Isaac/6.2/Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/exported/policy.pt"
        ),
        low_level_decimation=4,
        low_level_actions=ActionsCfg().joint_pos,
        low_level_observations=ObservationsCfg.PolicyCfg(),
    )


@configclass
class MicroDuckObstacleEventsCfg(EventsCfg):
    obstacle_field = EventTerm(
        func=reset_obstacle_field,
        mode="reset",
        params={
            "layout": "mixed",
            "layout_weights": {
                "ring": 0.0,
                "open": 0.10,
                "barrier": 0.10,
                "slalom": 0.0,
                "passage": 0.05,
                "thin_low": 0.05,
                "blocked": 0.02,
                "corner": 0.03,
                "scattered": 0.65,
            },
        },
    )


@configclass
class MicroDuckNavigationCommandsCfg(CommandsCfg):
    """Emulate joystick steering toward a fixed goal, without obstacle-aware routing."""

    base_velocity = mdp.MicroDuckNavigationCommandCfg(
        rel_destination_envs=1.0,
        asset_name="robot",
        resampling_time_range=(3.0, 6.0),
        rel_standing_envs=0.1,
        rel_forward_envs=0.65,
        rel_turn_in_place_envs=0.15,
        forward_min_speed=0.15,
        debug_vis=False,
        ranges=walking_mdp.MicroDuckVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.35), lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.75, 0.75)
        ),
    )


@configclass
class MicroDuckNavigationRewardsCfg:
    """Bounded task rewards; the frozen walker already owns joint-level gait control."""

    progress = RewTerm(func=mdp.navigation_gated_progress, weight=5.0)
    ungated_progress = RewTerm(func=mdp.navigation_progress, weight=0.0)
    command_tracking = RewTerm(func=mdp.navigation_velocity_tracking, weight=1.0)
    yaw_tracking = RewTerm(func=mdp.navigation_yaw_tracking, weight=4.0)
    # Event terms cancel manager dt scaling; arrival credit is revoked on subsequent failure.
    goal_arrival = RewTerm(func=mdp.NavigationGoalArrival, weight=30.0)
    barrier_bypass = RewTerm(func=mdp.NavigationObstacleBypass, weight=5.0)
    # Safe stopping near an obstacle should not accumulate a proximity penalty.
    clearance = RewTerm(func=mdp.obstacle_clearance_risk, weight=0.0, params={"safety_distance": 0.2})
    approach = RewTerm(
        func=mdp.obstacle_approach_risk,
        weight=-4.0,
        params={"safety_distance": 0.2, "time_horizon": 2.0, "commanded_risk_weight": 1.0},
    )
    command_rate = RewTerm(func=mdp.navigation_action_rate, weight=-5.0)
    # terminated_penalty cancels manager dt scaling: one failure costs exactly 30.
    failure = RewTerm(func=mdp.terminated_penalty, weight=-30.0)
    upright = RewTerm(
        func=walking_mdp.upright,
        weight=0.0,
        params={"std": math.sqrt(0.05), "asset_cfg": SceneEntityCfg("robot", body_names=["trunk_base"])},
    )


@configclass
class MicroDuckNavigationTerminationsCfg(TerminationsCfg):
    obstacle_collision = DoneTerm(func=mdp.obstacle_collision)
    field_boundary = DoneTerm(func=mdp.field_boundary, time_out=True)


@configclass
class MicroDuckNavigationCurriculumCfg:
    """Navigation starts with fixed, bounded randomization rather than walking schedules."""

    pass


@configclass
class MicroDuckObstacleAvoidanceEnvCfg(MicroDuckVelocityFlatEnvCfg):
    """SimReady obstacle navigation above a frozen walking actor.

    Robot and cone assets use Isaac Lab public URLs; download SimReady packages to assets/simready.
    MICRODUCK_NAVIGATION_PROPS optionally selects a different package directory.
    """

    obstacle_footprints: dict[str, navigation.ObstacleFootprintCfg] = OBSTACLE_FOOTPRINTS
    """Planar collision geometry keyed by scene asset; insertion order defines critic slots."""

    scene: MicroDuckObstacleSceneCfg = MicroDuckObstacleSceneCfg(num_envs=256, env_spacing=10.0)
    observations: MicroDuckObstacleObservationsCfg = MicroDuckObstacleObservationsCfg()
    actions: MicroDuckObstacleActionsCfg = MicroDuckObstacleActionsCfg()
    commands: MicroDuckNavigationCommandsCfg = MicroDuckNavigationCommandsCfg()
    rewards: MicroDuckNavigationRewardsCfg = MicroDuckNavigationRewardsCfg()
    terminations: MicroDuckNavigationTerminationsCfg = MicroDuckNavigationTerminationsCfg()
    curriculum: MicroDuckNavigationCurriculumCfg = MicroDuckNavigationCurriculumCfg()
    recorders = None
    events: MicroDuckObstacleEventsCfg = MicroDuckObstacleEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0
        self.sim.physics.solver_cfg.nconmax = 200
        self.sim.physics.solver_cfg.njmax = 1024
        # Preserve the frozen walking controller's 50 Hz observation/delay contract.
        self.commands.head_pose.ranges = ((0.0, 0.0),) * 4
        self.commands.body_pose.ranges = ((0.0, 0.0),) * 6
        self.events.reset_base.params["pose_range"] = {
            "x": (-0.15, 0.15),
            "y": (-0.15, 0.15),
            "z": (-0.005, 0.005),
            "yaw": (-3.14, 3.14),
        }
        self.events.push_robot.interval_range_s = (5.0, 10.0)
        self.events.push_robot.params["velocity_range"] = {"x": (-0.1, 0.1), "y": (-0.1, 0.1)}

        # Larger natural-scale props need wider layouts; slots/contact/ray targets stay unchanged.
        self.events.obstacle_field.params.update(
            layout="mixed",
            spacing_scale=1.6,
            shuffle_groups=(("obstacle", "crate", "tall_box", "low_block", "cylinder", "thin_post"),),
        )
        self.terminations.field_boundary.params = {"extent": 5.0}
        self.scene.env_spacing = 14.0
        # Mesh contacts at 16k environments exceed the primitive task buffers.
        self.sim.physics.collision_cfg.max_triangle_pairs = 320_000_000
        self.sim.physics.collision_cfg.rigid_contact_max = 16_000_000
