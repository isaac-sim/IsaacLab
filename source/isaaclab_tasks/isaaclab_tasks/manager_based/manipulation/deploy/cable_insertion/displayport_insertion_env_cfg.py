# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base RL environment for inserting a DisplayPort plug into a socket.

This mirrors :mod:`cable_insertion_env_cfg` (the GB300 cable-insertion base)
but targets the right-angle DisplayPort plug/socket assets in
``display_cable_insertion_assets``.

Assets (``display_port_plug_fixed.usd``, ``display_port_socket_fixed.usd``) were
produced from STEP source via the ``omniverse-cad-to-simready`` pipeline and
post-processed by ``physical-ai-skill-hub-dev/scripts/finalize_dp_assets.py``.
All issues from ``output_dir/displayport_asset_fixes_required.md`` have been
resolved offline: geometry is in metres, single root :class:`RigidBodyAPI`,
no embedded ``PhysicsScene``, ``convexDecomposition`` on the plug, ``triangleMesh``
on the socket (all bodies enabled). Assets load with plain
:class:`~isaaclab.sim.UsdFileCfg` at ``scale=(1,1,1)``; no custom spawner needed.

Geometry constants below were derived from the **live-sim-verified** seated pose
of the drop-test (plug pos ``(0,0,0.2096)`` rot ``(0.70711,0.70711,0,0)``;
socket pos ``(0,0,0.15)`` rot ``(0.5,0.5,0.5,-0.5)`` in Isaac Lab quat order),
re-parameterized through the same quaternion helpers as the GB300 base so the
keypoint goal exactly reproduces the verified mate.
"""

import os
from dataclasses import MISSING

from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.simulation_cfg import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg

import isaaclab_tasks.manager_based.manipulation.deploy.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.deploy.cable_insertion.cable_insertion_env_cfg import (
    _quat_mul,
    _quat_rotate_vec,
)
from isaaclab_tasks.manager_based.manipulation.deploy.mdp.noise_models import ResetSampledConstantNoiseModelCfg

CABLE_INSERTION_DIR = os.path.dirname(os.path.abspath(__file__))
DISPLAY_ASSETS_DIR = os.path.join(CABLE_INSERTION_DIR, "display_cable_insertion_assets")

# _DP_SPAWNER = (  # old: runtime USD patching for original multi-body simready assets
#     "isaaclab_tasks.manager_based.manipulation.deploy.cable_insertion"
#     ".config.displayport_basic.spawners:spawn_usd_with_physics"
# )

# ---------------------------------------------------------------------------
# USD body-frame offsets (DisplayPort asset geometry)
# ---------------------------------------------------------------------------
# Root -> insertion(mate) point in each asset's local frame. Derived from the
# verified seated pose with the mating reference point chosen at the socket top
# face. See module docstring; round-trip-verified against the GB300 helpers.
SOCKET_INSERTION_OFFSET = [0.0375, 0.0, 0.0]
PLUG_INSERTION_OFFSET = [0.0, 0.0, 0.0221]
# Plug orientation relative to socket at the mated pose, (x, y, z, w).
PLUG_GOAL_ROT = [0.0, -0.70711, 0.0, 0.70711]
PLUG_GOAL_ROT_INV = [0.0, 0.70711, 0.0, 0.70711]


def compute_socket_root(geometry_pos, socket_rot):
    """Compute socket USD root position from a desired insertion-geometry world position.

    Inverts :data:`SOCKET_INSERTION_OFFSET` (expressed in the socket's local
    frame) for a given world-frame socket rotation.
    """
    rotated = _quat_rotate_vec(socket_rot, SOCKET_INSERTION_OFFSET)
    return (
        geometry_pos[0] - rotated[0],
        geometry_pos[1] - rotated[1],
        geometry_pos[2] - rotated[2],
    )


def compute_plug_pose(geometry_pos, socket_rot, z_clearance=0.0):
    """Compute plug USD root position and world-frame rotation.

    Returns ``(plug_root_pos, plug_rot)`` such that the plug insertion point
    lands at ``geometry_pos`` (plus optional vertical clearance) with the
    correct goal orientation relative to the socket.
    """
    plug_rot = _quat_mul(socket_rot, tuple(PLUG_GOAL_ROT))
    plug_offset_world = _quat_rotate_vec(plug_rot, PLUG_INSERTION_OFFSET)
    plug_root = (
        geometry_pos[0] - plug_offset_world[0],
        geometry_pos[1] - plug_offset_world[1],
        geometry_pos[2] - plug_offset_world[2] + z_clearance,
    )
    return plug_root, plug_rot


# ---------------------------------------------------------------------------
# Default socket/plug workspace positions (drop-test layout, socket opening up)
# ---------------------------------------------------------------------------
_INSERTION_POINT = [0.0, 0.0, 0.1875]
_DEFAULT_SOCKET_ROT = (0.5, 0.5, 0.5, -0.5)

_SOCKET_ROOT_POS = compute_socket_root(_INSERTION_POINT, _DEFAULT_SOCKET_ROT)
_PLUG_ROOT_POS, _DEFAULT_PLUG_ROT = compute_plug_pose(
    _INSERTION_POINT, _DEFAULT_SOCKET_ROT, z_clearance=0.033,
)

# _DP_SCALE = (0.0254, 0.0254, 0.0254)  # old: inch→metre workaround for original simready assets; geometry is now natively in metres

##
# Asset Configurations
##


@configclass
class DisplayPortPlug(RigidObjectCfg):
    """DisplayPort right-angle plug (held asset) — dynamic."""

    prim_path = "{ENV_REGEX_NS}/DisplayPortPlug"
    spawn = sim_utils.UsdFileCfg(
        # func=_DP_SPAWNER,  # old: runtime patcher for original multi-body simready assets
        # usd_path=os.path.join(DISPLAY_ASSETS_DIR, "2584n111_displayport_cord_plug_latch_removed_simready.usd"),  # old: inches, multi-body, instanced
        # usd_path=os.path.join(DISPLAY_ASSETS_DIR, "display_port_plug_fixed_watertight.usd"),  # old: convexDecomposition, hulls too wide → blocks slot entrance
        usd_path=os.path.join(DISPLAY_ASSETS_DIR, "display_port_plug_fixed_sdf.usd"),
        # scale=_DP_SCALE,  # old: inch→metre workaround; finalize_dp_assets.py baked metres into vertices
        scale=(1.0, 1.0, 1.0),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            kinematic_enabled=False,
            # Gentle depenetration: high values turn residual mate overlap into
            # an explosive ejection. 0.5 lets PhysX resolve overlaps smoothly.
            max_depenetration_velocity=0.5,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=128,
            solver_velocity_iteration_count=1,
            # Leave uncapped at the PhysX default; pairing a cap with a high
            # depenetration velocity amplified contact blow-ups.
            max_contact_impulse=None,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.03),
        # Body5 clearance to blade is ~0.27mm. PhysX fires contact at (plug + socket) offsets combined
        # (0.27mm physical gap). Keep plug offset small so there's real clearance before repulsion starts.
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.00001, rest_offset=-0.00005),
    )
    init_state = RigidObjectCfg.InitialStateCfg(pos=_PLUG_ROOT_POS, rot=_DEFAULT_PLUG_ROT)


@configclass
class DisplayPortSocket(RigidObjectCfg):
    """DisplayPort socket (fixed asset) — kinematic."""

    prim_path = "{ENV_REGEX_NS}/DisplayPortSocket"
    spawn = sim_utils.UsdFileCfg(
        # func=_DP_SPAWNER,  # old: runtime patcher for original multi-body simready assets
        # usd_path=os.path.join(DISPLAY_ASSETS_DIR, "2584n111_displayport_cord_socket_screws_removed_simready.usd"),  # old: inches, multi-body
        # usd_path=os.path.join(DISPLAY_ASSETS_DIR, "display_port_socket_fixed_watertight.usd"),  # old: triangleMesh, may have entrance artifacts
        usd_path=os.path.join(DISPLAY_ASSETS_DIR, "display_port_socket_fixed_sdf.usd"),
        # scale=_DP_SCALE,  # old: inch→metre workaround
        scale=(1.0, 1.0, 1.0),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            kinematic_enabled=True,
            max_depenetration_velocity=5.0,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=128,
            solver_velocity_iteration_count=1,
            max_contact_impulse=1e32,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=None),
        # rest_offset negative on socket too: combined rest = -0.15mm so blade can slide in.
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0001, rest_offset=-0.0001),
    )
    init_state = RigidObjectCfg.InitialStateCfg(pos=_SOCKET_ROOT_POS, rot=_DEFAULT_SOCKET_ROT)


##
# Environment configuration
##


@configclass
class DisplayportInsertionSceneCfg(InteractiveSceneCfg):
    """Configuration for the DisplayPort insertion scene."""

    # replicate_physics = False  # old: required when spawner patched de-instanced geometry at runtime
    replicate_physics = True

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    dp_plug = DisplayPortPlug()
    dp_socket = DisplayPortSocket()

    robot: ArticulationCfg = MISSING

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = MISSING
    gripper_action: ActionTerm | None = None


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})
        joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})
        socket_pos = ObsTerm(
            func=mdp.rigid_object_pos_w,
            params={"asset_cfg": SceneEntityCfg("dp_socket"), "offset": SOCKET_INSERTION_OFFSET},
            noise=ResetSampledConstantNoiseModelCfg(
                noise_cfg=UniformNoiseCfg(n_min=-0.01, n_max=0.01, operation="add")
            ),
        )
        socket_quat = ObsTerm(
            func=mdp.rigid_object_quat_w,
            params={"asset_cfg": SceneEntityCfg("dp_socket")},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})
        joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})
        socket_pos = ObsTerm(
            func=mdp.rigid_object_pos_w,
            params={"asset_cfg": SceneEntityCfg("dp_socket"), "offset": SOCKET_INSERTION_OFFSET},
        )
        socket_quat = ObsTerm(
            func=mdp.rigid_object_quat_w,
            params={"asset_cfg": SceneEntityCfg("dp_socket")},
        )
        plug_pos = ObsTerm(
            func=mdp.rigid_object_pos_w,
            params={"asset_cfg": SceneEntityCfg("dp_plug"), "offset": PLUG_INSERTION_OFFSET},
        )
        plug_quat = ObsTerm(
            func=mdp.rigid_object_quat_w,
            params={"asset_cfg": SceneEntityCfg("dp_plug")},
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    plug_socket_keypoint_tracking = RewTerm(
        func=mdp.keypoint_two_body_error,
        weight=-1.5,
        params={
            "asset_cfg_1": SceneEntityCfg("dp_socket"),
            "asset_cfg_2": SceneEntityCfg("dp_plug"),
            "offset_1": SOCKET_INSERTION_OFFSET,
            "offset_2": PLUG_INSERTION_OFFSET,
            "rot_offset_2": PLUG_GOAL_ROT_INV,
            "keypoint_scale": 0.15,
        },
    )

    plug_socket_keypoint_tracking_exp = RewTerm(
        func=mdp.keypoint_two_body_error_exp,
        weight=3.0,
        params={
            "asset_cfg_1": SceneEntityCfg("dp_socket"),
            "asset_cfg_2": SceneEntityCfg("dp_plug"),
            "offset_1": SOCKET_INSERTION_OFFSET,
            "offset_2": PLUG_INSERTION_OFFSET,
            "rot_offset_2": PLUG_GOAL_ROT_INV,
            "kp_exp_coeffs": [(50, 0.0001), (300, 0.0001), (600, 0.0001)],
            "kp_use_sum_of_exps": False,
            "keypoint_scale": 0.15,
        },
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-5.0e-06)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class DisplayportInsertionEnvCfg(ManagerBasedRLEnvCfg):
    """Base configuration for DisplayPort plug/socket insertion."""

    scene: DisplayportInsertionSceneCfg = DisplayportInsertionSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    sim: SimulationCfg = SimulationCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physics=PhysxCfg(
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_collision_stack_size=2**30,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
        ),
    )

    def __post_init__(self):
        """Post initialization."""
        self.episode_length_s = 6.66
        self.viewer.eye = (0.5, -1.8, 1.2)
        self.viewer.lookat = (0.5, 0.0, 0.5)
        self.decimation = 8
        self.sim.render_interval = self.decimation
        self.sim.dt = 1.0 / 240.0
