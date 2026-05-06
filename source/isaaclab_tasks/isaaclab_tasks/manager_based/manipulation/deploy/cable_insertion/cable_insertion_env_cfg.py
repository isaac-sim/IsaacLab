# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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
from isaaclab_tasks.manager_based.manipulation.deploy.mdp.noise_models import ResetSampledConstantNoiseModelCfg

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(CONFIG_DIR, "cable_insertion_assets")

# ---------------------------------------------------------------------------
# USD body-frame offsets (GB300 asset geometry)
# ---------------------------------------------------------------------------
# The GB300 socket/plug USD root frames are far from the actual insertion
# geometry.  These constants map root -> insertion point in each asset's
# local frame and are used for observations, rewards, and computing the
# correct root positions so the geometry ends up at the desired workspace
# location.
SOCKET_INSERTION_OFFSET = [0.0254, 0.5347, 0.0543]
PLUG_INSERTION_OFFSET = [0.03, 0.0, 0.0]
# Plug goal rotation relative to socket: euler(180, 0, -90) deg as (x,y,z,w).
PLUG_GOAL_ROT = [0.70711, -0.70711, 0.0, 0.0]
PLUG_GOAL_ROT_INV = [-0.70711, 0.70711, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Pure-python quaternion helpers (for module-level constant computation)
# ---------------------------------------------------------------------------

def _quat_rotate_vec(q_xyzw, v):
    """Apply quaternion rotation to a 3D vector."""
    qx, qy, qz, qw = q_xyzw
    vx, vy, vz = v
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    )


def _quat_mul(q1_xyzw, q2_xyzw):
    """Multiply two quaternions in (x, y, z, w) format."""
    x1, y1, z1, w1 = q1_xyzw
    x2, y2, z2, w2 = q2_xyzw
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def compute_socket_root(geometry_pos, socket_rot):
    """Compute socket USD root position from desired insertion-geometry world position.

    The GB300 socket USD root frame is offset from the actual insertion
    geometry by ``SOCKET_INSERTION_OFFSET`` in the socket's local frame.
    This function inverts that offset for a given world-frame socket rotation.
    """
    rotated = _quat_rotate_vec(socket_rot, SOCKET_INSERTION_OFFSET)
    return (
        geometry_pos[0] - rotated[0],
        geometry_pos[1] - rotated[1],
        geometry_pos[2] - rotated[2],
    )


def compute_plug_pose(geometry_pos, socket_rot, z_clearance=0.0):
    """Compute plug USD root position and world-frame rotation.

    Returns ``(plug_root_pos, plug_rot)`` such that the plug insertion
    point lands at ``geometry_pos`` (plus optional vertical clearance)
    with the correct goal orientation relative to the socket.
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
# Default socket/plug workspace positions (identity socket rotation)
# ---------------------------------------------------------------------------
_INSERTION_POINT = [0.5, 0.0, 0.0]
_DEFAULT_SOCKET_ROT = (0.0, 0.0, 0.0, 1.0)

_SOCKET_ROOT_POS = compute_socket_root(_INSERTION_POINT, _DEFAULT_SOCKET_ROT)
_PLUG_ROOT_POS, _DEFAULT_PLUG_ROT = compute_plug_pose(
    _INSERTION_POINT, _DEFAULT_SOCKET_ROT, z_clearance=0.033,
)

##
# Asset Configurations
##


@configclass
class GB300Plug(RigidObjectCfg):
    """Configuration for GB300 Plug (held asset)."""

    prim_path = "{ENV_REGEX_NS}/GB300Plug"
    spawn = sim_utils.UsdFileCfg(
        usd_path=os.path.join(ASSETS_DIR, "plug_A_no_snapfit_latch_no_bulge_collision_mesh.usd"),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            kinematic_enabled=False,
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
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
    )
    init_state = RigidObjectCfg.InitialStateCfg(pos=_PLUG_ROOT_POS, rot=PLUG_GOAL_ROT)


@configclass
class GB300Socket(RigidObjectCfg):
    """Configuration for GB300 Socket (fixed asset)."""

    prim_path = "{ENV_REGEX_NS}/GB300Socket"
    spawn = sim_utils.UsdFileCfg(
        usd_path=os.path.join(ASSETS_DIR, "socket_A_simplified_minimal.usd"),
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
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
    )
    init_state = RigidObjectCfg.InitialStateCfg(pos=_SOCKET_ROOT_POS, rot=(0.0, 0.0, 0.0, 1.0))


##
# Environment configuration
##


@configclass
class CableInsertionSceneCfg(InteractiveSceneCfg):
    """Configuration for the cable insertion scene."""

    replicate_physics = False

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    gb300_plug = GB300Plug()
    gb300_socket = GB300Socket()

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
            params={"asset_cfg": SceneEntityCfg("gb300_socket"), "offset": SOCKET_INSERTION_OFFSET},
            noise=ResetSampledConstantNoiseModelCfg(
                noise_cfg=UniformNoiseCfg(n_min=-0.01, n_max=0.01, operation="add")
            ),
        )
        socket_quat = ObsTerm(
            func=mdp.rigid_object_quat_w,
            params={"asset_cfg": SceneEntityCfg("gb300_socket")},
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
            params={"asset_cfg": SceneEntityCfg("gb300_socket"), "offset": SOCKET_INSERTION_OFFSET},
        )
        socket_quat = ObsTerm(
            func=mdp.rigid_object_quat_w,
            params={"asset_cfg": SceneEntityCfg("gb300_socket")},
        )
        plug_pos = ObsTerm(
            func=mdp.rigid_object_pos_w,
            params={"asset_cfg": SceneEntityCfg("gb300_plug"), "offset": PLUG_INSERTION_OFFSET},
        )
        plug_quat = ObsTerm(
            func=mdp.rigid_object_quat_w,
            params={"asset_cfg": SceneEntityCfg("gb300_plug")},
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
            "asset_cfg_1": SceneEntityCfg("gb300_socket"),
            "asset_cfg_2": SceneEntityCfg("gb300_plug"),
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
            "asset_cfg_1": SceneEntityCfg("gb300_socket"),
            "asset_cfg_2": SceneEntityCfg("gb300_plug"),
            "offset_1": SOCKET_INSERTION_OFFSET,
            "offset_2": PLUG_INSERTION_OFFSET,
            "rot_offset_2": PLUG_GOAL_ROT_INV,
            "kp_exp_coeffs": [(50, 0.0001), (300, 0.0001), (600, 0.0001)],
            "kp_use_sum_of_exps": False,
            "keypoint_scale": 0.15,
        },
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-5.0e-06)

    # action_l2 = RewTerm(func=mdp.action_l2, weight=-5.0e-06)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CableInsertionEnvCfg(ManagerBasedRLEnvCfg):
    scene: CableInsertionSceneCfg = CableInsertionSceneCfg(num_envs=4096, env_spacing=2.5)
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
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.decimation = 8
        self.sim.render_interval = self.decimation
        self.sim.dt = 1.0 / 240.0
