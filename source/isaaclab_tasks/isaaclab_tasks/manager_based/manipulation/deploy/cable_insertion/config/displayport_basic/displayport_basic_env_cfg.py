# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Passive drop-test environment for DisplayPort plug/socket assets.

No robot, no actions. The plug starts above the socket and falls under
gravity — use this to visually verify asset stability and insertion
feasibility before wiring into a full RL environment.
"""

import os

from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.simulation_cfg import SimulationCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.deploy.mdp as mdp

# ---------------------------------------------------------------------------
# Asset paths
# ---------------------------------------------------------------------------
CABLE_INSERTION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
DISPLAY_ASSETS_DIR = os.path.join(CABLE_INSERTION_DIR, "display_cable_insertion_assets")

# ---------------------------------------------------------------------------
# Plug orientation for drop test
# ---------------------------------------------------------------------------
# The DisplayPort plug has a 90-degree physical turn (right-angle connector).
# We apply the same rotation convention as the GB300 cable insertion env:
#   euler(180, 0, -90) → flip plug 180° around X (face downward toward socket)
#   then rotate -90° around Z (account for the 90° connector turn).
# Quaternion in (x, y, z, w) format.
PLUG_DROP_ROT = (0.70711, 0.70711, 0.0, 0.0)

# Alternative rotations to try if the default doesn't align the connector:
#   Pure 180° X flip (no Z correction):  (1.0, 0.0, 0.0, 0.0)
#   euler(180, 0, +90):                  (0.70711, 0.70711, 0.0, 0.0)
#   Identity (no rotation):              (0.0, 0.0, 0.0, 1.0)

SOCKET_HEIGHT = 0.15  # metres above ground
PLUG_CLEARANCE = 0.02  # metres above socket opening


##
# Asset Configurations
##


@configclass
class DisplayPortPlug(RigidObjectCfg):
    """DisplayPort right-angle plug — dynamic, falls under gravity."""

    prim_path = "{ENV_REGEX_NS}/DisplayPortPlug"
    spawn = sim_utils.UsdFileCfg(
        # func=_SPAWNER,  # old: custom spawner needed for old assets (deinstance, strip physics scenes)
        # usd_path=os.path.join(DISPLAY_ASSETS_DIR, "simready_plug.usd"),
        # usd_path=os.path.join(DISPLAY_ASSETS_DIR, "2584n111_displayport_cord_plug_latch_removed_simready.usd"),  # old: inches, needs scale=0.0254
        # usd_path="/home/shauryad/workspaces/rl_policy/physical-ai-skill-hub-dev/outputs/plug/conform/fet001-minimal/plug_material_physics.usd",  # old: absolute path, convexHull
        usd_path=os.path.join(DISPLAY_ASSETS_DIR, "display_port_plug_fixed_sdf.usd"),
        # pipeline output post-processed by finalize_dp_assets.py: metres, single root RB, convexDecomposition.
        scale=(1.0, 1.0, 1.0),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            kinematic_enabled=False,
            # Gentle depenetration: a high value (5.0) turns any residual contact
            # overlap at the tight mate into an explosive ejection. 0.5 lets PhysX
            # resolve overlaps smoothly without launching the plug.
            max_depenetration_velocity=0.5,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=128,
            solver_velocity_iteration_count=1,
            # No max-impulse cap (1e32 effectively uncapped it but combined with a
            # high depenetration velocity amplified contact blow-ups). Use the
            # PhysX default by leaving it unset.
            max_contact_impulse=None,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.03),
        # DP blade-to-cavity clearance is ~0.27mm at the tightest body (Body5).
        # contact_offset=0.3mm overshoots that gap → persistent shell penetration
        # creating a net lateral force each step that eventually ejects the plug.
        # Use 0.1mm so the contact shell stays within the cavity clearance.
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0001, rest_offset=0.0),
    )
    # NOTE: seated pose under verification (scripts/dp_verify_seated_geometry.py).
    # The dp_insertion_test.py best-of-24 solver pose was geometrically wrong
    # (blade floated ~22 mm beside the socket, not in the hole). rot is (w,x,y,z).
    init_state = RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.2096),
        rot=PLUG_DROP_ROT,
    )


@configclass
class DisplayPortSocket(RigidObjectCfg):
    """DisplayPort socket — kinematic (fixed in place)."""

    prim_path = "{ENV_REGEX_NS}/DisplayPortSocket"
    spawn = sim_utils.UsdFileCfg(
        # func=_SPAWNER,  # old: custom spawner needed for old assets
        # usd_path=os.path.join(DISPLAY_ASSETS_DIR, "2584N111_Displayport Cord_socket_screws_removed_material_physics.usd"),
        # usd_path=os.path.join(DISPLAY_ASSETS_DIR, "simready_socket.usd"),
        # usd_path=os.path.join(DISPLAY_ASSETS_DIR, "2584n111_displayport_cord_socket_screws_removed_simready.usd"),  # old: inches, needs scale=0.0254
        # usd_path="/home/shauryad/workspaces/rl_policy/physical-ai-skill-hub-dev/outputs/socket/conform/fet001-minimal/socket_material_physics.usd",  # old: absolute path
        usd_path=os.path.join(DISPLAY_ASSETS_DIR, "display_port_socket_fixed_sdf.usd"),
        # pipeline output post-processed by finalize_dp_assets.py: metres, kinematic root, all bodies enabled.
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
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0001, rest_offset=0.0),
    )
    init_state = RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, SOCKET_HEIGHT),
        rot=(0.5, 0.5, 0.5, -0.5),
    )


##
# Scene
##


@configclass
class DisplayportBasicSceneCfg(InteractiveSceneCfg):
    """Minimal scene: DisplayPort plug + socket, ground, and light. No robot."""

    # replicate_physics = False  # old: required when spawner patched de-instanced geometry at runtime
    replicate_physics = True

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    dp_plug = DisplayPortPlug()
    dp_socket = DisplayPortSocket()

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


##
# MDP — minimal (no robot, no actions)
##


@configclass
class ActionsCfg:
    """No actions — passive drop test."""

    pass


@configclass
class ObservationsCfg:
    """Track plug and socket pose for monitoring the drop."""

    @configclass
    class PolicyCfg(ObsGroup):
        plug_pos = ObsTerm(
            func=mdp.rigid_object_pos_w,
            params={"asset_cfg": SceneEntityCfg("dp_plug")},
        )
        plug_quat = ObsTerm(
            func=mdp.rigid_object_quat_w,
            params={"asset_cfg": SceneEntityCfg("dp_plug")},
        )
        socket_pos = ObsTerm(
            func=mdp.rigid_object_pos_w,
            params={"asset_cfg": SceneEntityCfg("dp_socket")},
        )
        socket_quat = ObsTerm(
            func=mdp.rigid_object_quat_w,
            params={"asset_cfg": SceneEntityCfg("dp_socket")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset to initial pose on episode boundary."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class RewardsCfg:
    """No rewards — observation-only test."""

    pass


@configclass
class TerminationsCfg:
    """End episode after timeout so the scene resets and replays the drop."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment configuration
##


@configclass
class DisplayportBasicEnvCfg(ManagerBasedRLEnvCfg):
    """Passive drop-test environment for DisplayPort plug/socket assets.

    The plug starts above the socket and falls under gravity.
    No robot, no actions — just physics simulation to validate asset
    geometry, collision meshes, and insertion feasibility.
    """

    scene: DisplayportBasicSceneCfg = DisplayportBasicSceneCfg(num_envs=1, env_spacing=2.5)
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
        self.episode_length_s = 10.0
        self.viewer.eye = (0.3, 0.3, 0.3)
        self.viewer.lookat = (0.0, 0.0, SOCKET_HEIGHT)
        self.decimation = 1
        self.sim.render_interval = 1
        self.sim.dt = 1.0 / 240.0
