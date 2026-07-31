# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from dataclasses import MISSING

from isaaclab_newton.physics import (
    HydroelasticSDFCfg,
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
)
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxCollisionPropertiesCfg, PhysxRigidBodyPropertiesCfg

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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg

import isaaclab_tasks.contrib.deploy.mdp as mdp
import isaaclab_tasks.contrib.deploy.mdp.terminations as gear_assembly_terminations
from isaaclab_tasks.contrib.deploy.mdp.noise_models import ResetSampledConstantNoiseModelCfg
from isaaclab_tasks.utils import PresetCfg, preset

# Get the directory where this configuration file is located
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(CONFIG_DIR, "assets")
NEWTON_GEAR_ASSETS_DIR = os.path.join(ASSETS_DIR, "newton")

# A 256-world GPU shard reached 1.54M broad-phase pairs during randomized resets.
# Keep power-of-two headroom so reset spikes do not discard candidate contacts.
_GEAR_MAX_TRIANGLE_PAIRS = 4_194_304
_PHYSX_GEAR_OFFSETS = {
    "gear_small": [0.076125, 0.0, 0.0],
    "gear_medium": [0.030375, 0.0, 0.0],
    "gear_large": [-0.045375, 0.0, 0.0],
}
_NEWTON_GEAR_OFFSETS = {
    "gear_small": [0.0823685, 0.0, 0.0],
    "gear_medium": [0.0366185, 0.0, 0.0],
    "gear_large": [-0.0391315, 0.0, 0.0],
}


def _gear_usd_path(default_usd_path: str, asset_name: str) -> PresetCfg:
    """Create a gear USD path preset with Newton-specific collision assets.

    Args:
        default_usd_path: Factory asset USD path used by the default and PhysX presets.
        asset_name: Gear asset directory and USD stem.

    Returns:
        Preset that resolves to package-local point-SDF or hydroelastic assets for Newton collision presets.
    """
    return preset(
        default=default_usd_path,
        newton_mjwarp=os.path.join(NEWTON_GEAR_ASSETS_DIR, asset_name, f"{asset_name}.usda"),
        newton_sdf=os.path.join(NEWTON_GEAR_ASSETS_DIR, asset_name, f"{asset_name}.usda"),
        newton_hydroelastic=os.path.join(NEWTON_GEAR_ASSETS_DIR, asset_name, f"{asset_name}_hydroelastic.usda"),
    )


##
# Environment configuration
##


@configclass
class GearAssemblyPhysicsCfg(PresetCfg):
    """Physics backend presets for gear assembly.

    Gear insertion is contact-rich (gear teeth, shaft walls, gripper fingers), so the
    Newton (MuJoCo) solver limits are set conservatively. Select a preset at runtime
    with the ``presets=<name>`` CLI override:

    * ``default`` and ``physx`` -- PhysX with contact buffers sized for assembly.
    * ``newton_mjwarp`` -- Newton with MuJoCo's internal contact solver.
    * ``newton_sdf`` -- Newton's collision pipeline with reduced point-SDF contacts.
    * ``newton_hydroelastic`` -- Newton's own collision pipeline (``use_mujoco_contacts=False``)
      with SDF-based hydroelastic contacts. Produces distributed contact areas instead of
      point contacts, which can improve fidelity for the gear-teeth/shaft-wall interaction.
      More expensive; A/B test against ``newton_sdf`` before committing to it for training.

    Note:
        ``collision_cfg`` (and therefore hydroelastic contacts) is only valid when the Newton
        collision pipeline is active, i.e. ``use_mujoco_contacts=False``. Setting it alongside
        ``use_mujoco_contacts=True`` raises ``ValueError``, which is why it lives in a separate
        preset rather than ``newton_mjwarp``.
    """

    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=200,
            nconmax=100,
            impratio=10.0,
            cone="elliptic",
            iterations=100,
            ls_iterations=50,
            use_mujoco_contacts=True,
            update_data_interval=10,
        ),
        num_substeps=20,
        default_shape_cfg=NewtonShapeCfg(gap=0.005),
        collision_decimation=0,
        debug_mode=False,
    )
    newton_sdf: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=4096,
            nconmax=4096,
            impratio=10.0,
            cone="elliptic",
            iterations=100,
            ls_iterations=50,
            use_mujoco_contacts=False,
            ccd_iterations=35,
            update_data_interval=10,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(
            reduce_contacts=True,
            max_triangle_pairs=_GEAR_MAX_TRIANGLE_PAIRS,
        ),
        num_substeps=20,
        default_shape_cfg=NewtonShapeCfg(gap=0.005),
        collision_decimation=0,
        debug_mode=False,
    )
    newton_hydroelastic: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            # The hydroelastic SDF pipeline produces distributed contact areas (thousands of points
            # for a gripped concave gear), so the per-world contact/constraint buffers must be far
            # larger than the MuJoCo ``newton_mjwarp`` preset's. Sized for ~4k constraints/contacts.
            njmax=4096,
            nconmax=4096,
            impratio=10.0,
            cone="elliptic",
            iterations=100,
            ls_iterations=50,
            # Hand collision detection to Newton's pipeline so hydroelastic SDF contacts apply.
            use_mujoco_contacts=False,
            ccd_iterations=35,
            update_data_interval=10,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(
            max_triangle_pairs=_GEAR_MAX_TRIANGLE_PAIRS,
            sdf_hydroelastic_config=HydroelasticSDFCfg(
                reduce_contacts=True,
                normal_matching=True,
            ),
        ),
        num_substeps=20,
        default_shape_cfg=NewtonShapeCfg(gap=0.005),
        collision_decimation=0,
        debug_mode=False,
    )
    physx: PhysxCfg = PhysxCfg(
        # Important to prevent collisionStackSize buffer overflow in contact-rich environments.
        gpu_collision_stack_size=2**30,
        gpu_max_rigid_contact_count=2**23,
        gpu_max_rigid_patch_count=2**23,
    )
    default = physx


@configclass
class GearAssemblySceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # Replicate physics so each environment gets its own physics instance. The Newton backend
    # only creates per-environment bodies through the physics-replication path; with
    # ``replicate_physics=False`` every environment collapses onto a single physics instance
    # (root states come back shaped ``(1, ...)`` instead of ``(num_envs, ...)``). Per-environment
    # gear/base variation is applied at reset via the randomization events
    # (``write_root_pose_to_sim``), so it does not rely on USD-level authoring and is preserved.
    replicate_physics = preset(
        default=False,
        physx=False,
        newton_mjwarp=True,
        newton_sdf=True,
        newton_hydroelastic=True,
    )

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    factory_gear_base = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/FactoryGearBase",
        # TODO: change to common isaac sim directory
        spawn=sim_utils.UsdFileCfg(
            usd_path=_gear_usd_path(
                f"{ISAAC_NUCLEUS_DIR}/Props/Factory/gear_assets/factory_gear_base/factory_gear_base.usd",
                "factory_gear_base",
            ),
            activate_contact_sensors=False,
            rigid_props=PhysxRigidBodyPropertiesCfg(
                disable_gravity=False,
                kinematic_enabled=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=None),
            collision_props=PhysxCollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.0200, 0.2100, -0.1), rot=(0.0, 0.0, 0.70711, 0.70711)),
    )

    factory_gear_small = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/FactoryGearSmall",
        # TODO: change to common isaac sim directory
        spawn=sim_utils.UsdFileCfg(
            usd_path=_gear_usd_path(
                f"{ISAAC_NUCLEUS_DIR}/Props/Factory/gear_assets/factory_gear_small/factory_gear_small.usd",
                "factory_gear_small",
            ),
            activate_contact_sensors=False,
            rigid_props=PhysxRigidBodyPropertiesCfg(
                disable_gravity=False,
                kinematic_enabled=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=None),
            collision_props=PhysxCollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.0200, 0.2100, -0.1), rot=(0.0, 0.0, 0.70711, 0.70711)),
    )

    factory_gear_medium = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/FactoryGearMedium",
        # TODO: change to common isaac sim directory
        spawn=sim_utils.UsdFileCfg(
            usd_path=_gear_usd_path(
                f"{ISAAC_NUCLEUS_DIR}/Props/Factory/gear_assets/factory_gear_medium/factory_gear_medium.usd",
                "factory_gear_medium",
            ),
            activate_contact_sensors=False,
            rigid_props=PhysxRigidBodyPropertiesCfg(
                disable_gravity=False,
                kinematic_enabled=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=None),
            collision_props=PhysxCollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.0200, 0.2100, -0.1), rot=(0.0, 0.0, 0.70711, 0.70711)),
    )

    factory_gear_large = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/FactoryGearLarge",
        # TODO: change to common isaac sim directory
        spawn=sim_utils.UsdFileCfg(
            usd_path=_gear_usd_path(
                f"{ISAAC_NUCLEUS_DIR}/Props/Factory/gear_assets/factory_gear_large/factory_gear_large.usd",
                "factory_gear_large",
            ),
            activate_contact_sensors=False,
            rigid_props=PhysxRigidBodyPropertiesCfg(
                disable_gravity=False,
                kinematic_enabled=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=None),
            collision_props=PhysxCollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.0200, 0.2100, -0.1), rot=(0.0, 0.0, 0.70711, 0.70711)),
    )

    # robots
    robot: ArticulationCfg = MISSING

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )

    stand = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Stand",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
        ),
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

        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})
        joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})
        gear_shaft_pos = ObsTerm(
            func=mdp.gear_shaft_pos_w,
            params={},  # Will be populated in __post_init__
            noise=ResetSampledConstantNoiseModelCfg(
                noise_cfg=UniformNoiseCfg(n_min=-0.005, n_max=0.005, operation="add")
            ),
        )
        gear_shaft_quat = ObsTerm(func=mdp.gear_shaft_quat_w)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})
        joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})
        gear_shaft_pos = ObsTerm(func=mdp.gear_shaft_pos_w, params={})  # Will be populated in __post_init__
        gear_shaft_quat = ObsTerm(func=mdp.gear_shaft_quat_w)

        gear_pos = ObsTerm(func=mdp.gear_pos_w)
        gear_quat = ObsTerm(func=mdp.gear_quat_w)

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_gear = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.05, 0.05],
                "y": [-0.05, 0.05],
                "z": [0.1, 0.15],
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("factory_gear_small"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    end_effector_gear_keypoint_tracking = RewTerm(
        func=mdp.keypoint_entity_error,
        weight=-1.5,
        params={
            "asset_cfg_1": SceneEntityCfg("factory_gear_base"),
            "keypoint_scale": 0.15,
        },
    )

    end_effector_gear_keypoint_tracking_exp = RewTerm(
        func=mdp.keypoint_entity_error_exp,
        weight=1.5,
        params={
            "asset_cfg_1": SceneEntityCfg("factory_gear_base"),
            "kp_exp_coeffs": [(50, 0.0001), (300, 0.0001)],
            "kp_use_sum_of_exps": False,
            "keypoint_scale": 0.15,
        },
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-5.0e-06)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    gear_dropped = DoneTerm(
        func=gear_assembly_terminations.reset_when_gear_dropped,
        params={
            "distance_threshold": 0.15,  # 15cm from gripper
            "robot_asset_cfg": SceneEntityCfg("robot"),
        },
    )

    gear_orientation_exceeded = DoneTerm(
        func=gear_assembly_terminations.reset_when_gear_orientation_exceeds_threshold,
        params={
            "roll_threshold_deg": 7.0,  # Maximum roll deviation in degrees
            "pitch_threshold_deg": 7.0,  # Maximum pitch deviation in degrees
            "yaw_threshold_deg": 180.0,  # Maximum yaw deviation in degrees
            "robot_asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class GearAssemblyEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: GearAssemblySceneCfg = GearAssemblySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    # PhysX remains the backwards-compatible default; explicit Newton presets select
    # package-local SDF collision assets. See :class:`GearAssemblyPhysicsCfg`.
    sim: SimulationCfg = SimulationCfg(physics=GearAssemblyPhysicsCfg())

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.episode_length_s = 6.66
        self.viewer.eye = (3.5, 3.5, 3.5)
        # simulation settings
        self.decimation = preset(
            default=4,
            physx=4,
            newton_mjwarp=3,
            newton_sdf=3,
            newton_hydroelastic=3,
        )
        self.sim.render_interval = preset(
            default=4,
            physx=4,
            newton_mjwarp=3,
            newton_sdf=3,
            newton_hydroelastic=3,
        )
        self.sim.dt = preset(
            default=1.0 / 120.0,
            physx=1.0 / 120.0,
            newton_mjwarp=0.01,
            newton_sdf=0.01,
            newton_hydroelastic=0.01,
        )

        physx_gear_offsets = _PHYSX_GEAR_OFFSETS
        newton_gear_offsets = _NEWTON_GEAR_OFFSETS
        gear_offsets = preset(
            default=physx_gear_offsets,
            physx=physx_gear_offsets,
            newton_mjwarp=newton_gear_offsets,
            newton_sdf=newton_gear_offsets,
            newton_hydroelastic=newton_gear_offsets,
        )
        self.gear_offsets = physx_gear_offsets

        # Populate observation and reward term parameters with backend-specific shaft offsets.
        self.observations.policy.gear_shaft_pos.params["gear_offsets"] = gear_offsets
        self.observations.critic.gear_shaft_pos.params["gear_offsets"] = gear_offsets
        reward_gear_offsets = preset(
            default=None,
            physx=None,
            newton_mjwarp=newton_gear_offsets,
            newton_sdf=newton_gear_offsets,
            newton_hydroelastic=newton_gear_offsets,
        )
        self.rewards.end_effector_gear_keypoint_tracking.params["gear_offsets"] = reward_gear_offsets
        self.rewards.end_effector_gear_keypoint_tracking_exp.params["gear_offsets"] = reward_gear_offsets
