# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the dexterous hand from Shadow Robot.

The following configurations are available:

* :obj:`SHADOW_HAND_CFG`: Shadow Hand with implicit actuator model.
* :obj:`SHADOW_HAND_MENAGERIE_CFG`: Mujoco Menagerie conversion, MuJoCo physics variant (Newton/MJWarp).
* :obj:`SHADOW_HAND_MENAGERIE_PHYSX_CFG`: Mujoco Menagerie conversion, PhysX physics variant.

Reference:

* https://www.shadowrobot.com/dexterous-hand-series/

"""

import os
from collections.abc import Callable
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from pxr import Usd

MENAGERIE_ASSET_ROOT = os.environ.get(
    "MENAGERIE_ASSET_ROOT", "omniverse://isaac-dev.ov.nvidia.com/Isaac/Samples/Mujoco_Menagerie"
)
"""Root of the Mujoco Menagerie asset conversions. Override with the ``MENAGERIE_ASSET_ROOT``
environment variable to point at a local mirror when the Nucleus server is unreachable."""

##
# Configuration
##

SHADOW_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/ShadowRobot/ShadowHand/shadow_hand_instanceable.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
        fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(limit_stiffness=30.0, damping=0.1),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        rot=(0.0, -0.7071, 0.7071, 0.0),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["robot0_WR.*", "robot0_(FF|MF|RF|LF|TH)J(3|2|1)", "robot0_(LF|TH)J4", "robot0_THJ0"],
            effort_limit_sim={
                "robot0_WRJ1": 4.785,
                "robot0_WRJ0": 2.175,
                "robot0_(FF|MF|RF|LF)J1": 0.7245,
                "robot0_FFJ(3|2)": 0.9,
                "robot0_MFJ(3|2)": 0.9,
                "robot0_RFJ(3|2)": 0.9,
                "robot0_LFJ(4|3|2)": 0.9,
                "robot0_THJ4": 2.3722,
                "robot0_THJ3": 1.45,
                "robot0_THJ(2|1)": 0.99,
                "robot0_THJ0": 0.81,
            },
            stiffness={
                "robot0_WRJ.*": 5.0,
                "robot0_(FF|MF|RF|LF|TH)J(3|2|1)": 1.0,
                "robot0_(LF|TH)J4": 1.0,
                "robot0_THJ0": 1.0,
            },
            damping={
                "robot0_WRJ.*": 0.5,
                "robot0_(FF|MF|RF|LF|TH)J(3|2|1)": 0.1,
                "robot0_(LF|TH)J4": 0.1,
                "robot0_THJ0": 0.1,
            },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Shadow Hand robot."""


SHADOW_HAND_MENAGERIE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MENAGERIE_ASSET_ROOT}/shadow_hand/right_hand/right_hand.usda",
        # The Menagerie exports default to the "physx" variant; Newton/MJWarp needs the
        # "mujoco" variant, which composes the MjcTendon distal couplings.
        variants={"Physics": "mujoco"},
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force", ensure_drives_exist=True),
        fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(damping=0.1),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        # Same compensation as the ShadowHandNewton reference: Newton's import bakes the root
        # body's USD orientation into the root fixed joint, so it must be re-applied at spawn.
        rot=(0.0, 0.0, -0.70710678118, 0.70710678118),
        joint_pos={".*": 0.0},
    ),
    actuators={
        # Derived from the ShadowHandNewton reference actuator table via the naming map:
        # fingers keep their index (robot0_FFJn -> rh_FFJn), thumb/wrist shift by +1
        # (robot0_THJn -> rh_THJ{n+1}, robot0_WRJn -> rh_WRJ{n+1}).
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["rh_WR.*", "rh_(FF|MF|RF|LF)J(3|2|1)", "rh_LFJ4", "rh_THJ.*"],
            effort_limit_sim={
                "rh_WRJ2": 4.785,
                "rh_WRJ1": 2.175,
                "rh_(FF|MF|RF|LF)J1": 0.7245,
                "rh_FFJ(3|2)": 0.9,
                "rh_MFJ(3|2)": 0.9,
                "rh_RFJ(3|2)": 0.9,
                "rh_LFJ(4|3|2)": 0.9,
                "rh_THJ5": 2.3722,
                "rh_THJ4": 1.45,
                "rh_THJ(3|2)": 0.99,
                "rh_THJ1": 0.81,
            },
            stiffness={
                "rh_WRJ.*": 5.0,
                "rh_(FF|MF|RF|LF)J(3|2|1)": 1.0,
                "rh_LFJ4": 1.0,
                "rh_THJ.*": 1.0,
            },
            damping={
                "rh_WRJ.*": 0.5,
                "rh_(FF|MF|RF|LF)J(3|2|1)": 0.1,
                "rh_LFJ4": 0.1,
                "rh_THJ.*": 0.1,
            },
            friction=1e-2,
            armature=2e-3,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of the Mujoco Menagerie Shadow Hand (right), MuJoCo physics variant.

Intended for the Newton/MJWarp backend. Physics content matches the
``ShadowHandNewton`` reference asset exactly (joints, limits, masses, tendons);
only the joint/body naming differs.
"""


def _author_shadow_hand_fixed_tendons(prim: "Usd.Prim") -> None:
    """Author the PhysX fixed tendons that the Menagerie ``physx`` layer omits.

    Replicates the legacy ``ShadowHand`` asset's distal-middle finger couplings: per finger,
    tendon length ``-0.00805 * theta_middle + 0.00705 * theta_distal`` constrained to
    ``+/-0.001`` around rest length 0, so the distal joint tracks the middle joint.
    """
    from pxr import Sdf, Usd

    joints = {p.GetName(): p for p in Usd.PrimRange(prim) if p.GetName().startswith("rh_")}
    for finger in ("FF", "MF", "RF", "LF"):
        tendon = f"rh_T_{finger}J1c"
        root_joint = joints[f"rh_{finger}J2"]
        axis_joint = joints[f"rh_{finger}J1"]
        root_joint.AddAppliedSchema(f"PhysxTendonAxisRootAPI:{tendon}")
        for attr, type_name, value in (
            # All gains zero: parity with the legacy asset's RUNTIME values (its configured
            # FixedTendonPropertiesCfg never lands due to the instanceable-prim issue, so the
            # baseline trains with zero-gain tendons). A stiff coupling (limitStiffness 30)
            # drags coupled joints to their limits against the weak position drives.
            ("limitStiffness", Sdf.ValueTypeNames.Float, 0.0),
            ("damping", Sdf.ValueTypeNames.Float, 0.0),
            ("stiffness", Sdf.ValueTypeNames.Float, 0.0),
            ("restLength", Sdf.ValueTypeNames.Float, 0.0),
            ("lowerLimit", Sdf.ValueTypeNames.Float, -0.001),
            ("upperLimit", Sdf.ValueTypeNames.Float, 0.001),
            ("gearing", Sdf.ValueTypeNames.FloatArray, [-0.00805]),
        ):
            root_joint.CreateAttribute(f"physxTendon:{tendon}:{attr}", type_name).Set(value)
        axis_joint.AddAppliedSchema(f"PhysxTendonAxisAPI:{tendon}")
        axis_joint.CreateAttribute(f"physxTendon:{tendon}:gearing", Sdf.ValueTypeNames.FloatArray).Set([0.00705])


def spawn_shadow_hand_menagerie_physx(
    prim_path: str,
    cfg: "ShadowHandMenageriePhysxSpawnCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> "Usd.Prim":
    """Spawn the Menagerie Shadow Hand and author the missing PhysX fixed tendons.

    The MuJoCo USD Converter does not translate MJCF tendon couplings into PhysX tendon
    schemas, so the ``physx`` variant leaves the four distal finger joints uncoupled and
    undriven. This spawner authors them post-spawn until the converter gains support.
    """
    # Deferred imports: this module is imported by the task-config registry before the
    # simulation app starts, when pxr and the spawner internals are not yet loadable.
    from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
    from isaaclab.sim.utils import clone

    @clone
    def _spawn(prim_path, cfg, translation=None, orientation=None, **inner_kwargs):
        prim = _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)
        _author_shadow_hand_fixed_tendons(prim)
        return prim

    return _spawn(prim_path, cfg, translation, orientation, **kwargs)


@configclass
class ShadowHandMenageriePhysxSpawnCfg(sim_utils.UsdFileCfg):
    """Spawn configuration for the Menagerie Shadow Hand on PhysX with authored fixed tendons."""

    func: Callable = spawn_shadow_hand_menagerie_physx


SHADOW_HAND_MENAGERIE_PHYSX_CFG = ArticulationCfg(
    spawn=ShadowHandMenageriePhysxSpawnCfg(
        usd_path=f"{MENAGERIE_ASSET_ROOT}/shadow_hand/right_hand/right_hand.usda",
        variants={"Physics": "physx"},
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
        # Tunes the tendons authored by the spawner, matching the legacy asset's runtime values.
        fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(limit_stiffness=30.0, damping=0.1),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        # Maps the Menagerie native frame onto the legacy asset's spawned pose under the
        # task's PhysX preset (fingers -Y beneath the falling cube), derived from the
        # measured legacy fingertip layout.
        rot=(-0.0061, -0.0054, -0.6743, 0.7384),
        joint_pos={".*": 0.0},
    ),
    actuators={
        # Derived from SHADOW_HAND_CFG (legacy 0-indexed naming) via the naming map:
        # all joints shift by +1 (robot0_FFJn -> rh_FFJ{n+1}, ..., robot0_WRJn -> rh_WRJ{n+1}).
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["rh_WR.*", "rh_(FF|MF|RF|LF)J(4|3|2)", "rh_LFJ5", "rh_THJ.*"],
            # The legacy asset bakes a 5729.58 deg/s joint velocity limit; the Menagerie
            # layers author none, which destabilizes PhysX.
            velocity_limit_sim=100.0,
            effort_limit_sim={
                "rh_WRJ2": 4.785,
                "rh_WRJ1": 2.175,
                "rh_(FF|MF|RF|LF)J2": 0.7245,
                "rh_FFJ(4|3)": 0.9,
                "rh_MFJ(4|3)": 0.9,
                "rh_RFJ(4|3)": 0.9,
                "rh_LFJ(5|4|3)": 0.9,
                "rh_THJ5": 2.3722,
                "rh_THJ4": 1.45,
                "rh_THJ(3|2)": 0.99,
                "rh_THJ1": 0.81,
            },
            stiffness={
                "rh_WRJ.*": 5.0,
                "rh_(FF|MF|RF|LF)J(4|3|2)": 1.0,
                "rh_LFJ5": 1.0,
                "rh_THJ.*": 1.0,
            },
            damping={
                "rh_WRJ.*": 0.5,
                "rh_(FF|MF|RF|LF)J(4|3|2)": 0.1,
                "rh_LFJ5": 0.1,
                "rh_THJ.*": 0.1,
            },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of the Mujoco Menagerie Shadow Hand (right), PhysX physics variant.

.. note::
    The Menagerie PhysX variant carries no fixed-tendon authoring (converter gap), so the
    spawner authors the four distal-middle couplings from the legacy asset's template. See
    :func:`spawn_shadow_hand_menagerie_physx`.
"""
