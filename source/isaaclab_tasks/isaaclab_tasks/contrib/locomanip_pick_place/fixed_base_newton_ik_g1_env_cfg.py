# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed-base G1 upper-body task solving IK with Newton instead of Pink.

Prototype for moving the teleoperation IK off the CPU. Pink solves a QP on one core and measures
at 9.9 ms per step on this scene, against 11.2 ms for the whole physics step, so it is roughly
half the CPU-side work on the XR critical path. Newton solves on GPU instead.

The Newton IK action term supports fixed-base articulations only
(``NewtonInverseKinematicsAction`` raises otherwise), which is why this prototype targets the
fixed-base task rather than the locomanipulation one.
"""

from __future__ import annotations

from isaaclab_newton.ik.newton_ik_objectives_cfg import NewtonIKJointLimitObjectiveCfg, NewtonIKPoseObjectiveCfg
from isaaclab_newton.ik.newton_ik_solver_cfg import NewtonIKSolverCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.sim.spawners.materials import NewtonMaterialCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from .configs.pink_controller_cfg import G1_UPPER_BODY_IK_ACTION_CFG
from .fixed_base_upper_body_ik_g1_env_cfg import FixedBaseUpperBodyIKG1EnvCfg, FixedBaseUpperBodyIKG1SceneCfg
from .locomanipulation_g1_env_cfg import (
    _PACKING_TABLE_COLLIDER_POS,
    _PACKING_TABLE_COLLIDER_SIZE,
    _newton_object_spawn,
)
from .newton_ik_teleop_actions import TeleopNewtonInverseKinematicsActionCfg

_IK_JOINT_NAMES = list(G1_UPPER_BODY_IK_ACTION_CFG.pink_controlled_joint_names)
"""Joints the IK solves for, matching the Pink configuration so the two are comparable."""

_ROBOT_CONTACT_MATERIAL = [
    UsdPhysicsRigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
    NewtonMaterialCfg(contact_stiffness=1.0e6, contact_damping=2000.0),
]
"""Contact material for the robot.

Newton resolves an omitted friction value to zero, so without this the hands have no friction at
all and cannot hold anything. Authored directly rather than through a preset: preset branches
resolve by the selected preset name, so a ``preset(default=None, newton_mjwarp=...)`` silently
yields ``None`` under any other selector.
"""

_WRIST_OFFSET_ROT = (0.70710678, 0.0, 0.70710678, 0.0)
"""Fixed rotation applied to each wrist objective, as Warp ``xyzw``.

The controller frame and the wrist link frame have their x and z axes exchanged, which shows up
as roll and yaw being swapped on the hand. Exchanging two axes is a half turn about the diagonal
between them, here ``(1, 0, 1)/sqrt(2)``.
"""

_HAND_JOINT_NAMES = list(G1_UPPER_BODY_IK_ACTION_CFG.hand_joint_names)
"""Hand joints. Pink passes these through rather than solving them, so they stay a direct write."""


@configclass
class PhysicsCfg(PresetCfg):
    """Physics presets. Newton IK requires the Newton backend, which the base task does not set."""

    isaacsim_physx = PhysxCfg()
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=300,
            nconmax=200,
            cone="pyramidal",
            iterations=100,
            ls_iterations=15,
            use_mujoco_contacts=False,
        ),
        num_substeps=2,
    )
    default = newton_mjwarp


@configclass
class NewtonIKActionsCfg:
    """Newton IK over the arms and waist, with the hand joints written directly."""

    upper_body_ik = TeleopNewtonInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=_IK_JOINT_NAMES,
        # Each solver iteration costs about 4.3 ms at one environment, where the solve is bound by
        # kernel-launch latency rather than compute. The config default of 24 measures 117 ms
        # per step against Pink's 19; four keeps the step usable for a subjective comparison.
        controller=NewtonIKSolverCfg(optimizer="lm", jacobian_mode="analytic", iterations=4),
        objectives=[
            NewtonIKPoseObjectiveCfg(
                name="left_wrist",
                body_name="left_wrist_yaw_link",
                body_offset_rot=_WRIST_OFFSET_ROT,
                command_type="pose",
                use_relative_mode=False,
            ),
            NewtonIKPoseObjectiveCfg(
                name="right_wrist",
                body_name="right_wrist_yaw_link",
                body_offset_rot=_WRIST_OFFSET_ROT,
                command_type="pose",
                use_relative_mode=False,
            ),
            NewtonIKJointLimitObjectiveCfg(),
        ],
    )

    hand_joints = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=_HAND_JOINT_NAMES,
        # The retargeter emits these in the order the list declares them. Without this the term
        # resolves them in articulation order instead and the finger targets land on the wrong
        # joints, so the grip does nothing.
        preserve_order=True,
        use_default_offset=False,
    )


@configclass
class _FixedBaseNewtonSceneCfg(FixedBaseUpperBodyIKG1SceneCfg):
    """Adds the tabletop collider Newton needs.

    ``packing_table.usd`` authors its collider as a ``boundingCube`` ``PhysicsCollisionAPI`` on an
    Xform rather than on mesh prims, so Newton emits no shape for it and the object falls through.
    This reproduces the same bounding volume as an invisible static box.
    """

    packing_table_collider: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PackingTableCollider",
        init_state=AssetBaseCfg.InitialStateCfg(pos=list(_PACKING_TABLE_COLLIDER_POS)),
        spawn=sim_utils.CuboidCfg(
            size=_PACKING_TABLE_COLLIDER_SIZE,
            visible=False,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )


@configclass
class _FixedBaseNewtonPhysicsG1EnvCfg(FixedBaseUpperBodyIKG1EnvCfg):
    """Fixed-base G1 on the Newton backend, keeping Pink IK."""

    scene: _FixedBaseNewtonSceneCfg = _FixedBaseNewtonSceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=True
    )

    def __post_init__(self):
        super().__post_init__()
        self.sim.physics = PhysicsCfg()
        # Newton's body labels keep the asset's intermediate grouping prim, so the task's
        # PhysX-style pattern full-matches nothing and sensor init fails.
        for side in ("left", "right"):
            sensor = getattr(self.scene, f"{side}_hand_contact")
            sensor.prim_path = "{ENV_REGEX_NS}/Robot/" + f"{side}_hand/{side}_hand_[^/]*_link"
        # The authored steering wheel does not import under MJWarp: one decomposition piece is
        # rejected as "mesh volume is too small". The graspable primitive stands in, as it does
        # in the other Newton tasks.
        self.scene.object.spawn = _newton_object_spawn()
        # Newton gives an unauthored material zero friction; the hands would grip nothing.
        robot_spawn = self.scene.robot.spawn.copy()
        robot_spawn.physics_material = _ROBOT_CONTACT_MATERIAL
        self.scene.robot.spawn = robot_spawn


@configclass
class FixedBasePinkIKG1EnvCfg(_FixedBaseNewtonPhysicsG1EnvCfg):
    """Pink IK on Newton physics. Baseline for the Newton IK comparison."""


@configclass
class FixedBaseNewtonIKG1EnvCfg(_FixedBaseNewtonPhysicsG1EnvCfg):
    """Fixed-base G1 upper-body task with the IK solved by Newton."""

    def __post_init__(self):
        # Run the base first: it authors the Pink controller's URDF path, which a Newton solver
        # config has no field for. Swapping the action term afterwards sidesteps that entirely.
        super().__post_init__()
        self.actions = NewtonIKActionsCfg()
        # Newton IK takes targets in the robot's base frame, where Pink took world-frame poses and
        # rebased them itself. Teleop can emit them already rebased.
        self.isaac_teleop.target_frame_prim_path = "/World/envs/env_0/Robot/pelvis"
