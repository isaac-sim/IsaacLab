# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ANYmal-C scenes and model inputs of the FlashNeRD locomotion experiment, with joint-target control at 20 ms."""

from __future__ import annotations

import math

import newton
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg
from isaaclab_physicsnemo import (
    JOINT_TARGET_KEY,
    PARALLEL_NERD_CONFIG,
    HeadingJointStateCodec,
    RootGravityEncoder,
    isaaclab_canonical_view,
)
from physicsnemo_robotics.dynamics import (
    BodyHeadingFrame,
    BodyStateCodec,
    CompositeFeatureEncoder,
    ControlFieldEncoder,
    FixedContactFeatureAdapter,
    FixedRigidContactEncoder,
    expand_rigid_contact_shapes,
)

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils import configclass

from isaaclab_tasks.core.velocity.config.anymal_d.flat_env_cfg import AnymalDFlatEnvCfg

from isaaclab_assets.robots.anymal import ANYMAL_C_CFG

ANYMAL_CONTINUITY = "off"
"""Continuity mode of a deployed ANYmal artifact.

The PhysicsNeMo Robotics joint codec normalizes the free-root quaternion again when writing a prediction,
so the written state can differ from the solver's own record by one floating-point unit and ``"verify"``
would reject it. Isaac Lab reports every environment reset through the solver reset mask, so turning the
check off does not hide resets.
"""

ANYMAL_CONTACT_KEY = "contact/anymal_d_feet"
"""Trajectory field of the fixed foot-probe tokens, named as in the datasets recorded so far."""

ANYMAL_CONTACT_CHANNELS = ("point", "normal", "lever_arm", "gap", "relative_velocity", "contact_state")
"""Fixed-probe channels given to the model, as in the PhysicsNeMo Robotics G1 application."""

ANYMAL_MODEL_CONFIG: dict[str, object] = {
    **PARALLEL_NERD_CONFIG,
    "num_heads": 10,
    "intermediate_size": 896,
    "head_hidden": 192,
    "temporal_attention_spans": [4, 4, 4, 4],
}
"""ParallelNeRD geometry of the FlashNeRD study's G1 model with the 13-frame receptive field of its ANYmal model."""

_FOOT_BODIES = ("LF_FOOT", "LH_FOOT", "RF_FOOT", "RH_FOOT")
_BASE_BODY = "base"


@configclass
class AnymalCFlatTeacherEnvCfg(AnymalDFlatEnvCfg):
    """ANYmal-C velocity tracking on flat ground, stepped by MuJoCo Warp at 5 ms with a 20 ms control period.

    The ANYdrive actuator network runs at every 5 ms physics step, the rate it is accurate at. The scene
    removes everything a learned solver cannot reproduce:

    * the contact-force sensor, with the rewards and termination that read it, so a fallen robot is
      terminated by its base orientation instead;
    * events that change model properties (material, mass, center of mass) or apply external wrenches;
    * the joint-torque penalty, because a model that receives joint-position targets does not expose the
      torques, so policies trained in both physics optimize the same reward.

    Every environment owns its ground plane, so the fixed foot probes of :func:`anymal_foot_contacts` can
    query it.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.sim.physics = NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=60, nconmax=25, cone="elliptic", impratio=100.0, integrator="implicitfast"
            ),
            num_substeps=1,
            use_cuda_graph=False,
            deterministic_mode="not_guaranteed",
            default_shape_cfg=NewtonShapeCfg(margin=0.0, ke=160000.0, kd=1100.0),
        )
        # PhysicsNeMo Robotics fixed probes query per-world surfaces, so each environment gets its own plane.
        ground_material = self.scene.terrain.physics_material
        self.scene.terrain = None
        self.scene.ground = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Ground",
            spawn=sim_utils.GroundPlaneCfg(
                size=(self.scene.env_spacing, self.scene.env_spacing), physics_material=ground_material
            ),
        )
        self.scene.contact_forces = None
        self.rewards.feet_air_time = None
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2 = None
        self.terminations.base_contact = None
        self.terminations.bad_orientation = TerminationTermCfg(
            func=mdp.bad_orientation, params={"limit_angle": 0.5 * math.pi}
        )
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class AnymalCExplorationEnvCfg(AnymalCFlatTeacherEnvCfg):
    """ANYmal-C teacher with the FlashNeRD task-independent locomotion data protocol.

    Episodes start from randomized base poses, base velocities, and joint configurations, last up to 100
    control periods (2 s), and keep falls. Record it with
    ``TrajectoryDatasetRecorder(control_targets=env.scene["robot"])`` and
    :class:`~isaaclab_physicsnemo.UniformRandomActions` with ``high=1.5``: the joint-position action scales by
    0.5 rad, which gives targets within 0.75 rad of the default pose every control period. The Newton model
    equals that of :class:`AnymalCFlatTeacherEnvCfg`.
    """

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 2.0
        self.terminations.bad_orientation = None
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.1, 0.1),
                "roll": (-0.3, 0.3),
                "pitch": (-0.3, 0.3),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (-0.5, 0.5),
                "roll": (-1.0, 1.0),
                "pitch": (-1.0, 1.0),
                "yaw": (-1.0, 1.0),
            },
        }
        self.events.reset_robot_joints.func = mdp.reset_joints_by_offset
        self.events.reset_robot_joints.params = {"position_range": (-0.5, 0.5), "velocity_range": (-1.0, 1.0)}


@configclass
class AnymalCTargetControlEnvCfg(AnymalCFlatTeacherEnvCfg):
    """ANYmal-C scene for a learned model of the 20 ms control period, which consumes joint-position targets.

    One physics step is one control period, and the actuators only forward their targets to Newton's
    ``control.joint_target_q``: zero-gain implicit drives leave the Newton model identical to the teacher's.
    """

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = 0.02
        self.decimation = 1
        self.sim.render_interval = 1
        self.scene.robot.actuators = {"legs": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0)}


def anymal_foot_contacts(model: newton.Model) -> FixedRigidContactEncoder:
    """Build the four fixed foot-sphere probes of the FlashNeRD ANYmal locomotion model.

    Each probe is the center of a foot collision sphere and queries the environment's ground plane. Probe
    geometry is expressed in the base heading frame, so it does not depend on the world position or yaw.

    Args:
        model: Finalized Newton model of a scene built from :class:`AnymalCFlatTeacherEnvCfg`.

    Returns:
        The probe encoder, whose tokens are recorded under :data:`ANYMAL_CONTACT_KEY`.

    Raises:
        ValueError: If the model has no body or shape layout, a foot has no single collision sphere, or an
            environment has no single ground plane.
    """
    body_world, shape_world = model.body_world, model.shape_world
    shape_body, shape_type = model.shape_body, model.shape_type
    if body_world is None or shape_world is None or shape_body is None or shape_type is None:
        raise ValueError("The Newton model has no body or shape layout.")
    body_world = body_world.numpy()
    world_bodies = [body for body in range(model.body_count) if body_world[body] == 0]
    body_names = [model.body_label[body].rsplit("/", 1)[-1] for body in world_bodies]
    shape_world, shape_body, shape_type = shape_world.numpy(), shape_body.numpy(), shape_type.numpy()
    world_shapes = [shape for shape in range(model.shape_count) if shape_world[shape] == 0]

    def shape_slots(body: int, geometry: newton.GeoType) -> list[int]:
        return [
            slot
            for slot, shape in enumerate(world_shapes)
            if shape_body[shape] == body and shape_type[shape] == geometry
        ]

    feet = [shape_slots(world_bodies[body_names.index(name)], newton.GeoType.SPHERE) for name in _FOOT_BODIES]
    ground = shape_slots(-1, newton.GeoType.PLANE)
    if any(len(slots) != 1 for slots in feet) or len(ground) != 1:
        raise ValueError("ANYmal probes need one collision sphere per foot and one ground plane per environment.")
    # The probe geometry fingerprint must match between the recording, training, and deployment scenes.
    with isaaclab_canonical_view(model):
        body_codec = BodyStateCodec(model, model_frame=BodyHeadingFrame(body=body_names.index(_BASE_BODY)))
        return FixedRigidContactEncoder(
            model,
            body_codec,
            points=expand_rigid_contact_shapes(model, body_codec, [slots[0] for slots in feet]),
            environment_shape_slots=ground,
            environment_geometry="per_world",
            key=ANYMAL_CONTACT_KEY,
        )


def anymal_dynamics_features(
    codec: HeadingJointStateCodec, contacts: FixedRigidContactEncoder
) -> CompositeFeatureEncoder:
    """Condition the model on the joint-position targets, base-frame gravity, and the fixed foot probes.

    This is the input recipe of the PhysicsNeMo Robotics G1 application with the joint-position targets and
    the four foot-sphere probes of the FlashNeRD ANYmal experiment.

    Args:
        codec: Heading-frame state codec of the robot.
        contacts: Probe encoder from :func:`anymal_foot_contacts`.

    Returns:
        The feature encoder of the artifact.
    """
    return CompositeFeatureEncoder(
        ControlFieldEncoder("joint_target_q", per_world_shape=(codec.layout.dof_q,), key=JOINT_TARGET_KEY),
        RootGravityEncoder(codec),
        FixedContactFeatureAdapter(contacts, ANYMAL_CONTACT_CHANNELS),
    )
