# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ANYmal-D flat-terrain configurations and model inputs for learned-dynamics data collection and deployment."""

from __future__ import annotations

import math

import newton
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg
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
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils import configclass

from isaaclab_tasks.core.velocity.config.anymal_d.flat_env_cfg import AnymalDFlatEnvCfg

from ..heading import HeadingJointStateCodec, RootGravityEncoder
from ..recording import CONTROL_KEY
from ..signature import isaaclab_canonical_view

ANYMAL_D_CONTINUITY = "off"
"""Continuity mode of a deployed ANYmal-D artifact.

The PhysicsNeMo Robotics joint codec normalizes the free-root quaternion again when writing a prediction,
so the written state can differ from the solver's own record by one floating-point unit and ``"verify"``
would reject it. Isaac Lab reports every environment reset through the solver reset mask, so turning the
check off does not hide resets.
"""

ANYMAL_D_CONTACT_KEY = "contact/anymal_d_feet"
"""Trajectory field of the fixed foot-probe tokens."""

ANYMAL_D_CONTACT_CHANNELS = ("point", "normal", "lever_arm", "gap", "relative_velocity", "contact_state")
"""Fixed-probe channels given to the model, as in the PhysicsNeMo Robotics G1 application."""

ANYMAL_D_MODEL_CONFIG: dict[str, object] = {
    "num_heads": 10,
    "head_dim": 64,
    "intermediate_size": 896,
    "head_hidden": 192,
    "temporal_attention_spans": [4, 3, 3, 3],
    "rms_norm_eps": 1.0e-6,
    "linear_probe": True,
}
"""ParallelNeRD (FlashNeRD) geometry of the G1 model in the FlashNeRD main-results study."""

_FOOT_BODIES = ("LF_FOOT", "LH_FOOT", "RF_FOOT", "RH_FOOT")
_BASE_BODY = "base"


@configclass
class AnymalDFlatTeacherEnvCfg(AnymalDFlatEnvCfg):
    """ANYmal-D velocity tracking on flat ground with a learned-solver-compatible scene.

    The physics step stays Isaac Lab's 5 ms with ``decimation = 4``, because the ANYdrive actuator network
    runs once per physics step and is only accurate at that rate. MuJoCo Warp uses its internal contacts.
    The configuration removes everything a learned solver cannot reproduce:

    * the contact-force sensor, with the rewards and termination that read it;
    * events that change model properties (material, mass, center of mass) or apply external wrenches.

    A fallen robot is terminated by its base orientation instead of a base contact. Every environment
    owns its ground plane, so the fixed foot probes of :func:`anymal_d_foot_contacts` can query it.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
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
        # sensors and terms a learned solver cannot provide
        self.scene.contact_forces = None
        self.rewards.feet_air_time = None
        self.rewards.undesired_contacts = None
        self.terminations.base_contact = None
        self.terminations.bad_orientation = TerminationTermCfg(
            func=mdp.bad_orientation, params={"limit_angle": 0.5 * math.pi}
        )
        # events that change the trained model or apply external wrenches
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class AnymalDExplorationEnvCfg(AnymalDFlatTeacherEnvCfg):
    """ANYmal-D teacher with the task-independent data protocol of FlashNeRD.

    Episodes start from randomized base poses, base velocities, and joint configurations, last up to
    100 control periods of 20 ms (2 s), and keep falls instead of terminating them. One environment step is
    one 5 ms physics step, as recording requires, so pair it with
    :class:`~isaaclab_physicsnemo.UniformRandomActions` with ``high=1.5`` and ``hold_steps=4``: the
    joint-position action scales by 0.5 rad, which gives targets within 0.75 rad of the default pose,
    resampled every 20 ms. The Newton model equals that of :class:`AnymalDFlatTeacherEnvCfg`, so artifacts
    trained on this data deploy there.
    """

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 1
        self.sim.render_interval = 1
        task_independent_episodes(self)


def task_independent_episodes(cfg: AnymalDFlatTeacherEnvCfg) -> None:
    """Apply the FlashNeRD task-independent locomotion data protocol to an ANYmal teacher configuration.

    Episodes start from randomized base poses, base velocities, and joint configurations, last up to 2 s,
    and keep falls instead of terminating them. The Newton model is unchanged.

    Args:
        cfg: Teacher configuration, modified in place.
    """
    cfg.episode_length_s = 2.0
    cfg.terminations.bad_orientation = None
    cfg.events.reset_base.params = {
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
    cfg.events.reset_robot_joints.func = mdp.reset_joints_by_offset
    cfg.events.reset_robot_joints.params = {"position_range": (-0.5, 0.5), "velocity_range": (-1.0, 1.0)}


def anymal_d_foot_contacts(model: newton.Model) -> FixedRigidContactEncoder:
    """Build the four fixed foot-sphere probes of the FlashNeRD ANYmal locomotion model.

    Each probe is the center of a foot collision sphere and queries the environment's ground plane. Probe
    geometry is expressed in the base heading frame, so it does not depend on the world position or yaw.

    Args:
        model: Finalized Newton model of an ANYmal-D scene built from :class:`AnymalDFlatTeacherEnvCfg`.

    Returns:
        The probe encoder, whose tokens are recorded under :data:`ANYMAL_D_CONTACT_KEY`.

    Raises:
        ValueError: If a foot has no single collision sphere or an environment has no single ground plane.
    """
    body_world = model.body_world.numpy()
    world_bodies = [body for body in range(model.body_count) if body_world[body] == 0]
    body_names = [model.body_label[body].rsplit("/", 1)[-1] for body in world_bodies]
    shape_world, shape_body, shape_type = model.shape_world.numpy(), model.shape_body.numpy(), model.shape_type.numpy()
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
        raise ValueError("ANYmal-D probes need one collision sphere per foot and one ground plane per environment.")
    # The probe geometry fingerprint must match between the recording, training, and deployment scenes.
    with isaaclab_canonical_view(model):
        body_codec = BodyStateCodec(model, model_frame=BodyHeadingFrame(body=body_names.index(_BASE_BODY)))
        return FixedRigidContactEncoder(
            model,
            body_codec,
            points=expand_rigid_contact_shapes(model, body_codec, [slots[0] for slots in feet]),
            environment_shape_slots=ground,
            environment_geometry="per_world",
            key=ANYMAL_D_CONTACT_KEY,
        )


def anymal_d_state_codec(model: newton.Model) -> HeadingJointStateCodec:
    """Express the ANYmal-D joint state in the base heading frame, as the G1 application does.

    Args:
        model: Finalized Newton model of an ANYmal-D scene built from :class:`AnymalDFlatTeacherEnvCfg`.
    """
    return HeadingJointStateCodec(model)


def anymal_d_dynamics_features(
    codec: HeadingJointStateCodec, contacts: FixedRigidContactEncoder
) -> CompositeFeatureEncoder:
    """Condition the model on applied joint efforts, base-frame gravity, and the fixed foot-probe features.

    This is the input recipe of the PhysicsNeMo Robotics G1 application with the four foot-sphere probes
    of the FlashNeRD ANYmal experiment.

    Args:
        codec: State codec from :func:`anymal_d_state_codec`.
        contacts: Probe encoder from :func:`anymal_d_foot_contacts`.

    Returns:
        The feature encoder of the artifact.
    """
    return CompositeFeatureEncoder(
        ControlFieldEncoder("joint_f", per_world_shape=(codec.layout.dof_qd,), key=CONTROL_KEY),
        RootGravityEncoder(codec),
        FixedContactFeatureAdapter(contacts, ANYMAL_D_CONTACT_CHANNELS),
    )
