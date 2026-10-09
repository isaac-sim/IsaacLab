# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ANYmal-C configurations of the FlashNeRD locomotion experiment, with joint-target control at 20 ms."""

from __future__ import annotations

from physicsnemo_robotics.dynamics import (
    CompositeFeatureEncoder,
    ControlFieldEncoder,
    FixedContactFeatureAdapter,
    FixedRigidContactEncoder,
)

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.anymal import ANYMAL_C_CFG

from ..heading import HeadingJointStateCodec, RootGravityEncoder
from ..recording import JOINT_TARGET_KEY
from .anymal_d import (
    ANYMAL_D_CONTACT_CHANNELS,
    ANYMAL_D_MODEL_CONFIG,
    AnymalDFlatTeacherEnvCfg,
    task_independent_episodes,
)

ANYMAL_C_MODEL_CONFIG: dict[str, object] = {**ANYMAL_D_MODEL_CONFIG, "temporal_attention_spans": [4, 4, 4, 4]}
"""ParallelNeRD geometry of the FlashNeRD study's G1 model with the 13-frame receptive field of its ANYmal model."""


@configclass
class AnymalCFlatTeacherEnvCfg(AnymalDFlatTeacherEnvCfg):
    """ANYmal-C velocity tracking on flat ground, stepped by MuJoCo Warp at 5 ms with a 20 ms control period.

    The ANYdrive actuator network runs at every 5 ms physics step. The joint-torque penalty is removed,
    because a learned model that receives joint-position targets does not expose the torques, so policies
    trained in both physics optimize the same reward.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.rewards.dof_torques_l2 = None


@configclass
class AnymalCExplorationEnvCfg(AnymalCFlatTeacherEnvCfg):
    """ANYmal-C teacher with the FlashNeRD task-independent locomotion data protocol.

    Record it with ``TrajectoryDatasetRecorder(control_targets=env.scene["robot"])`` and
    :class:`~isaaclab_physicsnemo.UniformRandomActions` with ``high=1.5``, which draws targets within 0.75 rad
    of the default pose every 20 ms control period.
    """

    def __post_init__(self):
        super().__post_init__()
        task_independent_episodes(self)


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


def anymal_target_dynamics_features(
    codec: HeadingJointStateCodec, contacts: FixedRigidContactEncoder
) -> CompositeFeatureEncoder:
    """Condition the model on the joint-position targets, base-frame gravity, and the fixed foot probes.

    Args:
        codec: Heading-frame state codec of the robot.
        contacts: Foot-probe encoder from :func:`~isaaclab_physicsnemo.tasks.anymal_d_foot_contacts`.

    Returns:
        The feature encoder of the artifact.
    """
    return CompositeFeatureEncoder(
        ControlFieldEncoder("joint_target_q", per_world_shape=(codec.layout.dof_q,), key=JOINT_TARGET_KEY),
        RootGravityEncoder(codec),
        FixedContactFeatureAdapter(contacts, ANYMAL_D_CONTACT_CHANNELS),
    )
