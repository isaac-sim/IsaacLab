# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bimanual dVRK PSM configuration for absolute world-frame needle pass."""

import math

import numpy as np
from isaaclab_teleop import IsaacTeleopCfg

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots.dvrk import (
    DVRK_PSM_ARM_JOINT_NAMES,
    DVRK_PSM_CFG,
    DVRK_PSM_JAW_CLOSED_POS,
    DVRK_PSM_JAW_JOINT_NAMES,
    DVRK_PSM_JAW_OPEN_POS,
    DVRK_PSM_TOOL_TIP_BODY_NAME,
)

from ... import mdp
from ...needle_pass_env_cfg import HANDOFF_PHASE_CFG, NeedlePassEnvCfg

# ``isaaclab_teleop`` defers the optional ``isaacteleop`` import until session
# startup.  This marker lets the environment test suite classify the task as a
# teleoperation environment without making ordinary task discovery depend on
# the external package.
_TELEOP_AVAILABLE = True

LEFT_PSM_ROOT_POS = (-0.149189, -0.124189, 0.161400)
RIGHT_PSM_ROOT_POS = (0.149189, -0.124189, 0.161400)
PSM_ROOT_ROT_XYZW = (0.0, 0.0, 0.0, 1.0)

# Symmetric homes place both tools in the shared hand-off region.  These values
# are kept with the root placements so retargeter and articulation homes cannot
# drift independently.
LEFT_TOOL_HOME_POS_W = (-0.025, 0.0, 0.060)
RIGHT_TOOL_HOME_POS_W = (0.025, 0.0, 0.060)
LEFT_TOOL_HOME_ROT_XYZW = (0.29235514, -0.40562109, 0.13872189, 0.85484281)
RIGHT_TOOL_HOME_ROT_XYZW = (0.29235514, 0.40562109, -0.13872189, 0.85484281)

LEFT_WORKSPACE_LOWER = (-0.18, -0.16, 0.015)
LEFT_WORKSPACE_UPPER = (0.08, 0.16, 0.20)
RIGHT_WORKSPACE_LOWER = (-0.08, -0.16, 0.015)
RIGHT_WORKSPACE_UPPER = (0.18, 0.16, 0.20)

LEFT_ARM_HOME = (0.886077124, -0.659058036, 0.200, 0.0, 0.0, 0.0)
RIGHT_ARM_HOME = (-0.886077124, -0.659058036, 0.200, 0.0, 0.0, 0.0)

# The two grasp poses below are ``needle_channel_*`` (``T_N_C``) rows emitted
# by Isaac Sim 5.1's native antipodal grasp generator, with quaternions
# reordered from the generator's wxyz output to Isaac Lab's xyzw convention.
# ``N`` is the scaled needle body and ``C`` is the generated channel frame,
# whose +Z axis is the jaw-gap axis.  These are not the mesh-local
# ``native_*`` fields: the asset's nested authored transform separates those
# frames by about 19 mm.  Candidate selection never interpolated or
# geometrically altered poses; the only component reordering is the convention
# conversion above.  The donor was selected only after a fixed native candidate
# was physically closed, settled with gravity enabled, and retained through a
# fixed tool disturbance.
ISAAC_GRASP_GENERATOR_EXTENSION = "isaacsim.replicator.grasping"
ISAAC_GRASP_GENERATOR_EXTENSION_VERSION = "1.0.9"
ISAAC_GRASP_GENERATOR_API = "isaacsim.replicator.grasping.GraspingManager.generate_grasp_poses"
ISAAC_GRASP_GENERATOR_SIM_VERSION = "5.1.0"
ISAAC_GRASP_GENERATOR_SEED = 12
ISAAC_GRASP_GENERATOR_CANDIDATE_COUNT = 8192
ISAAC_GRASP_GENERATOR_ORIENTATIONS_PER_CENTRE = 32
ISAAC_GRASP_GENERATOR_CENTRE_COUNT = 256
ISAAC_GRASP_ASSET_SHA256 = "2b317a61f93631a7192e7ed2839ef20f7a75c05aa5f84a3905696134a64f36d7"
ISAAC_GRASP_WRAPPER_SHA256 = "01bc820d1777a1655a5c42b3ebac997c6281335a12d12f7636c3e25721f3a2d5"
ISAAC_GRASP_CONFIG_SHA256 = "b308ec31bf9bf425c686007e0dc0ad72f09ae7e1f67e1015ac53dc92a017e798"
ISAAC_GRASP_CANDIDATES_SHA256 = "7c601982d72759ca901fad9b59fa1df80a092221d1cb91eda88938b2b83bc374"
ISAAC_GRASP_MANIFEST_SHA256 = "13c72a5fb58db7c211619b72dcbdf27890a25a35a5aa8e3185ab4ae3139970ee"

DONOR_GRASP_CANDIDATE_INDEX = 2321
DONOR_GRASP_T_N_C_POS_M = (
    0.0003148044879752905,
    0.0033030783449336226,
    0.0003504408852082775,
)
DONOR_GRASP_T_N_C_ROT_XYZW = (
    -0.06216530304406737,
    -0.299876591345807,
    0.06093660132734946,
    0.9499980187763187,
)
DONOR_GRASP_CONTACT_POINTS_N_M = (
    (0.0008196471425340884, 0.0032317539769975326, -0.0003599608352924837),
    (-0.00019003816658350742, 0.0033744027128697148, 0.0010608426057090398),
)
DONOR_GRASP_OUTWARD_NORMALS_N = (
    (0.5773406198878056, -0.08156690886849895, -0.812419010120518),
    (-0.6585541989377149, 0.12276505562071868, 0.7424520915048636),
)

# Exact emitted candidate selected for the receiver trial.  It is not a pose
# perturbation or interpolation.
RECEIVER_GRASP_CANDIDATE_INDEX = 51
RECEIVER_GRASP_T_N_C_POS_M = (
    0.0023141994825170917,
    -0.009712701723612324,
    0.0004488201068876367,
)
RECEIVER_GRASP_T_N_C_ROT_XYZW = (
    -0.01452245833926486,
    0.36639395824675103,
    0.38287722060063434,
    0.8479089570874903,
)

# Fixed collision-channel calibration ``T_T_C`` in psm_tool_tip_link.  Its
# origin is the measured midpoint of the two active jaw collision volumes,
# not the tool-tip-link origin.  The reset and receiver targets are
# deterministic compositions of this calibration with generated ``T_N_C``
# poses and the measured donor-held acquisition pose; focused tests reconstruct
# all identities.
DVRK_JAW_CHANNEL_T_T_C_POS_M = (0.0, 0.0, 0.004)
DVRK_JAW_CHANNEL_T_T_C_ROT_XYZW = (
    0.0,
    0.7071067811865476,
    0.0,
    0.7071067811865476,
)

# The native transform is the acquisition seed.  The public reset must start
# held, so it uses the gravity-on physical equilibrium below instead of a
# collision-free geometric approximation.  This state has no attachment: it
# was measured after real jaw closure and is requalified by the CUDA test.
DONOR_GRASP_NATIVE_SEED_POS = (-0.02676631338705744, -0.00554576569408158, 0.06096145185775791)
DONOR_GRASP_NATIVE_SEED_ROT_XYZW = (
    0.04784599008904877,
    0.5946084191654624,
    0.24809474119226224,
    0.7632827686095777,
)
NEEDLE_RESET_POS = (-0.024928808212280273, -0.0031707286834716797, 0.05836881697177887)
NEEDLE_RESET_ROT_XYZW = (
    0.028613094240427017,
    0.6356675028800964,
    0.2438839077949524,
    0.7318665981292725,
)
# Candidate 51 was acquired against this donor-held pose after the fixed
# gravity-on reset settling trace.  Keeping the measured acquisition pose
# explicit makes the fixed controller target reproducible without moving the
# free needle or hiding a state write in the action path.
RECEIVER_ACQUISITION_NEEDLE_POS_W = (-0.0245427893868402, -0.0031316915911458786, 0.058761484034582104)
RECEIVER_ACQUISITION_NEEDLE_ROT_XYZW = (
    0.0413061008969846,
    0.613496097694513,
    0.24468171703916028,
    0.7496980735529877,
)
# Measured candidate-51 receiver-frame equilibrium after a guarded release
# from the donor.  It is an acceptance target only and is never written into
# the simulation.
RECEIVER_NEEDLE_TARGET_POS_T = (0.0018065175972878933, 0.008040599524974823, -0.0016274424269795418)
RECEIVER_NEEDLE_TARGET_ROT_XYZW = (
    -0.25603383779525757,
    0.3513960838317871,
    -0.31044724583625793,
    0.8453344702720642,
)
RECEIVER_TOOL_TARGET_POS_W = (-0.02371715009212494, -0.008205749094486237, 0.052002355456352234)
RECEIVER_TOOL_TARGET_ROT_XYZW = (
    0.4864428639411926,
    0.3236384689807892,
    0.2469031661748886,
    0.7730914950370789,
)

# The reset writes the observed equilibrium joint positions, then the normal
# action path drives both donor jaws fully closed.  Separating state from drive
# target preserves the physically settled hold rather than forcing an
# interpenetrating fully-closed configuration at reset.
DONOR_HELD_RESET_JAW_POS = (-0.20328494906425476, 0.003166106529533863)
DONOR_GRASP_JAW_POS = DVRK_PSM_JAW_CLOSED_POS
DONOR_GRASP_CLOSEDNESS = 1.0

# The load-qualified donor acquisition used these bounded drives, derived from
# the Large Needle Driver reflected jaw inertia and specified torque/speed
# limits.  The task uses the same drives for reset retention and teleoperation.
DVRK_NEEDLE_PASS_JAW_REFLECTED_INERTIA_KG_M2 = 3.32e-7
# This setting is under deterministic CUDA end-to-end qualification.  Torque
# and velocity limits remain fixed; 150 rad/s is the bounded midpoint between
# the stable-but-under-retained 120 rad/s setting and the unstable 200 rad/s
# setting.
DVRK_NEEDLE_PASS_JAW_NATURAL_FREQUENCY_RAD_S = 150.0
DVRK_NEEDLE_PASS_JAW_DAMPING_RATIO = 1.0
DVRK_NEEDLE_PASS_JAW_EFFORT_LIMIT_N_M = 0.16
DVRK_NEEDLE_PASS_JAW_VELOCITY_LIMIT_RAD_S = 2.1
DVRK_NEEDLE_PASS_JAW_ACTUATOR = ImplicitActuatorCfg(
    joint_names_expr=list(DVRK_PSM_JAW_JOINT_NAMES),
    stiffness=DVRK_NEEDLE_PASS_JAW_REFLECTED_INERTIA_KG_M2 * DVRK_NEEDLE_PASS_JAW_NATURAL_FREQUENCY_RAD_S**2,
    damping=(
        2.0
        * DVRK_NEEDLE_PASS_JAW_DAMPING_RATIO
        * DVRK_NEEDLE_PASS_JAW_REFLECTED_INERTIA_KG_M2
        * DVRK_NEEDLE_PASS_JAW_NATURAL_FREQUENCY_RAD_S
    ),
    effort_limit_sim=DVRK_NEEDLE_PASS_JAW_EFFORT_LIMIT_N_M,
    velocity_limit_sim=DVRK_NEEDLE_PASS_JAW_VELOCITY_LIMIT_RAD_S,
)

DVRK_HANDOFF_PHASE_CFG = HANDOFF_PHASE_CFG.replace(
    # Candidate 51's measured receiver reaction axes remain 20.6 degrees from
    # perfectly opposed during the guarded transfer.  The 25-degree gate keeps
    # 4.4 degrees of geometric margin while still requiring bilateral load and
    # dwell before any donor opening command may pass.
    opposed_normal_tolerance_rad=math.radians(25.0),
    receiver_relative_position_target_m=RECEIVER_NEEDLE_TARGET_POS_T,
    receiver_relative_orientation_target_xyzw=RECEIVER_NEEDLE_TARGET_ROT_XYZW,
    receiver_relative_position_limit_m=0.003,
    receiver_relative_orientation_limit_rad=math.radians(15.0),
)


def _joint_home(arm_home: tuple[float, ...], jaw_home: tuple[float, float]) -> dict[str, float]:
    return {
        **dict(zip(DVRK_PSM_ARM_JOINT_NAMES, arm_home, strict=True)),
        **dict(zip(DVRK_PSM_JAW_JOINT_NAMES, jaw_home, strict=True)),
    }


def _psm_cfg(
    prim_path: str,
    root_pos: tuple[float, float, float],
    arm_home: tuple[float, ...],
    jaw_home: tuple[float, float],
):
    return DVRK_PSM_CFG.replace(
        prim_path=prim_path,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=root_pos,
            rot=PSM_ROOT_ROT_XYZW,
            joint_pos=_joint_home(arm_home, jaw_home),
            joint_vel={".*": 0.0},
        ),
        actuators={**DVRK_PSM_CFG.actuators, "jaws": DVRK_NEEDLE_PASS_JAW_ACTUATOR},
    )


def _tool_home_transform(
    position: tuple[float, float, float], orientation_xyzw: tuple[float, float, float, float]
) -> np.ndarray:
    """Build the world-frame homogeneous tool-home transform."""
    quaternion = np.asarray(orientation_xyzw, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm <= np.finfo(np.float64).eps:
        raise ValueError("dVRK tool-home orientation must be a finite non-zero quaternion")
    x, y, z, w = quaternion / norm

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    )
    transform[:3, 3] = position
    return transform


def _build_dvrk_needle_pass_pipeline():
    """Build the bimanual world-frame dVRK IsaacTeleop pipeline.

    The flattened output follows the task action declaration exactly:
    ``left pose (7), left jaws (2), right pose (7), right jaws (2)``.
    IsaacTeleop supplies ``world_T_anchor`` at runtime, so both controller
    streams, tool homes, and workspace bounds remain in the shared simulation
    world frame expected by the task's world-frame differential-IK actions.
    """
    try:
        from isaacteleop.retargeters import (
            DVRKPSMClutchConfig,
            DVRKPSMClutchRetargeter,
            DVRKPSMGripperConfig,
            DVRKPSMGripperRetargeter,
            TensorReorderer,
        )
    except ImportError as exc:
        raise RuntimeError(
            "dVRK needle-pass teleoperation requires the dVRK retargeters from NVIDIA/IsaacTeleop PR 769; "
            "until they are released, run scripts/tools/install_isaacteleop_pr769_for_tests.sh"
        ) from exc
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    side_configs = {
        "left": {
            "controller": ControllersSource.LEFT,
            "home_position": LEFT_TOOL_HOME_POS_W,
            "home_orientation": LEFT_TOOL_HOME_ROT_XYZW,
            "workspace_lower": LEFT_WORKSPACE_LOWER,
            "workspace_upper": LEFT_WORKSPACE_UPPER,
            "initial_closedness": DONOR_GRASP_CLOSEDNESS,
        },
        "right": {
            "controller": ControllersSource.RIGHT,
            "home_position": RIGHT_TOOL_HOME_POS_W,
            "home_orientation": RIGHT_TOOL_HOME_ROT_XYZW,
            "workspace_lower": RIGHT_WORKSPACE_LOWER,
            "workspace_upper": RIGHT_WORKSPACE_UPPER,
            "initial_closedness": 0.0,
        },
    }
    for side, side_cfg in side_configs.items():
        if not all(
            lower <= home <= upper
            for lower, home, upper in zip(
                side_cfg["workspace_lower"], side_cfg["home_position"], side_cfg["workspace_upper"], strict=True
            )
        ):
            raise ValueError(f"{side} dVRK tool home must lie inside its world workspace")

    controllers = ControllersSource(name="controllers")
    world_transform = ValueInput("world_T_anchor", TransformMatrix())
    world_controllers = controllers.transformed(world_transform.output(ValueInput.VALUE))

    connected_outputs = {}
    for side, side_cfg in side_configs.items():
        controller_key = side_cfg["controller"]
        controller_output = world_controllers.output(controller_key)

        clutch = DVRKPSMClutchRetargeter(
            DVRKPSMClutchConfig(
                input_device=controller_key,
                home_reference_T_ee=_tool_home_transform(side_cfg["home_position"], side_cfg["home_orientation"]),
                workspace_lower=side_cfg["workspace_lower"],
                workspace_upper=side_cfg["workspace_upper"],
                translation_scale=1.0,
                orientation_offset=(0.0, 0.0, 0.0, 1.0),
                clutch_threshold=0.5,
            ),
            name=f"{side}_ee_pose",
        )
        gripper = DVRKPSMGripperRetargeter(
            DVRKPSMGripperConfig(
                input_device=controller_key,
                jaw_open=DVRK_PSM_JAW_OPEN_POS,
                jaw_closed=DVRK_PSM_JAW_CLOSED_POS,
                initial_closedness=side_cfg["initial_closedness"],
                clutch_threshold=0.5,
                trigger_deadband=0.05,
                opening_intent_duration_s=0.12,
            ),
            name=f"{side}_jaws",
        )
        connected_outputs[f"{side}_pose"] = clutch.connect({controller_key: controller_output}).output(
            DVRKPSMClutchRetargeter.OUTPUT_POSE
        )
        connected_outputs[f"{side}_jaws"] = gripper.connect({controller_key: controller_output}).output(
            DVRKPSMGripperRetargeter.OUTPUT_JAW_TARGETS
        )

    left_pose_elements = [
        "left_pos_x",
        "left_pos_y",
        "left_pos_z",
        "left_quat_x",
        "left_quat_y",
        "left_quat_z",
        "left_quat_w",
    ]
    left_jaw_elements = ["left_jaw_1", "left_jaw_2"]
    right_pose_elements = [
        "right_pos_x",
        "right_pos_y",
        "right_pos_z",
        "right_quat_x",
        "right_quat_y",
        "right_quat_z",
        "right_quat_w",
    ]
    right_jaw_elements = ["right_jaw_1", "right_jaw_2"]
    reorderer = TensorReorderer(
        input_config={
            "left_pose": left_pose_elements,
            "left_jaws": left_jaw_elements,
            "right_pose": right_pose_elements,
            "right_jaws": right_jaw_elements,
        },
        output_order=left_pose_elements + left_jaw_elements + right_pose_elements + right_jaw_elements,
        name="action_reorderer",
        input_types={
            "left_pose": "array",
            "left_jaws": "array",
            "right_pose": "array",
            "right_jaws": "array",
        },
    )
    connected_reorderer = reorderer.connect(connected_outputs)
    return OutputCombiner({"action": connected_reorderer.output("output")})


@configclass
class DVRKNeedlePassEnvCfg(NeedlePassEnvCfg):
    """Donor-held dVRK needle hand-off with motion-controller input and an 18D action ABI."""

    requires_cuda: bool = True
    """The contact-qualified dVRK needle pass is supported only on CUDA PhysX."""

    def __post_init__(self):
        super().__post_init__()

        if not (
            DVRK_PSM_JAW_OPEN_POS[0] < DVRK_PSM_JAW_CLOSED_POS[0] <= 0.0
            and DVRK_PSM_JAW_OPEN_POS[1] > DVRK_PSM_JAW_CLOSED_POS[1] >= 0.0
        ):
            raise ValueError("dVRK ordered jaw endpoints do not satisfy the paired-jaw contract")
        if not (
            DVRK_PSM_JAW_OPEN_POS[0] <= DONOR_HELD_RESET_JAW_POS[0] <= DVRK_PSM_JAW_CLOSED_POS[0]
            and DVRK_PSM_JAW_OPEN_POS[1] >= DONOR_HELD_RESET_JAW_POS[1] >= DVRK_PSM_JAW_CLOSED_POS[1]
            and DONOR_GRASP_JAW_POS == DVRK_PSM_JAW_CLOSED_POS
            and DONOR_GRASP_CLOSEDNESS == 1.0
        ):
            raise ValueError("donor reset must use a valid held equilibrium and a fully closed jaw target")

        self.scene.left_psm = _psm_cfg(
            "{ENV_REGEX_NS}/LeftPSM",
            LEFT_PSM_ROOT_POS,
            LEFT_ARM_HOME,
            DONOR_HELD_RESET_JAW_POS,
        )
        self.scene.right_psm = _psm_cfg(
            "{ENV_REGEX_NS}/RightPSM",
            RIGHT_PSM_ROOT_POS,
            RIGHT_ARM_HOME,
            DVRK_PSM_JAW_OPEN_POS,
        )
        self.scene.needle.init_state = RigidObjectCfg.InitialStateCfg(
            pos=NEEDLE_RESET_POS,
            rot=NEEDLE_RESET_ROT_XYZW,
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        )

        phase_terms = (
            self.events.reset_all,
            self.observations.policy.handoff_phase,
            self.observations.subtask_terms.donor_hold,
            self.observations.subtask_terms.co_hold,
            self.observations.subtask_terms.receiver_only_hold,
            self.observations.subtask_terms.retained_lift,
            self.rewards.phase_progress,
            self.rewards.retained_lift,
            self.terminations.success,
            self.terminations.needle_dropped_or_out_of_bounds,
        )
        for term in phase_terms:
            term.params = {**term.params, "phase_cfg": DVRK_HANDOFF_PHASE_CFG}

        self.actions.left_arm_action = mdp.WorldFrameDifferentialInverseKinematicsActionCfg(
            asset_name="left_psm",
            joint_names=list(DVRK_PSM_ARM_JOINT_NAMES),
            body_name=DVRK_PSM_TOOL_TIP_BODY_NAME,
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
            ),
            scale=1.0,
        )
        self.actions.left_jaw_action = mdp.DonorReleaseGuardedPairedJawJointPositionActionCfg(
            asset_name="left_psm",
            joint_names=list(DVRK_PSM_JAW_JOINT_NAMES),
            scale=1.0,
            offset=0.0,
            use_default_offset=False,
            preserve_order=True,
            phase_cfg=DVRK_HANDOFF_PHASE_CFG,
            # A release is a genuine opening request.  Before a measured
            # co-hold it is clamped to the load-qualified donor grasp.
            release_aperture_threshold_rad=0.0,
            # Preserve the load-qualified donor-held reset.  The interlock
            # blocks any outward donor-jaw motion; deliberate further closing
            # remains a normal actuator command.
            hold_jaw_pos=DONOR_GRASP_JAW_POS,
        )
        self.actions.right_arm_action = mdp.WorldFrameDifferentialInverseKinematicsActionCfg(
            asset_name="right_psm",
            joint_names=list(DVRK_PSM_ARM_JOINT_NAMES),
            body_name=DVRK_PSM_TOOL_TIP_BODY_NAME,
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
                # The recipient traverses a native grasp channel near the PSM
                # wrist singularity.  Lower damping preserves the bounded
                # public differential-IK solve while allowing it to converge
                # to the generated pose instead of stalling millimetres away.
                ik_params={"lambda_val": 0.003},
            ),
            scale=1.0,
        )
        self.actions.right_jaw_action = mdp.PairedJawJointPositionActionCfg(
            asset_name="right_psm",
            joint_names=list(DVRK_PSM_JAW_JOINT_NAMES),
            scale=1.0,
            offset=0.0,
            use_default_offset=False,
            preserve_order=True,
        )

        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=_build_dvrk_needle_pass_pipeline,
            sim_device=self.sim.device,
            xr_cfg=self.xr,
        )


__all__ = [
    "DVRKNeedlePassEnvCfg",
    "DVRK_HANDOFF_PHASE_CFG",
    "DVRK_JAW_CHANNEL_T_T_C_POS_M",
    "DVRK_JAW_CHANNEL_T_T_C_ROT_XYZW",
    "DONOR_GRASP_CANDIDATE_INDEX",
    "DONOR_GRASP_CLOSEDNESS",
    "DONOR_GRASP_CONTACT_POINTS_N_M",
    "DONOR_GRASP_JAW_POS",
    "DONOR_GRASP_NATIVE_SEED_POS",
    "DONOR_GRASP_NATIVE_SEED_ROT_XYZW",
    "DONOR_GRASP_OUTWARD_NORMALS_N",
    "DONOR_GRASP_T_N_C_POS_M",
    "DONOR_GRASP_T_N_C_ROT_XYZW",
    "DONOR_HELD_RESET_JAW_POS",
    "DVRK_NEEDLE_PASS_JAW_ACTUATOR",
    "ISAAC_GRASP_ASSET_SHA256",
    "ISAAC_GRASP_CANDIDATES_SHA256",
    "ISAAC_GRASP_CONFIG_SHA256",
    "ISAAC_GRASP_GENERATOR_API",
    "ISAAC_GRASP_GENERATOR_CANDIDATE_COUNT",
    "ISAAC_GRASP_GENERATOR_CENTRE_COUNT",
    "ISAAC_GRASP_GENERATOR_EXTENSION",
    "ISAAC_GRASP_GENERATOR_EXTENSION_VERSION",
    "ISAAC_GRASP_GENERATOR_ORIENTATIONS_PER_CENTRE",
    "ISAAC_GRASP_GENERATOR_SEED",
    "ISAAC_GRASP_GENERATOR_SIM_VERSION",
    "ISAAC_GRASP_MANIFEST_SHA256",
    "ISAAC_GRASP_WRAPPER_SHA256",
    "LEFT_ARM_HOME",
    "LEFT_PSM_ROOT_POS",
    "LEFT_TOOL_HOME_POS_W",
    "LEFT_TOOL_HOME_ROT_XYZW",
    "LEFT_WORKSPACE_LOWER",
    "LEFT_WORKSPACE_UPPER",
    "NEEDLE_RESET_POS",
    "NEEDLE_RESET_ROT_XYZW",
    "PSM_ROOT_ROT_XYZW",
    "RIGHT_ARM_HOME",
    "RIGHT_PSM_ROOT_POS",
    "RIGHT_TOOL_HOME_POS_W",
    "RIGHT_TOOL_HOME_ROT_XYZW",
    "RIGHT_WORKSPACE_LOWER",
    "RIGHT_WORKSPACE_UPPER",
    "RECEIVER_GRASP_CANDIDATE_INDEX",
    "RECEIVER_GRASP_T_N_C_POS_M",
    "RECEIVER_GRASP_T_N_C_ROT_XYZW",
    "RECEIVER_ACQUISITION_NEEDLE_POS_W",
    "RECEIVER_ACQUISITION_NEEDLE_ROT_XYZW",
    "RECEIVER_NEEDLE_TARGET_POS_T",
    "RECEIVER_NEEDLE_TARGET_ROT_XYZW",
    "RECEIVER_TOOL_TARGET_POS_W",
    "RECEIVER_TOOL_TARGET_ROT_XYZW",
]
