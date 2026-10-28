# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed-base G1 Inspire hands manipulating a shoe at a packing workstation."""

from isaaclab_newton.sim.schemas import MujocoRigidBodyCfg
from isaaclab_visualizers.newton import NewtonGLBackendCfg

import isaaclab.envs.mdp as env_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import clone, configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.visualizers import VisualizerCfg

from isaaclab_assets.robots.unitree import G1_INSPIRE_FTP_CFG

from . import mdp
from .g1_factory_scene import PACKING_TABLE_HEIGHT, spawn_packing_workstation
from .g1_inspire_hand import INDEX_CLOSED, PAD_OFFSETS, TCP_BODY_OFFSETS, THUMB_PITCH, THUMB_YAW, spawn_g1_shoelace
from .mdp.collision_ik_cfg import G1CollisionAwareIKActionCfg
from .mdp.inspire_hand_cfg import InspireFingerActionCfg
from .mdp.rewards import binary_gripper_switch_rate
from .shoelace_assets import ShoelaceUsdCfg
from .shoelace_env_cfg import CABLE_CFGS, ObservationsCfg, RewardsCfg, ShoelaceEnvCfg, ShoelaceSceneCfg, TerminationsCfg

_POLICY_DEFAULTS = ObservationsCfg.PolicyCfg()

_ARM_JOINTS = r"{side}_(shoulder_.*|elbow|wrist_.*)_joint"
_ARM_MIRROR_SIGNS = {
    "shoulder_pitch_joint": 1.0,
    "shoulder_roll_joint": -1.0,
    "shoulder_yaw_joint": -1.0,
    "elbow_joint": 1.0,
    "wrist_roll_joint": -1.0,
    "wrist_pitch_joint": 1.0,
    "wrist_yaw_joint": -1.0,
}
# Finger targets and near-tip surface landmarks from a constrained fit around a 2.8 mm target.
# The nominal closed surface midpoint defines a fixed wrist TCP; actual grasp scoring stays live.
_INDEX_OPEN = tuple(value - 0.25 for value in INDEX_CLOSED)
_INDEX_STROKES = tuple(closed - opened for closed, opened in zip(INDEX_CLOSED, _INDEX_OPEN, strict=True))
_FINGER_CFGS = tuple(
    SceneEntityCfg("robot", joint_names=[joint], body_names=[body])
    for joint, body in (
        ("L_thumb_proximal_pitch_joint", "L_thumb_distal"),
        ("L_index_proximal_joint", "L_index_intermediate"),
        ("R_thumb_proximal_pitch_joint", "R_thumb_distal"),
        ("R_index_proximal_joint", "R_index_intermediate"),
    )
)
# Closed-pad midpoint measured with the robot USD's forward kinematics, in each wrist frame.
_ROBOT_CFGS = tuple(SceneEntityCfg("robot", body_names=[f"{side}_wrist_yaw_link"]) for side in ("left", "right"))


def _robot_cfg() -> ArticulationCfg:
    """Configure fixed-base G1 with explicit MJWarp gravity compensation."""
    robot = clone(G1_INSPIRE_FTP_CFG)
    robot.actuators["hands"].stiffness = 50.0
    robot.spawn = ShoelaceUsdCfg(
        func=spawn_g1_shoelace,
        usd_path=robot.spawn.usd_path,
        rigid_props={"/.*": [robot.spawn.rigid_props, MujocoRigidBodyCfg(gravcomp=1.0)]},
        articulation_props=robot.spawn.articulation_props,
        fix_root_link=True,
        make_uninstanceable=True,
        friction_overrides={"/[LR]_(thumb|index)_[^/]+/.*": 90.0},
        collision_props={
            "/.*": [sim_utils.UsdPhysicsCollisionCfg(collision_enabled=False)],
            "/[LR]_(thumb|index)_[^/]+/.*": [sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)],
        },
    )
    # Preserve the trained hand-to-shoe geometry at the native tabletop height.
    robot.init_state.pos = (0.0, -0.38, PACKING_TABLE_HEIGHT + 0.015)
    robot.init_state.rot = (0.0, 0.0, 0.7071067811865476, 0.7071067811865476)
    # Mirror the left pregrasp through the robot's sagittal plane. The original left
    # pose was fitted 60 mm above its nominal tail; asymmetric tails need not align equally.
    robot.init_state.joint_pos = {
        ".*(hip|knee|ankle|waist).*": 0.0,
        "[LR]_(middle|ring|pinky)_.*_joint": 0.0,
    }
    left_arm_positions = {
        "shoulder_pitch_joint": -0.012014,
        "shoulder_roll_joint": 0.083006,
        "shoulder_yaw_joint": -0.022565,
        "elbow_joint": -0.125478,
        "wrist_roll_joint": -0.105378,
        "wrist_pitch_joint": 0.257075,
        "wrist_yaw_joint": -0.155942,
    }
    for joint, position in left_arm_positions.items():
        # Reflection reverses axial rotations about X/Z; pitch and elbow rotate about Y.
        robot.init_state.joint_pos[f"left_{joint}"] = position
        robot.init_state.joint_pos[f"right_{joint}"] = _ARM_MIRROR_SIGNS[joint] * position
    for arm, prefix in enumerate(("L", "R")):
        robot.init_state.joint_pos.update(_thumb_targets(prefix, THUMB_PITCH[arm], THUMB_YAW[arm]))
        robot.init_state.joint_pos[f"{prefix}_index_proximal_joint"] = _INDEX_OPEN[arm]
        robot.init_state.joint_pos[f"{prefix}_index_intermediate_joint"] = 0.0
    return robot


def _arm_action(side: str, arm: int) -> env_mdp.DifferentialInverseKinematicsActionCfg:
    """Control the nominal closed pinch point rigidly attached to the wrist."""
    return G1CollisionAwareIKActionCfg(
        side=side,
        asset_name="robot",
        joint_names=[_ARM_JOINTS.format(side=side)],
        body_name=f"{side}_wrist_yaw_link",
        controller=DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=True, ik_method="dls", ik_params={"lambda_val": 0.05}
        ),
        scale=(0.003, 0.003, 0.003, 0.01, 0.01, 0.01),
        body_offset=env_mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=TCP_BODY_OFFSETS[arm][0]),
    )


def _thumb_targets(prefix: str, pitch: float, yaw: float) -> dict[str, float]:
    """Hold the thumb at its pinch pose with straight intermediate and distal joints."""
    return {
        f"{prefix}_thumb_proximal_yaw_joint": yaw,
        f"{prefix}_thumb_proximal_pitch_joint": pitch,
        f"{prefix}_thumb_intermediate_joint": 0.0,
        f"{prefix}_thumb_distal_joint": 0.0,
    }


def _pinch_action(prefix: str, arm: int) -> InspireFingerActionCfg:
    """Open/close only the index while holding all thumb joints at the calibrated pinch pose."""
    opened = _thumb_targets(prefix, THUMB_PITCH[arm], THUMB_YAW[arm])
    closed = opened.copy()
    opened.update({f"{prefix}_index_proximal_joint": _INDEX_OPEN[arm], f"{prefix}_index_intermediate_joint": 0.0})
    closed.update({f"{prefix}_index_proximal_joint": INDEX_CLOSED[arm], f"{prefix}_index_intermediate_joint": 0.0})
    return InspireFingerActionCfg(
        asset_name="robot",
        joint_names=[f"{prefix}_(thumb|index)_.*_joint"],
        open_command_expr=opened,
        close_command_expr=closed,
    )


@configclass
class G1ShoelaceSceneCfg(ShoelaceSceneCfg):
    """One humanoid and a kinematic shoe on a visual packing workstation."""

    robot_left = None
    robot_right = None
    robot = _robot_cfg()
    packing_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PackingTable",
        spawn=sim_utils.UsdFileCfg(
            func=spawn_packing_workstation,
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            make_uninstanceable=True,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.10, 0.0)),
    )

    def __post_init__(self) -> None:
        """Raise the complete shoe assembly and bind both hands' contact bodies."""
        self.shoelace_asset.init_state.pos = (0.0, 0.0, PACKING_TABLE_HEIGHT)
        self.shoe.init_state.pos = (0.0, 0.0, PACKING_TABLE_HEIGHT)
        self.finger_tail_contacts.robot_prim_names = ("Robot", "Robot")
        self.finger_tail_contacts.finger_body_names = (
            ("L_thumb_distal", "L_index_intermediate"),
            ("R_thumb_distal", "R_index_intermediate"),
        )


@configclass
class G1ShoelaceActionsCfg:
    """Two Cartesian arms and two index commands with fixed thumb targets, totalling 14 actions."""

    left_arm = _arm_action("left", 0)
    left_gripper = _pinch_action("L", 0)
    right_arm = _arm_action("right", 1)
    right_gripper = _pinch_action("R", 1)


@configclass
class G1ShoelaceObservationsCfg(ObservationsCfg):
    """Expose actual thumb/index configurations, target errors and uncapped pad-to-tail vectors."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        # Keep the pretrained G1 concatenation order independent of the Franka policy.
        left_joint_pos = clone(_POLICY_DEFAULTS.left_joint_pos)
        right_joint_pos = clone(_POLICY_DEFAULTS.right_joint_pos)
        left_joint_vel = clone(_POLICY_DEFAULTS.left_joint_vel)
        right_joint_vel = clone(_POLICY_DEFAULTS.right_joint_vel)
        finger_tracking_error = ObservationTermCfg(
            func=f"{__package__}.mdp.inspire_hand:inspire_finger_tracking_error",
            params={
                "finger_cfgs": _FINGER_CFGS,
                "motion_ranges": (0.12, _INDEX_STROKES[0], 0.12, _INDEX_STROKES[1]),
            },
            clip=(-1.0, 1.0),
        )
        tails_to_tcp = clone(_POLICY_DEFAULTS.tails_to_tcp)
        finger_tail_signed_distance = clone(_POLICY_DEFAULTS.finger_tail_signed_distance)
        finger_tail_relative_velocity = ObservationTermCfg(
            func=mdp.finger_tail_relative_velocity,
            params={"cable_cfgs": CABLE_CFGS, "finger_cfgs": _FINGER_CFGS},
            scale=10.0,
        )
        pad_to_tail = ObservationTermCfg(
            func=mdp.pad_to_tail,
            params={"cable_cfgs": CABLE_CFGS, "finger_cfgs": _FINGER_CFGS},
            scale=10.0,
        )
        last_action = clone(_POLICY_DEFAULTS.last_action)
        finger_joint_pos = ObservationTermCfg(
            func=env_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["[LR]_(thumb|index)_.*_joint"])},
        )
        finger_joint_vel = ObservationTermCfg(
            func=env_mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["[LR]_(thumb|index)_.*_joint"])},
            scale=0.1,
        )

        enable_corruption = False
        concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class G1ShoelaceRewardsCfg(RewardsCfg):
    """Shared grasp rewards with a cost per binary hand command reversal."""

    finger_action_switch = RewardTermCfg(func=binary_gripper_switch_rate, weight=-0.002)


@configclass
class G1ShoelaceTerminationsCfg(TerminationsCfg):
    """Reject simulated arm/torso overlap that exceeds the kinematic guard's tolerance."""

    arm_body_collision = TerminationTermCfg(
        func=f"{__package__}.mdp.collision_ik:g1_arm_body_collision",
        params={"penetration_limit": 0.005},
    )


@configclass
class G1ShoelaceEnvCfg(ShoelaceEnvCfg):
    """G1 Inspire variant sharing the cable, reset, reward and bilateral success definitions.

    The pelvis is fixed while both arms and index fingers move. Each hand has one
    index open/close action; the thumb retains its pinch target and the other three fingers stay open.
    The shoe remains kinematic, as in the Franka baseline. Its X axis is the humanoid's lateral axis.
    """

    scene: G1ShoelaceSceneCfg = G1ShoelaceSceneCfg(num_envs=4, env_spacing=2.5)
    actions: G1ShoelaceActionsCfg = G1ShoelaceActionsCfg()
    observations: G1ShoelaceObservationsCfg = G1ShoelaceObservationsCfg()
    rewards: G1ShoelaceRewardsCfg = G1ShoelaceRewardsCfg()
    terminations: G1ShoelaceTerminationsCfg = G1ShoelaceTerminationsCfg()
    tcp_body_offsets = TCP_BODY_OFFSETS
    """Nominal closed pinch centres in the left/right wrist frames, independent of finger state [m]."""
    finger_body_offsets = PAD_OFFSETS
    """The same surface landmarks used by the shared grasp rewards [m]."""
    grasp_open_apertures = (0.019776298,) * 2
    """Measured left/right surface spacing at the ready pose [m]."""
    grasp_capture_std: float = 0.004
    """Tail capture width for the Inspire pinch pads in grasp rewards and success [m]."""

    def __post_init__(self) -> None:
        """Bind the shared managers to one articulation with increasing finger closure angles."""
        policy = self.observations.policy
        for side in ("left", "right"):
            joint_cfg = SceneEntityCfg("robot", joint_names=[_ARM_JOINTS.format(side=side)])
            getattr(policy, f"{side}_joint_pos").params["asset_cfg"] = joint_cfg
            getattr(policy, f"{side}_joint_vel").params["asset_cfg"] = clone(joint_cfg)
            getattr(self.events, f"reset_{side}_arm").params["asset_cfg"] = clone(joint_cfg)
        # The left reset samples the original noise distribution; the following right
        # reset mirrors that sample instead of independently perturbing the other arm.
        self.events.reset_right_arm.func = f"{__package__}.mdp.events:mirror_arm_joints"
        self.events.reset_right_arm.params = {
            "source_cfg": SceneEntityCfg(
                "robot", joint_names=[f"left_{joint}" for joint in _ARM_MIRROR_SIGNS], preserve_order=True
            ),
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[f"right_{joint}" for joint in _ARM_MIRROR_SIGNS], preserve_order=True
            ),
            "mirror_signs": tuple(_ARM_MIRROR_SIGNS.values()),
        }
        policy.tails_to_tcp.params["robot_cfgs"] = tuple(clone(cfg) for cfg in _ROBOT_CFGS)
        self.rewards.dense_task.params["approach_robot_cfgs"] = tuple(clone(cfg) for cfg in _ROBOT_CFGS)
        for term in (
            self.rewards.pregrasp,
            self.rewards.dense_task,
            self.rewards.grasp_hold,
            self.terminations.success,
        ):
            term.params["finger_cfgs"] = tuple(clone(cfg) for cfg in _FINGER_CFGS)
        self.events.reset_shoe.params["position_range"] = {"x": (-0.01, 0.01), "y": (-0.01, 0.01)}
        self.sim.physics.solver_cfg.iterations = 8
        self.sim.physics.solver_cfg.entries[0].bodies = [r"{ENV_REGEX_NS}/(Robot|ShoelaceScene/Shoe)"]
        self.sim.visualizer_cfgs = [
            VisualizerCfg(
                backend=NewtonGLBackendCfg(),
                eye=(1.6, 1.8, PACKING_TABLE_HEIGHT + 0.72),
                lookat=(0.0, 0.0, PACKING_TABLE_HEIGHT - 0.08),
                focal_length=20.0,
                max_visible_envs=1,
                background_color=(0.20, 0.24, 0.29),
            )
        ]
