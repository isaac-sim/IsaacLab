# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""OpenArm pick-up task: grasp and lift the red cube (cube_2) with the LEFT arm only.

Action space (14D flat):
  [0:6]   left arm IK delta pose  (dx dy dz drx dry drz)
  [6]     left gripper command    (±1.0)
  [7:13]  right arm IK delta pose (kept zero during single-arm pick-up)
  [13]    right gripper command   (±1.0)

Scene: only cube_2 (red) is present — cube_1 and cube_3 are removed.

Cube randomisation: left-arm workspace only.
  Looking at the robot from the front camera (1,0,0.5):
    x = forward from robot base (depth in front of robot)
    y = left  (positive y = robot's left arm side)
  cube_2 is randomized within x:[0.17, 0.27]  y:[0.03, 0.17]
  — a compact band centred on the left arm's comfortable sweet spot.

Success condition: cube_2 centre rises above 0.20 m world-Z.
"""

import torch

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp_core
from isaaclab_tasks.manager_based.manipulation.stack import mdp

from . import stack_ik_abs_visuomotor_env_cfg


# ── Subtask / termination helpers ─────────────────────────────────────────────

def cube_is_lifted(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    min_height: float = 0.30,
) -> torch.Tensor:
    """Return 1.0 when cube centre is above *min_height* world-Z."""
    obj = env.scene[asset_cfg.name]
    return (obj.data.root_pos_w[:, 2] > min_height).unsqueeze(-1).float()


def cube_pickup_success(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    min_height: float = 0.20,
) -> torch.Tensor:
    """Episode terminates with success when cube centre rises above *min_height* world-Z."""
    obj = env.scene[asset_cfg.name]
    return obj.data.root_pos_w[:, 2] > min_height


# ── Events ────────────────────────────────────────────────────────────────────

@configclass
class PickUpEventCfg:
    """Reset robot to default pose; randomize cube_2 in the left-arm workspace only.

    cube_2 default world position: [0.55, 0.05, CUBE_Z]  (CUBE_Z ≈ 0.15 m)
    Target randomisation range:    x=[0.17, 0.27]  y=[0.03, 0.17]
    Required offsets:              x=[-0.38, -0.28]  y=[-0.02, 0.12]

    Range is intentionally compact — centred on the left arm's comfortable sweet spot.
    To widen: increase x max offset toward -0.20, or widen the y band symmetrically.
    """

    init_robot_pose = EventTerm(
        func=mdp_core.reset_scene_to_default,
        mode="reset",
    )

    randomize_cube_2 = EventTerm(
        func=mdp_core.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.38, -0.28),   # cube_2 default x=0.55 → actual x:[0.17, 0.27]
                "y": (-0.02, 0.12),    # cube_2 default y=0.05 → actual y:[0.03, 0.17]
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube_2"),
        },
    )


# ── Observations ──────────────────────────────────────────────────────────────

@configclass
class PickUpObservationsCfg:
    """Camera observations + EEF pose (needed by Mimic) + subtask completion signals."""

    @configclass
    class PolicyCfg(ObsGroup):
        front_cam = ObsTerm(
            func=mdp_core.image,
            params={"sensor_cfg": SceneEntityCfg("front_cam"), "data_type": "rgb", "normalize": False},
        )
        wrist_cam = ObsTerm(
            func=mdp_core.image,
            params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False},
        )
        right_wrist_cam = ObsTerm(
            func=mdp_core.image,
            params={"sensor_cfg": SceneEntityCfg("right_wrist_cam"), "data_type": "rgb", "normalize": False},
        )
        body_cam = ObsTerm(
            func=mdp_core.image,
            params={"sensor_cfg": SceneEntityCfg("body_cam"), "data_type": "rgb", "normalize": False},
        )
        # EEF pose — required by OpenArmPickUpIKAbsMimicEnv.get_robot_eef_pose()
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Subtask termination signals for Mimic auto-annotation.

        Both signals use cube height so they are robust and require no EEF tracking:
          grasp (min_height=0.155 m): cube_2 has just left the pad (~5 mm above rest).
                                      Marks end of the reach-and-grasp segment.
          lift  (min_height=0.30 m):  cube_2 is well elevated.
                                      Stored for debugging; not used as a Mimic term signal
                                      (it is the last subtask).
        """

        grasp = ObsTerm(
            func=cube_is_lifted,
            params={"asset_cfg": SceneEntityCfg("cube_2"), "min_height": 0.155},
        )
        lift = ObsTerm(
            func=cube_is_lifted,
            params={"asset_cfg": SceneEntityCfg("cube_2"), "min_height": 0.30},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


# ── Environment config ────────────────────────────────────────────────────────

@configclass
class OpenarmPickUpRedCubeEnvCfg(stack_ik_abs_visuomotor_env_cfg.OpenarmCubeStackVisuomotorEnvCfg):
    """OpenArm pick-up task: grasp and lift the red cube (cube_2) with the left arm.

    Key differences from the visuomotor stack env:
      - Only cube_2 (red) is present in the scene (cube_1 and cube_3 removed).
      - cube_2 is randomized within the left arm's reachable workspace only.
      - ee_frame targets openarm_left_ee_tcp (required by eef_pos/eef_quat obs).
      - subtask_terms observation group added for Mimic auto-annotation.
      - Success terminates when cube_2 rises above 0.20 m.
      - Right arm IK action kept so TAB-switching works during teleoperation.

    Mimic task: Isaac-PickUp-RedCube-OpenArm-IK-Abs-Mimic-v0
    """

    gripper_joint_names: list = ["openarm_left_finger_joint.*"]
    gripper_open_val: float = 0.044
    gripper_threshold: float = 0.018

    def __post_init__(self):
        super().__post_init__()  # cameras, IK-Abs action, pad, cubes

        # ── Remove blue and green cubes — only red cube participates ──────────
        # The scene builder skips attributes set to None.
        self.scene.cube_1 = None
        self.scene.cube_3 = None

        # ── Observations ─────────────────────────────────────────────────────
        self.observations.policy = PickUpObservationsCfg.PolicyCfg()
        self.observations.subtask_terms = PickUpObservationsCfg.SubtaskCfg()

        # ── ee_frame: track left EEF (required by eef_pos / eef_quat obs) ────
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_left_link1",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/openarm_left_ee_tcp",
                    name="end_effector",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
                )
            ],
        )

        # ── Terminations ─────────────────────────────────────────────────────
        self.terminations.time_out = None
        self.terminations.success = TerminationTermCfg(
            func=cube_pickup_success,
            params={"asset_cfg": SceneEntityCfg("cube_2"), "min_height": 0.20},
            time_out=False,
        )

        # ── Events: robot reset + cube_2 randomised in left-arm workspace ────
        self.events = PickUpEventCfg()

        # ── Right arm action (for TAB-switching during keyboard recording) ────
        self.actions.right_arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["openarm_right_joint[1-7]"],
            body_name="openarm_right_ee_tcp",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=True, ik_method="dls"
            ),
        )
        self.actions.right_gripper_action = mdp_core.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["openarm_right_finger_joint.*"],
            open_command_expr={"openarm_right_finger_joint.*": 0.044},
            close_command_expr={"openarm_right_finger_joint.*": 0.0},
        )

        self.image_obs_list = ["front_cam", "wrist_cam", "right_wrist_cam", "body_cam"]
