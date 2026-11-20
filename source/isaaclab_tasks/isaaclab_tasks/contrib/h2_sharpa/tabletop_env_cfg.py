# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint ordering, wrist calibration, and control shared by the H2 + Sharpa tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_newton.controllers import NewtonJointImpedanceControllerCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass, instantiate
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.string import resolve_matching_names_values

from isaaclab_assets.robots import H2_SHARPA_CFG

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

ARM_JOINT_NAMES = [
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
]

HAND_JOINT_NAMES = [
    f"{side}_{joint}"
    for side in ("left", "right")
    for joint in (
        "thumb_CMC_FE",
        "thumb_CMC_AA",
        "thumb_MCP_FE",
        "thumb_MCP_AA",
        "thumb_IP",
        "index_MCP_FE",
        "index_MCP_AA",
        "index_PIP",
        "index_DIP",
        "middle_MCP_FE",
        "middle_MCP_AA",
        "middle_PIP",
        "middle_DIP",
        "ring_MCP_FE",
        "ring_MCP_AA",
        "ring_PIP",
        "ring_DIP",
        "pinky_CMC",
        "pinky_MCP_FE",
        "pinky_MCP_AA",
        "pinky_PIP",
        "pinky_DIP",
    )
]

POLICY_JOINT_NAMES = ARM_JOINT_NAMES + HAND_JOINT_NAMES
"""Absolute policy joint positions: left arm, right arm, left hand, right hand."""

# pnp_apple_sim_new_bg_v1 episode_000027 frame 0, replayed from
# apple_pick_and_place_05142000_filtered episode_000043, in POLICY_JOINT_NAMES
# (left_arm, right_arm, left_hand, right_hand).
# fmt: off
_FRAME0_POLICY_58_POS = (
    # Left arm
    0.2754, 0.0215, -0.0674, -0.0774, 0.0518, -0.1481, -0.026,
    # Right arm
    0.1734, -0.0544, 0.0165, -0.0457, 0.0006, 0.0136, 0.1644,
    # Left hand
    0.2834, 0.0498, -0.4345, 0.2344, 0.7783, -0.1513, -0.0731, 0.0373, 0.1479, -0.154, 0.0019,
    0.0358, 0.1647, -0.1514, 0.1047, 0.0385, 0.1432, 0.1048, -0.1552, 0.1278, 0.0348, 0.0924,
    # Right hand
    -0.0452, 0.0498, 0.3536, 0.0689, 0.0869, -0.1104, 0.1215, 0.0687, 0.1541, -0.1028, 0.1457,
    0.081, 0.0946, -0.1501, 0.1255, 0.0361, 0.089, 0.1171, -0.1064, 0.0177, 0.0611, 0.0884,
)
# fmt: on

H2_TABLETOP_ROBOT_CFG = H2_SHARPA_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    actuators={
        **H2_SHARPA_CFG.actuators,
        # The impedance controller supplies the complete arm torque. Keep the actuator effort limits.
        "arms": H2_SHARPA_CFG.actuators["arms"].replace(stiffness=0.0, damping=0.0),
    },
    init_state=H2_SHARPA_CFG.init_state.replace(
        joint_pos={
            **H2_SHARPA_CFG.init_state.joint_pos,
            **dict(zip(POLICY_JOINT_NAMES, _FRAME0_POLICY_58_POS, strict=True)),
            "head_pitch_joint": 0.6,
        },
    ),
)
"""H2 + Sharpa robot in the starting pose shared by both tabletop tasks."""

LEFT_WRIST_CAMERA_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/left_hand_C_MC/left_wrist_camera",
    update_period=0.02,
    height=480,
    width=640,
    data_types=["rgb"],
    spawn=sim_utils.FisheyeCameraCfg(
        projection_type="fisheyePolynomial",
        focal_length=0.15435,
        focus_distance=400.0,
        f_stop=0.0,
        horizontal_aperture=0.576,
        vertical_aperture=0.432,
        fisheye_nominal_width=640.0,
        fisheye_nominal_height=480.0,
        fisheye_optical_centre_x=319.90624,
        fisheye_optical_centre_y=239.67252,
        fisheye_max_fov=196,
        fisheye_polynomial_a=0.0,
        fisheye_polynomial_b=5.788058552990647e-3,
        fisheye_polynomial_c=1.6881056765838268e-6,
        fisheye_polynomial_d=-4.2228624221772085e-8,
        fisheye_polynomial_e=1.4575948452911756e-10,
        fisheye_polynomial_f=-1.4645006973842839e-13,
        clipping_range=(0.01, 1.0e5),
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(0.07498591, 0.0007267178, 0.004823103),
        rot=(-0.6486821, 0.6996208, 0.1177079, 0.2754761),
        convention="opengl",
    ),
)
"""Left Sharpa Wave wrist camera, mounted on the hand's CAD bracket."""

RIGHT_WRIST_CAMERA_CFG = LEFT_WRIST_CAMERA_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot/right_hand_C_MC/right_wrist_camera",
    offset=CameraCfg.OffsetCfg(
        pos=(0.07964737, 0.001976802, 0.01021968),
        rot=(0.6884326, -0.664184, 0.2683462, 0.1136246),
        convention="opengl",
    ),
)
"""Right Sharpa Wave wrist camera, mirrored from the left mount."""


class H2ArmImpedanceAction(mdp.JointAction):
    """Track the H2 arm position commands with the robot's gains and gravity feed-forward."""

    def __init__(self, cfg: mdp.JointActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        arms = H2_SHARPA_CFG.actuators["arms"]
        controller_cfg = NewtonJointImpedanceControllerCfg(
            stiffness=resolve_matching_names_values(arms.stiffness, self._joint_names)[2],
            damping=resolve_matching_names_values(arms.damping, self._joint_names)[2],
            use_inertia_decoupling=False,
            use_coriolis_compensation=False,
            use_gravity_compensation=True,
        )
        self._controller = instantiate(controller_cfg, self.num_envs, self.action_dim, self.device)
        self._gravity_ids = self._joint_ids + self._asset.num_base_dofs

    def apply_actions(self):
        data = self._asset.data
        effort = self._controller.compute(
            self.processed_actions,
            data.joint_pos.torch[:, self._joint_ids],
            data.joint_vel.torch[:, self._joint_ids],
            gravity=data.gravity_compensation_forces.torch[:, self._gravity_ids],
        )
        self._asset.actuators.target_command.set_effort_index(value=effort, joint_ids=self._joint_ids)


@configclass
class ActionsCfg:
    """Absolute arm and hand position commands with gravity feed-forward on the arms."""

    arms = mdp.JointActionCfg(
        class_type=H2ArmImpedanceAction, asset_name="robot", joint_names=ARM_JOINT_NAMES, preserve_order=True
    )
    hands = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=HAND_JOINT_NAMES,
        use_default_offset=False,
        preserve_order=True,
    )


@configclass
class H2TabletopSceneCfg(InteractiveSceneCfg):
    """H2 robot and room shared by the two tabletop tasks."""

    robot = H2_TABLETOP_ROBOT_CFG

    background = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/background",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Environments/NuRec/IMG_6246_nurec_aligned_scaled.usdz"
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-1.35, -0.45, 2.0),
            rot=(0.0, 0.0, 0.70710678, 0.70710678),
        ),
    )

    room_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/room_left",
        spawn=sim_utils.CuboidCfg(
            size=(4.0, 0.05, 3.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.44, 0.43), roughness=0.8, metallic=0.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.5, 2.0, 1.5)),
    )
    room_right = room_left.replace(
        prim_path="{ENV_REGEX_NS}/room_right",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.5, -2.0, 1.5)),
    )
    room_front = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/room_front",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 4.0, 3.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.44, 0.43), roughness=0.8, metallic=0.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.5, 0.0, 1.5)),
    )
    room_back = room_front.replace(
        prim_path="{ENV_REGEX_NS}/room_back",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-2.5, 0.0, 1.5)),
    )
    room_ceiling = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/room_ceiling",
        spawn=sim_utils.CuboidCfg(
            size=(4.0, 4.0, 0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.44, 0.43), roughness=0.8, metallic=0.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.5, 0.0, 3.0)),
    )

    room_floor = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/room_floor",
        spawn=sim_utils.CuboidCfg(
            size=(8.0, 8.0, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.21), roughness=0.9, metallic=0.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.5, 0.0, 0.02)),
    )

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )


@configclass
class PolicyStateCfg(ObsGroup):
    robot_policy_joint_pos = ObsTerm(
        func=mdp.joint_pos,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=POLICY_JOINT_NAMES, preserve_order=True),
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False
