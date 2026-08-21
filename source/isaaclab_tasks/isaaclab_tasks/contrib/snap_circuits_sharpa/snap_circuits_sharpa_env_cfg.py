# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Snap Circuits workbench teleoperated with dual Sharpa Wave hands."""

from __future__ import annotations

import os
from pathlib import Path

from isaaclab_teleop import IsaacTeleopCfg
from isaaclab_teleop.xr_cfg import XrCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from . import mdp

_REPO_ROOT = Path(__file__).resolve().parents[5]
_DEFAULT_ASSET_ROOT = _REPO_ROOT / "docker" / "apple-vision-pro" / "snap-circuits" / "assets"
_ASSET_ROOT = Path(os.environ.get("ISAACLAB_SNAP_CIRCUITS_ASSET_ROOT", _DEFAULT_ASSET_ROOT)).expanduser()
_SHARPA_REPOSITORY = _ASSET_ROOT / "sharpa-urdf-usd-xml"
_SHARPA_DUAL_USD = _SHARPA_REPOSITORY / "wave_01" / "dual_sharpa_wave" / "dual_sharpa_wave.usda"
_DEMO_USD = _ASSET_ROOT / "prepared" / "snap_circuits_table.usda"
_CONFIG_DIR = Path(__file__).resolve().parent / "configs"

_SHARPA_JOINTS = [
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
]


def _build_sharpa_pipeline():
    """Return an AVP hand-tracking pipeline matching the 58-D action contract."""
    from isaacteleop.retargeters import (
        DexHandRetargeter,
        DexHandRetargeterConfig,
        Se3AbsRetargeter,
        Se3RetargeterConfig,
        TensorReorderer,
    )
    from isaacteleop.retargeting_engine.deviceio_source_nodes import HandsSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    hands = HandsSource(name="hands")
    transform_input = ValueInput("world_T_anchor", TransformMatrix())
    transformed_hands = hands.transformed(transform_input.output(ValueInput.VALUE))

    left_se3 = Se3AbsRetargeter(
        Se3RetargeterConfig(
            input_device=HandsSource.LEFT,
            zero_out_xy_rotation=False,
            use_wrist_rotation=True,
            use_wrist_position=True,
            target_offset_roll=0.0,
            target_offset_pitch=90.0,
            target_offset_yaw=0.0,
        ),
        name="left_wrist",
    )
    connected_left_se3 = left_se3.connect({HandsSource.LEFT: transformed_hands.output(HandsSource.LEFT)})

    right_se3 = Se3AbsRetargeter(
        Se3RetargeterConfig(
            input_device=HandsSource.RIGHT,
            zero_out_xy_rotation=False,
            use_wrist_rotation=True,
            use_wrist_position=True,
            target_offset_roll=180.0,
            target_offset_pitch=-90.0,
            target_offset_yaw=0.0,
        ),
        name="right_wrist",
    )
    connected_right_se3 = right_se3.connect({HandsSource.RIGHT: transformed_hands.output(HandsSource.RIGHT)})

    operator_to_mano = (0, -1, 0, -1, 0, 0, 0, 0, -1)
    left_joint_names = [f"left_{name}" for name in _SHARPA_JOINTS]
    right_joint_names = [f"right_{name}" for name in _SHARPA_JOINTS]

    left_dex = DexHandRetargeter(
        DexHandRetargeterConfig(
            hand_retargeting_config=str(_CONFIG_DIR / "sharpa_wave_left_dexpilot.yml"),
            hand_urdf=str(_SHARPA_REPOSITORY / "wave_01" / "left_sharpa_wave" / "left_sharpa_wave.urdf"),
            hand_joint_names=left_joint_names,
            hand_side="left",
            handtracking_to_baselink_frame_transform=operator_to_mano,
        ),
        name="left_hand",
    )
    connected_left_dex = left_dex.connect({HandsSource.LEFT: hands.output(HandsSource.LEFT)})

    right_dex = DexHandRetargeter(
        DexHandRetargeterConfig(
            hand_retargeting_config=str(_CONFIG_DIR / "sharpa_wave_right_dexpilot.yml"),
            hand_urdf=str(_SHARPA_REPOSITORY / "wave_01" / "right_sharpa_wave" / "right_sharpa_wave.urdf"),
            hand_joint_names=right_joint_names,
            hand_side="right",
            handtracking_to_baselink_frame_transform=operator_to_mano,
        ),
        name="right_hand",
    )
    connected_right_dex = right_dex.connect({HandsSource.RIGHT: hands.output(HandsSource.RIGHT)})

    left_pose_names = ["left_x", "left_y", "left_z", "left_qx", "left_qy", "left_qz", "left_qw"]
    right_pose_names = ["right_x", "right_y", "right_z", "right_qx", "right_qy", "right_qz", "right_qw"]
    output_order = left_pose_names + right_pose_names + left_joint_names + right_joint_names
    reorderer = TensorReorderer(
        input_config={
            "left_pose": left_pose_names,
            "right_pose": right_pose_names,
            "left_joints": left_joint_names,
            "right_joints": right_joint_names,
        },
        output_order=output_order,
        name="action_reorderer",
        input_types={
            "left_pose": "array",
            "right_pose": "array",
            "left_joints": "scalar",
            "right_joints": "scalar",
        },
    )
    connected_reorderer = reorderer.connect(
        {
            "left_pose": connected_left_se3.output("ee_pose"),
            "right_pose": connected_right_se3.output("ee_pose"),
            "left_joints": connected_left_dex.output("hand_joints"),
            "right_joints": connected_right_dex.output("hand_joints"),
        }
    )
    return OutputCombiner({"action": connected_reorderer.output("output")}), [left_dex, right_dex]


@configclass
class SnapCircuitsSharpaSceneCfg(InteractiveSceneCfg):
    """Packing table, prepared component layout, and dual Sharpa Wave hands."""

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.55, 0.0)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    demo_assets = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SnapCircuitsDemo",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.55, 1.0)),
        spawn=UsdFileCfg(usd_path=str(_DEMO_USD), semantic_tags=[("class", "demo_component")]),
    )

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=UsdFileCfg(
            usd_path=str(_SHARPA_DUAL_USD),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "head_.*": 0.0,
                "left_x_joint": -0.25,
                "left_y_joint": 0.25,
                "left_z_joint": 1.25,
                "left_(roll|pitch|yaw)_joint": 0.0,
                "right_x_joint": 0.25,
                "right_y_joint": 0.25,
                "right_z_joint": 1.25,
                "right_(roll|pitch|yaw)_joint": 0.0,
                "left_(thumb|index|middle|ring|pinky).*": 0.0,
                "right_(thumb|index|middle|ring|pinky).*": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=None,
                damping=None,
            )
        },
    )

    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.8, 0.8, 0.8), intensity=3000.0),
    )


@configclass
class ActionsCfg:
    sharpa_hands = mdp.SharpaWaveBimanualActionCfg(asset_name="robot")


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_robot = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class SnapCircuitsSharpaEnvCfg(ManagerBasedRLEnvCfg):
    """Teleoperation-only configuration for the Snap Circuits Sharpa demo."""

    scene: SnapCircuitsSharpaSceneCfg = SnapCircuitsSharpaSceneCfg(
        num_envs=1,
        env_spacing=3.0,
        replicate_physics=False,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    commands = None
    rewards = None
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum = None

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 1.0e9
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = 2
        self.num_rerenders_on_reset = 3

        # The work surface is at world z=1.0. A -0.15 m floor anchor presents
        # it at about 1.15 m for a tall standing operator.
        self.xr = XrCfg(anchor_pos=(0.0, 0.25, -0.15), anchor_rot=(0.0, 0.0, 0.0, 1.0))
        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=lambda: _build_sharpa_pipeline()[0],
            sim_device=self.sim.device,
            xr_cfg=self.xr,
            app_name="IsaacLabSnapCircuitsSharpa",
        )
