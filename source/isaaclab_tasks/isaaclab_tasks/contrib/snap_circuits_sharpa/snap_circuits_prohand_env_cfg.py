# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Snap Circuits workbench teleoperated with dual ProHand articulations."""

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
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from . import mdp

_REPO_ROOT = Path(__file__).resolve().parents[5]
_DEFAULT_ASSET_ROOT = _REPO_ROOT / "docker" / "apple-vision-pro" / "snap-circuits" / "assets"
_ASSET_ROOT = Path(os.environ.get("ISAACLAB_SNAP_CIRCUITS_ASSET_ROOT", _DEFAULT_ASSET_ROOT)).expanduser()
_PROHAND_REPOSITORY = _ASSET_ROOT / "pro-models"
_PROHAND_ASSETS = _PROHAND_REPOSITORY / "assets"
_PROHAND_LEFT_USD = _PROHAND_ASSETS / "usd" / "gen_1_D_L" / "gen_1_D_L.usda"
_PROHAND_RIGHT_USD = _PROHAND_ASSETS / "usd" / "gen_1_D_R" / "gen_1_D_R.usda"
_PROHAND_LEFT_URDF = _PROHAND_REPOSITORY / "assets" / "meshes" / "prohand_left_with_tips.urdf"
_PROHAND_RIGHT_URDF = _PROHAND_REPOSITORY / "assets" / "meshes" / "prohand_right_with_tips.urdf"
_DEMO_USD = _ASSET_ROOT / "prepared" / "snap_circuits_table.usda"
_CONFIG_DIR = Path(__file__).resolve().parent / "configs"

_PROHAND_FINGER_STEMS = [
    "t0_TM_abd",
    "t1_TM",
    "t2_MCP",
    "t3_DIP",
    "i0_CMC_abd",
    "i1_MCP",
    "i2_PIP",
    "i3_DIP",
    "m0_CMC_abd",
    "m1_MCP",
    "m2_PIP",
    "m3_DIP",
    "r0_CMC_abd",
    "r1_MCP",
    "r2_PIP",
    "r3_DIP",
    "p0_CMC_abd",
    "p1_MCP",
    "p2_PIP",
    "p3_DIP",
]


def _build_prohand_pipeline():
    """Return an AVP hand-tracking pipeline matching the 54-D action contract."""
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
        name="left_palm",
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
        name="right_palm",
    )
    connected_right_se3 = right_se3.connect({HandsSource.RIGHT: transformed_hands.output(HandsSource.RIGHT)})

    operator_to_mano = (0, -1, 0, -1, 0, 0, 0, 0, -1)
    left_joint_names = [f"L_{stem}" for stem in _PROHAND_FINGER_STEMS]
    right_joint_names = [f"R_{stem}" for stem in _PROHAND_FINGER_STEMS]
    left_dex = DexHandRetargeter(
        DexHandRetargeterConfig(
            hand_retargeting_config=str(_CONFIG_DIR / "prohand_left_dexpilot.yml"),
            hand_urdf=str(_PROHAND_LEFT_URDF),
            hand_joint_names=left_joint_names,
            hand_side="left",
            handtracking_to_baselink_frame_transform=operator_to_mano,
        ),
        name="left_hand",
    )
    connected_left_dex = left_dex.connect({HandsSource.LEFT: hands.output(HandsSource.LEFT)})
    right_dex = DexHandRetargeter(
        DexHandRetargeterConfig(
            hand_retargeting_config=str(_CONFIG_DIR / "prohand_right_dexpilot.yml"),
            hand_urdf=str(_PROHAND_RIGHT_URDF),
            hand_joint_names=right_joint_names,
            hand_side="right",
            handtracking_to_baselink_frame_transform=operator_to_mano,
        ),
        name="right_hand",
    )
    connected_right_dex = right_dex.connect({HandsSource.RIGHT: hands.output(HandsSource.RIGHT)})

    left_pose_names = ["left_x", "left_y", "left_z", "left_qx", "left_qy", "left_qz", "left_qw"]
    right_pose_names = ["right_x", "right_y", "right_z", "right_qx", "right_qy", "right_qz", "right_qw"]
    output_order = left_pose_names + left_joint_names + right_pose_names + right_joint_names
    reorderer = TensorReorderer(
        input_config={
            "left_pose": left_pose_names,
            "left_joints": left_joint_names,
            "right_pose": right_pose_names,
            "right_joints": right_joint_names,
        },
        output_order=output_order,
        name="action_reorderer",
        input_types={
            "left_pose": "array",
            "left_joints": "scalar",
            "right_pose": "array",
            "right_joints": "scalar",
        },
    )
    connected_reorderer = reorderer.connect(
        {
            "left_pose": connected_left_se3.output("ee_pose"),
            "left_joints": connected_left_dex.output("hand_joints"),
            "right_pose": connected_right_se3.output("ee_pose"),
            "right_joints": connected_right_dex.output("hand_joints"),
        }
    )
    return OutputCombiner({"action": connected_reorderer.output("output")})


def _prohand_cfg(side: str, usd_path: Path, position: tuple[float, float, float]) -> ArticulationCfg:
    """Build one locally fetched ProHand articulation configuration."""
    prefix = "L" if side == "left" else "R"
    return ArticulationCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{side.title()}ProHand",
        spawn=UsdFileCfg(
            usd_path=str(usd_path),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=position,
            joint_pos={f"{prefix}_.*": 0.0},
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


@configclass
class SnapCircuitsProHandSceneCfg(InteractiveSceneCfg):
    """Packing table, circuit parts, and two free-root ProHands."""

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
    left_hand = _prohand_cfg("left", _PROHAND_LEFT_USD, (-0.25, 0.3, 1.1))
    right_hand = _prohand_cfg("right", _PROHAND_RIGHT_USD, (0.25, 0.3, 1.1))
    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.8, 0.8, 0.8), intensity=3000.0),
    )


@configclass
class ActionsCfg:
    left_hand = mdp.ProHandActionCfg(asset_name="left_hand", side="left")
    right_hand = mdp.ProHandActionCfg(asset_name="right_hand", side="right")


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        left_joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("left_hand")})
        right_joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("right_hand")})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class SnapCircuitsProHandEnvCfg(ManagerBasedRLEnvCfg):
    """Teleoperation-only configuration for the Snap Circuits ProHand demo."""

    scene: SnapCircuitsProHandSceneCfg = SnapCircuitsProHandSceneCfg(
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

        # Match the Sharpa workbench at about 1.15 m above the tracked floor
        # for a tall standing operator.
        self.xr = XrCfg(anchor_pos=(0.0, 0.25, -0.15), anchor_rot=(0.0, 0.0, 0.0, 1.0))
        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=_build_prohand_pipeline,
            sim_device=self.sim.device,
            xr_cfg=self.xr,
            app_name="IsaacLabSnapCircuitsProHand",
        )
