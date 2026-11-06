# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bimanual VR teleoperation of the AgiBot G2 through AgiBot's ik_7d solver.

The chain is: VR controllers -> IsaacTeleop retargeting graph -> a 16-element
action tensor -> :class:`~isaaclab_contrib.mdp.Ik7dAction` -> ik_7d ->
``set_joint_position_target_index`` on the 14 arm joints.

Stock ``isaacteleop`` supplies everything on the VR side. Its
``Se3AbsRetargeter`` already emits ``[x, y, z, qx, qy, qz, qw]`` in xyzw, which
is byte-for-byte the first seven elements of each arm's :class:`Ik7dAction`
slice, so no reordering happens anywhere in this file. The one thing it does not
supply is the eighth element -- the elbow swivel that a 7-DoF arm has and a
6-DoF pose does not determine -- which :mod:`.swivel_retargeter` adds from the
thumbstick.
"""

from __future__ import annotations

from isaaclab_teleop import IsaacTeleopCfg, XrCfg

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

from isaaclab_contrib.assets import AGIBOT_G2_T2_CRS_CFG, AGIBOT_G2_T2_CRS_URDF
from isaaclab_contrib.controllers import Ik7dControllerCfg
from isaaclab_contrib.mdp import Ik7dActionCfg

##
# Action layout
##

ARMS = ["left", "right"]
"""Arm order. Fixes both the ik_7d solve order and the action tensor layout."""

ROBOT_NAME = "G2_t2_crs"
"""ik_7d model name. ``arm_plane_angle`` is code-generated per arm variant, so
this has to be one of ik_7d's supported robots and has to match the URDF."""


def _build_g2_ik7d_pipeline():
    """Build the IsaacTeleop retargeting graph for bimanual ik_7d teleoperation.

    Flattens the four node outputs into ``[left_pose(7), left_swivel(1),
    right_pose(7), right_swivel(1)]`` for :class:`~isaaclab_contrib.mdp.Ik7dAction`.

    Returns:
        The ``OutputCombiner`` pipeline exposing one ``"action"`` output.
    """
    # Imported here, not at module scope: this module is imported to *register*
    # the task, which must not require the isacteleop stack to be installed.
    from isaacteleop.retargeters import Se3AbsRetargeter, Se3RetargeterConfig, TensorReorderer
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    from .swivel_retargeter import SwivelRetargeter, SwivelRetargeterConfig

    controllers = ControllersSource(name="controllers")

    # World-to-anchor transform supplied by IsaacTeleopDevice. Applying it here
    # puts the controller poses in the simulation *world* frame, which is why the
    # action term below is configured with ``pose_frame="world"``.
    transform_input = ValueInput("world_T_anchor", TransformMatrix())
    transformed_controllers = controllers.transformed(transform_input.output(ValueInput.VALUE))

    se3_nodes = {}
    swivel_nodes = {}
    for arm in ARMS:
        side = ControllersSource.LEFT if arm == "left" else ControllersSource.RIGHT

        # ``target_offset_*``: grip-frame-to-EE-frame rotation, degrees, intrinsic
        # XYZ. Re-measure if the robot or the grip changes.
        se3_cfg = Se3RetargeterConfig(
            input_device=side,
            zero_out_xy_rotation=False,
            use_wrist_rotation=True,
            use_wrist_position=True,
            target_offset_roll=180.0,
            target_offset_pitch=0.0,
            target_offset_yaw=-90.0,
        )
        se3 = Se3AbsRetargeter(se3_cfg, name=f"{arm}_ee_pose")
        se3_nodes[arm] = (se3, se3.connect({side: transformed_controllers.output(side)}))

        # ik_7d routes a positive swivel along each arm's measured free direction,
        # and that direction is mirrored between the arms, so invert the right to
        # keep "push the stick outward -> elbow swings outward" on both.
        swivel_cfg = SwivelRetargeterConfig(controller_side=arm, invert=(arm == "right"))
        swivel = SwivelRetargeter(swivel_cfg, name=f"{arm}_swivel")
        swivel_nodes[arm] = (swivel, swivel.connect({f"controller_{arm}": transformed_controllers.output(side)}))

    # Se3AbsRetargeter emits 7 elements: [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w].
    pose_elements = {
        arm: [f"{arm}_{name}" for name in ("pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w")]
        for arm in ARMS
    }
    swivel_elements = {arm: [f"{arm}_swivel"] for arm in ARMS}

    input_config = {}
    input_types = {}
    output_order = []
    for arm in ARMS:
        input_config[f"{arm}_ee_pose"] = pose_elements[arm]
        input_types[f"{arm}_ee_pose"] = "array"
        input_config[f"{arm}_swivel"] = swivel_elements[arm]
        input_types[f"{arm}_swivel"] = "scalar"
        output_order += pose_elements[arm] + swivel_elements[arm]

    reorderer = TensorReorderer(
        input_config=input_config,
        output_order=output_order,
        name="action_reorderer",
        input_types=input_types,
    )
    connected_reorderer = reorderer.connect(
        {
            **{f"{arm}_ee_pose": se3_nodes[arm][1].output("ee_pose") for arm in ARMS},
            **{f"{arm}_swivel": swivel_nodes[arm][1].output("swivel") for arm in ARMS},
        }
    )

    pipeline = OutputCombiner({"action": connected_reorderer.output("output")})
    return pipeline


##
# Scene definition
##


@configclass
class G2TeleopSceneCfg(InteractiveSceneCfg):
    """A ground plane, a light and one G2.

    Deliberately bare. The point of this environment is the control chain, and
    every prop added in front of the robot is a collision the arms can catch on
    while the operator is still learning where the workspace is.
    """

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)))

    robot: ArticulationCfg = AGIBOT_G2_T2_CRS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=2000.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_ik = Ik7dActionCfg(
        asset_name="robot",
        controller=Ik7dControllerCfg(
            robot_name=ROBOT_NAME,
            urdf_path=AGIBOT_G2_T2_CRS_URDF,
            arms=ARMS,
        ),
        # The retargeting graph hands us poses in the simulation world frame; the
        # action term pulls them back through the robot's measured base-link pose,
        # which is the frame ik_7d works in.
        pose_frame="world",
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Enough state to replay or diagnose a session."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        # ``reset_joint_targets`` is off by default in Isaac Lab, on the reasoning
        # that targets belong to action terms. ``Ik7dAction`` writes only the 14
        # arm joints, so waist and head keep PhysX's zero initial target. The
        # torso unfolds out of ik_7d's home posture over the first few hundred
        # steps while the arms hold a pose the moving torso drags around. It reads
        # exactly like an IK bug and is not one. Do not remove this.
        params={"reset_joint_targets": True},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP.

    ``teleop_se3_agent.py`` sets ``time_out`` to ``None`` so a session is not cut
    short, but it does so unconditionally -- the attribute has to exist.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment
##


@configclass
class G2Ik7dTeleopEnvCfg(ManagerBasedRLEnvCfg):
    """Bimanual VR teleoperation of the AgiBot G2 via ik_7d."""

    scene: G2TeleopSceneCfg = G2TeleopSceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=True)

    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Unused managers.
    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 2
        self.episode_length_s = 600.0

        # 200 Hz physics. ik_7d solves both arms in ~0.5 ms (p99 0.63 ms), so the
        # 100 Hz control rate this gives is comfortable.
        self.sim.dt = 1.0 / 200.0
        self.sim.render_interval = 4

        self.viewer.eye = (1.6, -1.6, 1.6)
        self.viewer.lookat = (0.0, 0.0, 0.9)

        self.xr = XrCfg(
            anchor_pos=(0.0, 0.0, 0.0),
            anchor_rot=(0.0, 0.0, 0.0, 1.0),
        )

        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=_build_g2_ik7d_pipeline,
            sim_device=self.sim.device,
            xr_cfg=self.xr,
        )
