# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton reinforcement-learning configuration for KUKA-Allegro cube stacking."""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg

from isaaclab.assets import ArticulationCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.stack import mdp
from isaaclab_tasks.contrib.stack.config.franka.stack_rl_env_cfg import (
    CurriculumCfg,
    EventCfg,
    FrankaCubeStackRLEnvCfg,
    RewardsCfg,
)
from isaaclab_tasks.contrib.stack.mdp.kuka_allegro_reset import (
    KUKA_ALLEGRO_ALL_HAND_JOINT_NAMES,
    KUKA_ALLEGRO_DIVERSE_ARM_WORKSPACE_LOWER,
    KUKA_ALLEGRO_DIVERSE_ARM_WORKSPACE_UPPER,
    KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_CONTACT_POSES,
    KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_OPEN_COMMANDS,
    KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_OPEN_POSES,
    KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_PRELOAD_COMMANDS,
    KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_TOOL_OFFSETS,
    KUKA_ALLEGRO_GRASP_PAIR_JOINT_NAMES,
    KUKA_ALLEGRO_LARGE_CUBE_EDGE_LENGTH,
    KUKA_ALLEGRO_LARGE_CUBE_RESTING_HEIGHT,
    KUKA_ALLEGRO_STACK_ARM_POSES,
)

from isaaclab_assets.robots import KUKA_ALLEGRO_CFG

_ARM_JOINT_NAMES = ["iiwa7_joint_(1|2|3|4|5|6|7)"]
_PINCH_JOINT_NAMES = ["index_joint_(0|1|2|3)", "thumb_joint_(0|1|2|3)"]
_HAND_TIP_BODY_NAMES = (
    "index_biotac_tip",
    "middle_biotac_tip",
    "ring_biotac_tip",
    "thumb_biotac_tip",
)
_PINCH_CENTER_IN_PALM = (0.0570965, -0.0375159, 0.0498749)


def _make_kuka_allegro_event_cfg() -> EventCfg:
    """Create the production KUKA reset event."""
    cfg = EventCfg()
    cfg.reset_from_state_buffer.func = mdp.KukaAllegroResetStateTable
    cfg.reset_from_state_buffer.params.update(
        {
            # Cached hand-object states stay exact. Only deployment-like table
            # starts receive the validated arm perturbation.
            "arm_joint_noise_range": 0.0,
            "table_arm_joint_noise_range": 0.04,
            "table_cube_planar_translation_range": 0.0,
            "table_cube_rotation_range": 0.0,
        }
    )
    return cfg


@configclass
class KukaAllegroCubeStackRLEnvCfg(FrankaCubeStackRLEnvCfg):
    """Stack lightweight 8 cm cubes with 7 arm and 16 independent hand actions."""

    events: EventCfg = _make_kuka_allegro_event_cfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        self._configure_robot_and_physics()
        self._configure_actions()
        self._configure_observations()
        self._configure_objective()

    @staticmethod
    def _arm_entity_cfg() -> SceneEntityCfg:
        """Return an independently resolvable KUKA arm entity selection."""
        return SceneEntityCfg("robot", joint_names=_ARM_JOINT_NAMES)

    def _configure_robot_and_physics(self) -> None:
        """Install the KUKA-Allegro asset, large cubes, and validated Newton solver."""
        robot_semantic_tags = self.scene.robot.spawn.semantic_tags
        center_pregrasp = KUKA_ALLEGRO_STACK_ARM_POSES[4][1]
        robot_joint_positions = {
            **{f"iiwa7_joint_{joint_id + 1}": value for joint_id, value in enumerate(center_pregrasp)},
            **{
                joint_name: value
                for joint_name, value in zip(
                    KUKA_ALLEGRO_ALL_HAND_JOINT_NAMES,
                    KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_OPEN_COMMANDS[0],
                    strict=True,
                )
            },
        }
        self.scene.robot = KUKA_ALLEGRO_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                # The authored arm reaches negative X; rotate the fixed base so
                # the shared stack workspace remains in front of the robot.
                rot=(0.0, 0.0, 1.0, 0.0),
                joint_pos=robot_joint_positions,
            ),
        )
        self.scene.robot.spawn.semantic_tags = robot_semantic_tags
        self.scene.robot.spawn.rigid_props.disable_gravity = False
        self.scene.ee_frame = None

        self.sim.physics = NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                solver="newton",
                integrator="implicitfast",
                njmax=300,
                nconmax=200,
                impratio=1.0,
                cone="pyramidal",
                update_data_interval=2,
                iterations=100,
                ls_iterations=15,
                use_mujoco_contacts=False,
                ccd_iterations=35,
            ),
            num_substeps=2,
            collision_decimation=1,
            use_cuda_graph=True,
            collision_cfg=NewtonCollisionPipelineCfg(
                broad_phase="explicit",
                reduce_contacts=True,
                rigid_contact_max=4_000_000,
            ),
            default_shape_cfg=NewtonShapeCfg(margin=0.0, gap=0.0),
        )

        for cube in (self.scene.cube_1, self.scene.cube_2, self.scene.cube_3):
            cube.spawn.size = (KUKA_ALLEGRO_LARGE_CUBE_EDGE_LENGTH,) * 3
            cube.init_state.pos = (
                cube.init_state.pos[0],
                cube.init_state.pos[1],
                KUKA_ALLEGRO_LARGE_CUBE_RESTING_HEIGHT,
            )

        self.sim.default_visualizer_cfg.eye = (1.5, 1.5, 1.1)
        self.sim.default_visualizer_cfg.lookat = (0.48, 0.0, 0.18)

    def _configure_actions(self) -> None:
        """Configure one measured-state residual action per robot joint."""
        self.actions.arm_action = mdp.WorkspaceBoundedRelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=_ARM_JOINT_NAMES,
            scale=0.12,
            max_delta=0.12,
            workspace_lower=KUKA_ALLEGRO_DIVERSE_ARM_WORKSPACE_LOWER,
            workspace_upper=KUKA_ALLEGRO_DIVERSE_ARM_WORKSPACE_UPPER,
            gravity_compensation=True,
        )
        self.actions.gripper_action = mdp.ResetPreservingRelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=list(KUKA_ALLEGRO_ALL_HAND_JOINT_NAMES),
            preserve_order=True,
            scale=0.10,
            max_delta=0.10,
            joint_limit_margin=0.02,
            reset_preload_joint_names=KUKA_ALLEGRO_ALL_HAND_JOINT_NAMES,
            reset_preload_commands_by_pair=KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_PRELOAD_COMMANDS,
            reset_open_commands_by_pair=KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_OPEN_COMMANDS,
            preload_release_threshold=0.5,
            preload_release_steps=2,
        )

        # These gains retain the screened preload while the 0.10 rad target cap
        # and actuator effort limit remain the policy and physical backstops.
        hand_actuator = self.scene.robot.actuators["kuka_allegro_actuators"]
        hand_expression = "(index|middle|ring|thumb)_joint_(0|1|2|3)"
        hand_actuator.stiffness[hand_expression] = 20.0
        hand_actuator.damping[hand_expression] = 0.5

    def _configure_observations(self) -> None:
        """Expose arm, complete hand, grasp geometry, and order-invariant object state."""
        arm_cfg = self._arm_entity_cfg()
        all_hand_cfg = SceneEntityCfg(
            "robot",
            joint_names=list(KUKA_ALLEGRO_ALL_HAND_JOINT_NAMES),
            preserve_order=True,
        )
        pair_posture_params = {
            "joint_names_by_pair": KUKA_ALLEGRO_GRASP_PAIR_JOINT_NAMES,
            "open_joint_positions_by_pair": KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_OPEN_POSES,
            "closed_joint_positions_by_pair": KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_CONTACT_POSES,
            "finger_joint_counts": (4, 4),
        }

        self.observations.policy.joint_pos.params["asset_cfg"] = arm_cfg
        self.observations.policy.joint_vel.params["asset_cfg"] = self._arm_entity_cfg()
        self.observations.policy.object.params = {
            "tool_body_name": "palm_link",
            "tool_offset": _PINCH_CENTER_IN_PALM,
            "grasp_pair_tool_offsets": KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_TOOL_OFFSETS,
            "cube_height": KUKA_ALLEGRO_LARGE_CUBE_EDGE_LENGTH,
            "xy_threshold": 0.025,
            "height_threshold": 0.012,
        }
        self.observations.policy.gripper_pos = self.observations.policy.gripper_pos.replace(
            func=mdp.grasp_pair_gripper_posture,
            params=pair_posture_params,
        )
        self.observations.policy.eef_velocity = self.observations.policy.eef_velocity.replace(
            func=mdp.grasp_pair_tool_velocity,
            params={
                "tool_body_name": "palm_link",
                "tool_offsets_by_pair": KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_TOOL_OFFSETS,
            },
        )
        self.observations.policy.eef_axes = self.observations.policy.eef_axes.replace(
            func=mdp.tool_axes,
            params={"tool_body_name": "palm_link", "tool_offset": _PINCH_CENTER_IN_PALM},
        )
        self.observations.policy.cube_x_axes = ObsTerm(func=mdp.role_conditioned_cube_x_axes)
        self.observations.policy.hand_joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": all_hand_cfg})
        self.observations.policy.hand_joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=list(KUKA_ALLEGRO_ALL_HAND_JOINT_NAMES),
                    preserve_order=True,
                )
            },
        )
        self.observations.policy.hand_tip_positions = ObsTerm(
            func=mdp.body_positions_relative_to_tool,
            params={
                "body_cfg": SceneEntityCfg(
                    "robot",
                    body_names=_HAND_TIP_BODY_NAMES,
                    preserve_order=True,
                ),
                "tool_body_name": "palm_link",
                "tool_offset": (0.0, 0.0, 0.0),
            },
        )

    def _configure_objective(self) -> None:
        """Configure full-hand success, reset progress, and global epsilon sampling."""
        self.events = _make_kuka_allegro_event_cfg()
        self.rewards = RewardsCfg()
        self.curriculum = CurriculumCfg()
        self.rewards.joint_vel.params["asset_cfg"] = self._arm_entity_cfg()
        self.rewards.reset_progress = RewTerm(
            func=mdp.stack_success_pulse,
            params={"context_term_name": "learning_progress_context"},
            weight=0.5,
        )

        pair_goal_params = {
            "grasp_pair_tool_offsets": KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_TOOL_OFFSETS,
            "grasp_pair_joint_names": KUKA_ALLEGRO_GRASP_PAIR_JOINT_NAMES,
            "grasp_pair_open_joint_positions": KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_OPEN_POSES,
            "grasp_pair_closed_joint_positions": KUKA_ALLEGRO_FULL_HAND_GRASP_PAIR_CONTACT_POSES,
            "minimum_gripper_closure": 0.8,
            "maximum_gripper_closure": 0.2,
            "cube_height": KUKA_ALLEGRO_LARGE_CUBE_EDGE_LENGTH,
            "xy_threshold": 0.025,
            "height_threshold": 0.012,
        }
        self.terminations.learning_progress_context.params.update(
            {
                "tool_body_name": "palm_link",
                **pair_goal_params,
            }
        )
        self.terminations.progress_context.func = mdp.StableFullHandOrderInvariantStackGoal
        self.terminations.progress_context.params = {
            "minimum_episode_steps": 3,
            "hold_steps": 5,
            "maximum_cube_linear_velocity": 0.10,
            "maximum_cube_angular_velocity": 1.0,
            "minimum_fingertip_cube_clearance": 0.010,
            "fingertip_cfg": SceneEntityCfg(
                "robot",
                body_names=_HAND_TIP_BODY_NAMES,
                preserve_order=True,
            ),
            "xy_threshold": 0.025,
            "height_threshold": 0.012,
            "cube_height": KUKA_ALLEGRO_LARGE_CUBE_EDGE_LENGTH,
        }
        self.terminations.nonfinite_robot_state.params["robot_cfg"] = self._arm_entity_cfg()

        self.curriculum.reset_sampling.params.update(
            {
                "global_sampling": True,
            }
        )
        self.gripper_joint_names = _PINCH_JOINT_NAMES
