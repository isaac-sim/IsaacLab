# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reinforcement-learning configuration for Franka cube stacking."""

import math

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg
from isaaclab_newton.sim.schemas import NewtonMaterialPropertiesCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.schemas import CollisionBaseCfg, RigidBodyBaseCfg, UsdPhysicsRigidBodyCfg
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.contrib.stack import mdp
from isaaclab_tasks.contrib.stack.constants import (
    FRANKA_STACK_ARM_WORKSPACE_LOWER,
    FRANKA_STACK_ARM_WORKSPACE_UPPER,
)
from isaaclab_tasks.contrib.stack.spawners import ColoredCuboidCfg
from isaaclab_tasks.utils import SuccessMonitorCfg

from . import stack_joint_pos_env_cfg
from .franka_robot_cfg import FRANKA_PANDA_DEXSUITE_CFG


def _positive_finite(value: object, name: str) -> float:
    """Return a positive finite scalar or raise a configuration error."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number.")
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return value


def _positive_integer(value: object, name: str) -> int:
    """Return a positive integer or raise a configuration error."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


@configclass
class EventCfg:
    """Reset from a shared cache of validated full-task stack states."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_from_state_buffer = EventTerm(
        func=mdp.StackResetStateTable,
        mode="reset",
        params={
            "closed_finger_position": 0.020,
            "placed_finger_position": 0.021,
            "open_finger_position": 0.04,
            # Eighteen balanced workspace/source-order layouts each receive 64
            # independently scattered table starts. This yields 1,152 broad
            # deployment starts without making the table family dominate.
            "table_rows_per_layout": 64,
            # Cached intermediate states are authored hand-object contacts and
            # must remain exact. Deployment-like table starts receive broad
            # robot variation and coherent planar cube transforms instead.
            "arm_joint_noise_range": 0.0,
            "table_arm_joint_noise_range": 0.080,
            "table_cube_planar_translation_range": 0.015,
            "table_cube_rotation_range": 0.45,
            "fixed_recipe": None,
            "fixed_role_permutation": None,
        },
    )


@configclass
class FrankaStackStateObservationCfg(ObsGroup):
    """Canonical full-state interface for the Franka stack policy."""

    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
    )
    joint_target = ObsTerm(func=mdp.joint_position_target)
    actions = ObsTerm(func=mdp.last_action)
    object = ObsTerm(func=mdp.role_conditioned_stack_obs)
    gripper_pos = ObsTerm(func=mdp.gripper_pos)
    eef_velocity = ObsTerm(func=mdp.franka_ee_velocity)
    eef_axes = ObsTerm(func=mdp.franka_ee_axes)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class FrankaStackObservationsCfg:
    """Observation groups for state-based Franka stacking."""

    policy: FrankaStackStateObservationCfg = FrankaStackStateObservationCfg()


@configclass
class RewardsCfg:
    """Sparse terminal objective plus safety regularizers."""

    success = RewTerm(
        func=mdp.stack_success_pulse,
        params={"context_term_name": "progress_context"},
        # The reward function cancels RewardManager's step_dt integration, so
        # this is an exact +2 episode impulse at any policy frequency.
        weight=2.0,
    )
    failure = RewTerm(
        func=mdp.irrecoverable_stack_failure,
        params={"success_termination_name": "success"},
        # Preserve the established -0.0002 terminal impulse independently of
        # the policy timestep.
        weight=-2.0e-4,
    )
    action_l2 = RewTerm(func=mdp.action_term_l2, params={"action_name": "arm_action"}, weight=-1.0e-4)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-4)
    joint_vel = RewTerm(
        func=mdp.finite_joint_velocity_l2,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"]),
            "maximum_velocity": 3.0,
        },
        weight=-1.0e-4,
    )


@configclass
class CurriculumCfg:
    """Guaranteed table starts plus target-rate intermediate states."""

    reset_sampling = CurrTerm(
        func=mdp.StackResetTableCurriculum,
        params={
            "success_context_name": "learning_progress_context",
            "final_success_context_name": "progress_context",
            # Use the same shared rolling-success monitor as Lift and the
            # conveyor task. Its target-rate weights concentrate sampling near
            # 50% competence while retaining a floor at both extremes.
            "success_monitor": SuccessMonitorCfg(
                monitored_history_len=50,
                target_success_rate=0.50,
                kappa=1.0,
                temperature=1.0,
            ),
            # Preserve a deployment-facing learning stream regardless of the
            # adaptive intermediate-state distribution. The remaining 65% is
            # sampled from non-table rows by the rolling-success kernel.
            "table_sampling_probability": 0.35,
            # Keep equal layout coverage by default. The KUKA task opts into
            # one flat target-rate distribution over its active rows.
            "global_sampling": False,
        },
    )


@configclass
class FrankaCubeStackRLEnvCfg(stack_joint_pos_env_cfg.FrankaCubeStackEnvCfg):
    """Train a color-order-invariant three-cube stack from physical reset rows."""

    observations: FrankaStackObservationsCfg = FrankaStackObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        robot_init_state = self.scene.robot.init_state
        robot_semantic_tags = self.scene.robot.spawn.semantic_tags
        self.scene.robot = FRANKA_PANDA_DEXSUITE_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=robot_init_state,
        )
        self.scene.robot.spawn.semantic_tags = robot_semantic_tags
        # Newton needs a fixed rigid root to discover the Seattle table's
        # separately instanced visual and collision subtrees. Keep this
        # backend-specific override local to the RL tasks; Kuka inherits it.
        self.scene.table.spawn.rigid_props = [UsdPhysicsRigidBodyCfg(kinematic_enabled=True)]
        # The Seattle table's authored collision top is 3 mm below its visible
        # tabletop. Add an invisible native contact surface at visual z=0 so
        # cubes rest on what the policy and viewer see. The original collider
        # remains below as a fallback for the rest of the table geometry.
        self.scene.table_contact_surface = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/TableContactSurface",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.3439, 0.0, -0.02)),
            spawn=sim_utils.CuboidCfg(
                size=(1.28, 0.91, 0.04),
                visible=False,
                rigid_props=RigidBodyBaseCfg(kinematic_enabled=True),
                collision_props=CollisionBaseCfg(contact_offset=0.0, rest_offset=0.0),
                physics_material=NewtonMaterialPropertiesCfg(
                    static_friction=1.0,
                    dynamic_friction=0.8,
                    restitution=0.0,
                    torsional_friction=0.002,
                    rolling_friction=0.0001,
                    contact_stiffness=1.0e4,
                    contact_damping=200.0,
                ),
            ),
        )
        # Keep physical world gravity enabled for the arm and cubes. The arm
        # action models gravity compensation in simulation; each robot config
        # explicitly declares whether its deployment controller owns that term.
        self.sim.gravity = (0.0, 0.0, -9.81)
        self.scene.robot.spawn.rigid_props.disable_gravity = False

        # Use MJWarp only as the rigid-body constraint solver. Contact
        # generation is delegated to Newton's external collision pipeline.
        # The menagerie Franka plus three cubes reached 148 constraint rows in
        # the 4,096-environment reset distribution. Keep measured headroom so
        # contacts are never silently truncated while avoiding an oversized
        # contact buffer.
        self.sim.physics = NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                solver="newton",
                integrator="implicitfast",
                njmax=256,
                nconmax=100,
                impratio=10.0,
                cone="elliptic",
                use_mujoco_contacts=False,
            ),
            # Match the fast, stable Newton Cube Lift integration cadence:
            # two 5 ms solver steps per 10 ms physics tick. Ten 1 ms substeps
            # were stable but made collection roughly eight times slower.
            num_substeps=2,
            # Refresh external contacts for every 5 ms solver step. Reusing
            # one contact set for the full 10 ms tick is sufficient for Lift,
            # but lets a released three-cube tower drift or tip.
            collision_decimation=1,
            use_cuda_graph=True,
            collision_cfg=NewtonCollisionPipelineCfg(
                broad_phase="explicit",
                reduce_contacts=True,
            ),
            # Preserve the authored 4 cm cube and finger surfaces. A positive
            # gap creates speculative contacts before those surfaces touch;
            # fresh 5 ms collision queries make that unnecessary here.
            default_shape_cfg=NewtonShapeCfg(margin=0.0, gap=0.0),
        )
        self.scene.replicate_physics = True
        # Match Cube Lift's 50 Hz policy rate: two 100 Hz physics ticks for
        # every relative joint-position command.
        self.decimation = 2
        self.sim.render_interval = 2

        # Keep the standard Franka Lift joint-position interface, but express
        # each command as a residual around the measured reset pose. The reset
        # table contains valid grasps and near-placement states; a zero-mean
        # absolute target immediately destroys those states, whereas a zero
        # residual holds them. The larger scale restores enough joint-space
        # authority to acquire cubes from table starts.
        self.actions.arm_action = mdp.WorkspaceBoundedRelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            # At 50 Hz, a 0.05 rad measured-state residual caps the commanded
            # target slew at 2.5 rad/s. MJWarp does not hard-enforce actuator
            # velocity-limit fields.
            scale=0.05,
            max_delta=0.05,
            workspace_lower=FRANKA_STACK_ARM_WORKSPACE_LOWER,
            workspace_upper=FRANKA_STACK_ARM_WORKSPACE_UPPER,
            gravity_compensation=True,
            # The deployed FR3 controller supplies gravity compensation itself.
            controller_owns_gravity_compensation=True,
        )
        self.actions.gripper_action = mdp.ResetBufferedGripperActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
            # Give Newton 100 ms to establish reset-authored finger contacts.
            # Supported place endpoints are tagged as released, so this guard
            # cannot crush a cube that already rests on the stack.
            force_close_steps=5,
        )

        # The demonstration parent config installs a different reset event set.
        self.events = EventCfg()
        self.rewards = RewardsCfg()
        self.curriculum = CurriculumCfg()
        # Side-by-side table rows must have time for two picks, two placements,
        # and release. Near-goal rows still terminate early on success.
        self.episode_length_s = 20.0

        # Keep cube identity stable through an episode. The reset table maps
        # stack roles over all six physical cube permutations, providing order
        # invariance as data augmentation without dynamically re-sorting input
        # slots at the critical grasp-to-lift transition.
        arm_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"])
        self.scene.ee_frame = None

        # Provide one backend-neutral camera hint. The selected Kit or Newton
        # visualizer receives these values at runtime.
        self.sim.default_visualizer_cfg = VisualizerCfg(
            eye=(1.4, 1.4, 0.9),
            lookat=(0.5, 0.0, 0.1),
        )

        # Native cuboids avoid asset-specific block materials. Their standard geometry
        # owns both collision and rendering; a USD displayColor keeps the
        # colors available in both Kit and kitless Newton visualization.
        cube_colors = ((0.05, 0.15, 0.80), (0.80, 0.05, 0.05), (0.05, 0.65, 0.10))
        for cube, color in zip((self.scene.cube_1, self.scene.cube_2, self.scene.cube_3), cube_colors, strict=True):
            semantic_tags = cube.spawn.semantic_tags
            cube.spawn = ColoredCuboidCfg(
                size=(0.04, 0.04, 0.04),
                display_color=color,
                rigid_props=RigidBodyBaseCfg(disable_gravity=False),
                collision_props=CollisionBaseCfg(contact_offset=0.0, rest_offset=0.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                physics_material=NewtonMaterialPropertiesCfg(
                    static_friction=1.0,
                    dynamic_friction=0.8,
                    restitution=0.0,
                    torsional_friction=0.002,
                    rolling_friction=0.0001,
                    # Reduce visible resting overlap in a three-cube tower.
                    # Newton's 2.5 kN/m fallback shortens the 12 cm tower by
                    # roughly 0.9 mm; these gains reduce that to about 0.25 mm.
                    contact_stiffness=1.0e4,
                    contact_damping=200.0,
                ),
                semantic_tags=semantic_tags,
            )

        self.terminations.progress_context = DoneTerm(
            func=mdp.StableOrderInvariantStackGoal,
            params={
                "minimum_episode_steps": 3,
                "hold_steps": 5,
                "maximum_cube_velocity": 0.10,
                "minimum_finger_release_position": 0.023,
            },
        )
        self.terminations.learning_progress_context = DoneTerm(
            func=mdp.StackResetLearningProgress,
            params={"minimum_episode_steps": 3},
        )
        self.terminations.success = DoneTerm(
            func=mdp.success_after_minimum_horizon,
            params={
                "context_term_name": "progress_context",
                # The context already requires five consecutive stable,
                # released-stack frames. Terminate on that physical hold
                # instead of leaving a solved scene alive for five seconds,
                # which invites post-success oscillation and reward cycling.
                "minimum_episode_length_s": 0.1,
            },
        )
        self.terminations.time_out = DoneTerm(
            func=mdp.time_out,
            time_out=True,
        )
        self.terminations.nonfinite_robot_state = DoneTerm(
            func=mdp.nonfinite_robot_state,
            params={"robot_cfg": arm_cfg},
        )
        self.terminations.nonfinite_cube_state = DoneTerm(func=mdp.nonfinite_cube_state)
        self.terminations.cube_workspace_invalid = DoneTerm(
            func=mdp.cube_out_of_workspace,
        )

    def validate_config(self) -> None:
        """Validate the resolved Newton stack configuration."""
        self._validate_physics_contract()
        self._validate_action_contract()
        self._validate_scene_contract()
        self._validate_camera_contract()

    def _validate_physics_contract(self) -> None:
        """Validate Newton solver, timing, and contact-refresh invariants."""
        if not isinstance(self.sim.physics, NewtonCfg):
            raise TypeError("The Franka and KUKA stack RL tasks require the Newton physics backend.")
        if not isinstance(self.sim.physics.solver_cfg, MJWarpSolverCfg):
            raise TypeError("The stack RL contact configuration requires the Newton MJWarp solver.")
        if not self.scene.replicate_physics:
            raise ValueError("scene.replicate_physics must be enabled for vectorized stack training.")

        _positive_finite(self.sim.dt, "sim.dt")
        _positive_integer(self.decimation, "decimation")
        _positive_integer(self.sim.render_interval, "sim.render_interval")
        if self.sim.render_interval != self.decimation:
            raise ValueError("sim.render_interval must equal decimation so every policy observation can be rendered.")

        physics = self.sim.physics
        _positive_integer(physics.num_substeps, "sim.physics.num_substeps")
        _positive_integer(physics.collision_decimation, "sim.physics.collision_decimation")
        if physics.collision_decimation > physics.num_substeps:
            raise ValueError("collision_decimation cannot exceed num_substeps for contact-rich stacking.")
        _positive_integer(physics.solver_cfg.njmax, "sim.physics.solver_cfg.njmax")
        _positive_integer(physics.solver_cfg.nconmax, "sim.physics.solver_cfg.nconmax")
        if physics.solver_cfg.use_mujoco_contacts:
            raise ValueError("The stack task requires Newton's external collision pipeline for fresh contacts.")
        for field_name in ("margin", "gap"):
            value = getattr(physics.default_shape_cfg, field_name)
            if not math.isfinite(float(value)) or value < 0.0:
                raise ValueError(f"sim.physics.default_shape_cfg.{field_name} must be finite and non-negative.")

    def _validate_action_contract(self) -> None:
        """Validate that policy actions map uniquely to physical targets."""
        arm_action = self.actions.arm_action
        if not isinstance(arm_action, mdp.WorkspaceBoundedRelativeJointPositionActionCfg):
            raise TypeError("actions.arm_action must use bounded measured-state relative joint control.")
        action_scale = _positive_finite(arm_action.scale, "actions.arm_action.scale")
        maximum_delta = _positive_finite(arm_action.max_delta, "actions.arm_action.max_delta")
        if action_scale > maximum_delta:
            raise ValueError(
                "actions.arm_action.scale cannot exceed max_delta; hidden saturation aliases distinct PPO actions."
            )
        if arm_action.controller_owns_gravity_compensation and not arm_action.gravity_compensation:
            raise ValueError("Controller-owned gravity compensation requires gravity_compensation=True.")
        if len(arm_action.workspace_lower) != len(arm_action.workspace_upper) or not arm_action.workspace_lower:
            raise ValueError("The arm workspace bounds must have equal, non-zero lengths.")
        if any(
            not math.isfinite(float(lower))
            or not math.isfinite(float(upper))
            or lower + arm_action.joint_limit_margin >= upper - arm_action.joint_limit_margin
            for lower, upper in zip(arm_action.workspace_lower, arm_action.workspace_upper, strict=True)
        ):
            raise ValueError("Every arm workspace interval must be finite and wider than twice the joint-limit margin.")

        gripper_action = self.actions.gripper_action
        if isinstance(gripper_action, mdp.ResetBufferedGripperActionCfg):
            if (
                isinstance(gripper_action.force_close_steps, bool)
                or not isinstance(gripper_action.force_close_steps, int)
                or gripper_action.force_close_steps < 0
            ):
                raise ValueError("actions.gripper_action.force_close_steps must be a non-negative integer.")
        elif isinstance(gripper_action, mdp.ResetPreservingRelativeJointPositionActionCfg):
            gripper_scale = _positive_finite(gripper_action.scale, "actions.gripper_action.scale")
            gripper_maximum_delta = _positive_finite(
                gripper_action.max_delta,
                "actions.gripper_action.max_delta",
            )
            if gripper_scale > gripper_maximum_delta:
                raise ValueError(
                    "actions.gripper_action.scale cannot exceed max_delta; hidden saturation aliases distinct "
                    "PPO actions."
                )
        else:
            raise TypeError("The stack gripper action must preserve reset-authored grasp state.")

    def _validate_scene_contract(self) -> None:
        """Validate cube semantics and explicit Newton contact materials."""
        for cube_name in ("cube_1", "cube_2", "cube_3"):
            cube = getattr(self.scene, cube_name)
            if not cube.spawn.semantic_tags:
                raise ValueError(f"scene.{cube_name}.spawn.semantic_tags must identify the cube for perception tools.")
            material = cube.spawn.physics_material
            if not isinstance(material, NewtonMaterialPropertiesCfg):
                raise TypeError(f"scene.{cube_name} must use an explicit Newton contact material.")
            _positive_finite(material.contact_stiffness, f"scene.{cube_name}.contact_stiffness")
            _positive_finite(material.contact_damping, f"scene.{cube_name}.contact_damping")

        table_surface = getattr(self.scene, "table_contact_surface", None)
        if table_surface is None:
            raise ValueError("scene.table_contact_surface must align contacts with the visible tabletop.")
        table_material = table_surface.spawn.physics_material
        if not isinstance(table_material, NewtonMaterialPropertiesCfg):
            raise TypeError("scene.table_contact_surface must use an explicit Newton contact material.")
        _positive_finite(table_material.contact_stiffness, "scene.table_contact_surface.contact_stiffness")
        _positive_finite(table_material.contact_damping, "scene.table_contact_surface.contact_damping")

    def _validate_camera_contract(self) -> None:
        """Validate the reset-safe RGB observation cadence when a camera exists."""
        camera = getattr(self.scene, "base_camera", None)
        if camera is not None:
            if camera.update_period != 0.0:
                raise ValueError("The policy camera update_period must be zero so it refreshes at every render.")
            _positive_integer(camera.height, "scene.base_camera.height")
            _positive_integer(camera.width, "scene.base_camera.width")
            if "rgb" not in camera.data_types:
                raise ValueError("The camera stack actor requires RGB observations.")
            if self.num_rerenders_on_reset < 1:
                raise ValueError("Camera stack tasks require num_rerenders_on_reset >= 1 to avoid stale reset frames.")

    def play_mode(self) -> None:
        """Configure randomized table starts for policy evaluation."""
        super().play_mode()
        self.scene.env_spacing = 2.5
        self.episode_length_s = 30.0
        # Sample only from the randomized table partition. Avoid a brittle
        # numeric row ID so evaluation follows cache-size changes.
        self.events.reset_from_state_buffer.params["fixed_recipe"] = int(mdp.StackResetRecipe.TABLE)
        # Evaluation and export start from table rows, so reset-only gripper
        # assistance must not mask the policy action in the traced graph.
        if isinstance(self.actions.gripper_action, mdp.ResetBufferedGripperActionCfg):
            self.actions.gripper_action.force_close_steps = 0
        self.curriculum = None
