# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import tempfile

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_tasks.contrib import ISAACLAB_TELEOP_AVAILABLE
from isaaclab_tasks.contrib.packing_table import configure_packing_table_collisions
from isaaclab_tasks.contrib.teleop_physics_cfg import (
    TeleopPhysicsCfg,
    kitless_backend_prepared,
    prepare_humanoid_teleop_environment,
)

from .pickplace_gr1t2_env_cfg import (
    ActionsCfg,
    EventCfg,
    ObjectTableSceneCfg,
    ObservationsCfg,
    TerminationsCfg,
    apply_kitless_ovphysx_drive_gains,
    build_gr1t2_pickplace_pipeline,
    condition_kitless_gr1t2_stage,
    condition_kitless_ovphysx_robot_spawn_properties,
    prepare_gr1t2_for_kitless_newton_mjwarp,
)


@configclass
class PickPlaceGR1T2WaistEnabledEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the GR1T2 environment."""

    class_type: type | str = "isaaclab_tasks.contrib.teleop_env:HumanoidTeleopEnv"
    _prepared_kitless_backend: str | None = None
    haptic_feedback: object | None = None
    sim: SimulationCfg = SimulationCfg(physics=TeleopPhysicsCfg())

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()

    # Unused managers
    commands = None
    rewards = None
    curriculum = None

    # Temporary directory for URDF files
    temp_urdf_dir = tempfile.gettempdir()

    def __post_init__(self):
        """Post initialization."""
        if self._prepared_kitless_backend is not None:
            return
        # general settings
        self.decimation = 6
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1 / 120  # 120Hz
        self.sim.render_interval = 2

        # Add waist joint to pink_ik_cfg
        waist_joint_names = ["waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"]
        for joint_name in waist_joint_names:
            if joint_name not in self.actions.upper_body_ik.pink_controlled_joint_names:
                self.actions.upper_body_ik.pink_controlled_joint_names.append(joint_name)

        # Defer USD→URDF conversion to controller initialization (requires Isaac Sim at runtime).
        self.actions.upper_body_ik.controller.usd_path = self.scene.robot.spawn.usd_path
        self.actions.upper_body_ik.controller.urdf_output_dir = self.temp_urdf_dir

        if ISAACLAB_TELEOP_AVAILABLE:
            from isaaclab_teleop import GloveHapticFeedbackCfg, IsaacTeleopCfg, XrCfg

            # Isaac Capture-based teleoperation pipeline.
            self.xr = XrCfg(
                anchor_pos=(0.0, 0.0, 0.0),
                anchor_rot=(0.0, 0.0, 0.0, 1.0),
                # Exclude head visuals from the operator camera, not the source scene.
                excluded_visual_meshes=["/World/envs/env_0/Robot/head_.*_link/visuals/.*"],
            )
            self.isaac_teleop = IsaacTeleopCfg(
                prepare_environment=prepare_humanoid_teleop_environment,
                pipeline_builder=lambda: build_gr1t2_pickplace_pipeline()[0],
                sim_device=self.sim.device,
                xr_cfg=self.xr,
            )

            # Per-finger haptic glove feedback: vibrate each finger of the operator's
            # glove in proportion to how tightly it grips the object. The session
            # always requests the push-tensor extension the glove device needs, so
            # this stays inert (no glove connected) rather than failing.
            self.haptic_feedback = GloveHapticFeedbackCfg(
                left_sensor_name="left_hand_contact",
                right_sensor_name="right_hand_contact",
            )

    def stabilize_for_kitless_ovphysx(self) -> None:
        """Prepare the existing GR asset while retaining the movable waist."""
        if kitless_backend_prepared(self, "ovphysx"):
            condition_kitless_gr1t2_stage(self.scene.robot, self.actions.upper_body_ik)
            return
        action_cfg = self.actions.upper_body_ik
        action_cfg.controller.urdf_path = os.path.join(
            os.path.dirname(__file__), "assets", "GR1T2_fourier_hand_6dof_kinematics.urdf"
        )
        action_cfg.kitless_target_conditioning_enabled = True
        if self.isaac_teleop is not None:
            self.isaac_teleop.require_initial_bimanual_hand_tracking = True
        apply_kitless_ovphysx_drive_gains(self.scene.robot)
        condition_kitless_ovphysx_robot_spawn_properties(self.scene.robot)
        condition_kitless_gr1t2_stage(self.scene.robot, action_cfg)
        configure_packing_table_collisions(self.scene.packing_table, physx_offsets=True)
        self._prepared_kitless_backend = "ovphysx"

    def stabilize_for_kitless_newton_mjwarp(self) -> None:
        """Configure GR1T2 actuators and contacts for kitless Newton."""
        prepare_gr1t2_for_kitless_newton_mjwarp(self)
