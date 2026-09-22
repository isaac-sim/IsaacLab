# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import time

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import matrix_from_quat

from .rmp_flow_cfg import RmpFlowControllerCfg  # noqa: F401
from .utils import import_lula, resolve_rmpflow_path


class _LulaRmpFlow:
    """RMPFlow policy backed directly by the ``lula`` library.

    This mirrors the surface of ``isaacsim.robot_motion.motion_generation``'s ``RmpFlow`` that
    :class:`RmpFlowController` depends on (construction, active/watched joints, end-effector target,
    Euler integration of the policy acceleration), backed by the same ``lula`` library but without
    going through the Isaac Sim Kit motion-generation extension. This keeps a single code path across
    backends -- Kit and kitless (e.g. the Newton visualizer) -- since ``lula`` is importable in both.
    It assumes IsaacLab conventions: a meter-scaled stage and an identity robot base pose (the
    controller never relocates the base), and it does not manage obstacles or visuals.

    Note:
        ``lula`` is deprecated as of Isaac Sim 6.0 and slated for removal in a future release; cuMotion
        (``isaacsim.robot_motion.cumotion``) is the long-term replacement. See the migration notes at
        :data:`isaaclab.controllers.utils._LULA_EXT_NAME`.
    """

    def __init__(
        self,
        robot_description_path: str,
        urdf_path: str,
        rmpflow_config_path: str,
        end_effector_frame_name: str,
        maximum_substep_size: float,
        ignore_robot_state_updates: bool = False,
    ) -> None:
        """Load the robot description and world, then build the lula RMPFlow policy."""
        lula = import_lula()

        if maximum_substep_size <= 0:
            raise ValueError("maximum_substep_size argument must be positive.")

        self._lula = lula
        self.maximum_substep_size = maximum_substep_size
        self.ignore_robot_state_updates = ignore_robot_state_updates
        self.end_effector_frame_name = end_effector_frame_name
        self._rmpflow_config_path = rmpflow_config_path

        self._robot_description = lula.load_robot(robot_description_path, urdf_path)
        self._world = lula.create_world()
        self._build_policy()

        self._robot_joint_positions: np.ndarray | None = None
        self._robot_joint_velocities: np.ndarray | None = None

    def _build_policy(self) -> None:
        """(Re)create the lula RMPFlow policy from the configured robot and config files."""
        config = self._lula.create_rmpflow_config(
            self._rmpflow_config_path,
            self._robot_description,
            self.end_effector_frame_name,
            self._world.add_world_view(),
        )
        self._policy = self._lula.create_rmpflow(config)

    def get_active_joints(self) -> list[str]:
        """Return the names of the joints the policy actively controls (its c-space coords)."""
        rd = self._robot_description
        return [rd.c_space_coord_name(i) for i in range(rd.num_c_space_coords())]

    def get_watched_joints(self) -> list[str]:
        """Return the watched (non-controlled) joints; always empty since lula only watches active joints."""
        return []

    def reset(self) -> None:
        """Rebuild the policy and clear the cached internal joint state."""
        self._build_policy()
        self._robot_joint_positions = None
        self._robot_joint_velocities = None

    def set_end_effector_target(self, target_position=None, target_orientation=None) -> None:
        """Set (or clear) the end-effector position/orientation attractors from the target pose."""
        # identity base pose + meter stage -> targets pass through unchanged
        if target_position is None and target_orientation is None:
            self._policy.clear_end_effector_position_attractor()
            self._policy.clear_end_effector_orientation_attractor()
            return
        if target_position is not None:
            self._policy.set_end_effector_position_attractor(np.asarray(target_position, dtype=np.float64))
        else:
            self._policy.clear_end_effector_position_attractor()
        if target_orientation is not None:
            # lula expects a rotation matrix; IsaacLab quaternions are already (x, y, z, w) convention.
            rot = matrix_from_quat(torch.as_tensor(target_orientation, dtype=torch.float64)).numpy()
            self._policy.set_end_effector_orientation_attractor(self._lula.Rotation3(rot))
        else:
            self._policy.clear_end_effector_orientation_attractor()

    def compute_joint_targets(
        self,
        active_joint_positions: np.ndarray,
        active_joint_velocities: np.ndarray,
        watched_joint_positions: np.ndarray,
        watched_joint_velocities: np.ndarray,
        frame_duration: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Integrate the policy over one frame and return the next joint position/velocity targets."""
        if (
            self._robot_joint_positions is None
            or self._robot_joint_velocities is None
            or not self.ignore_robot_state_updates
        ):
            positions = np.array(active_joint_positions, dtype=np.float64)
            velocities = np.array(active_joint_velocities, dtype=np.float64)
        else:
            positions = self._robot_joint_positions
            velocities = self._robot_joint_velocities

        self._robot_joint_positions, self._robot_joint_velocities = self._euler_integration(
            positions, velocities, frame_duration
        )
        return self._robot_joint_positions, self._robot_joint_velocities

    def _evaluate_acceleration(self, joint_positions: np.ndarray, joint_velocities: np.ndarray) -> np.ndarray:
        """Query the lula policy for the joint-space acceleration at the given state."""
        joint_positions = np.asarray(joint_positions, dtype=np.float64)
        joint_velocities = np.asarray(joint_velocities, dtype=np.float64)
        joint_accel = np.zeros_like(joint_positions)
        self._policy.eval_accel(joint_positions, joint_velocities, joint_accel)
        return joint_accel

    def _euler_integration(
        self, joint_positions: np.ndarray, joint_velocities: np.ndarray, frame_duration: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Roll out the policy acceleration with fixed-substep Euler integration over the frame."""
        num_steps = int(np.ceil(frame_duration / self.maximum_substep_size))
        policy_timestep = frame_duration / num_steps
        for _ in range(num_steps):
            joint_accel = self._evaluate_acceleration(joint_positions, joint_velocities)
            joint_positions = joint_positions + policy_timestep * joint_velocities
            joint_velocities = joint_velocities + policy_timestep * joint_accel
        return joint_positions, joint_velocities


class _LulaRmpFlowSmoothed(_LulaRmpFlow):
    """Jerk-smoothed RMPFlow variant, ported from the Kit ``RmpFlowSmoothed`` policy.

    Adds wall-clock jerk monitoring on top of :class:`_LulaRmpFlow`: a large acceleration jump
    triggers a speed reduction and medium jumps are truncated, which smooths motion on physical
    robots. The smoothing math is pure NumPy, so this variant -- like the base policy -- runs without
    the Isaac Sim Kit motion-generation extension.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the base policy and the jerk-smoothing state and tunables."""
        super().__init__(*args, **kwargs)
        self._reset_smoothing_state()
        # tunables (mirror the Kit ``RmpFlowSmoothed`` defaults)
        self.min_time_between_jerk_reductions = 0.5
        self.min_speed_scalar = 0.2
        self.use_big_jerk_speed_scaling = True
        self.big_jerk_limit = 10.0
        self.use_medium_jerk_truncation = True
        self.max_medium_jerk = 7.0
        self.speed_scalar_alpha_blend = 0.985

    def _reset_smoothing_state(self) -> None:
        """Reset the speed scalar and cached acceleration/jerk-timing used for smoothing."""
        self.desired_speed_scalar = 1.0
        self.speed_scalar = 1.0
        self.time_at_last_jerk_reduction: float | None = None
        self.qdd: np.ndarray | None = None

    def reset(self) -> None:
        """Rebuild the policy (base reset) and clear the jerk-smoothing state."""
        super().reset()
        self._reset_smoothing_state()

    def _eval_speed_scaled_accel(self, joint_positions: np.ndarray, joint_velocities: np.ndarray) -> np.ndarray:
        """Evaluate acceleration with velocity/time rescaled by the current speed scalar."""
        qdd_eval = self._evaluate_acceleration(joint_positions, joint_velocities / self.speed_scalar)
        qdd_eval *= self.speed_scalar**2
        return qdd_eval

    def _euler_integration(
        self, joint_positions: np.ndarray, joint_velocities: np.ndarray, frame_duration: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Roll out the policy like the base class, but monitor and reduce/truncate jerk per substep."""
        num_steps = int(np.ceil(frame_duration / self.maximum_substep_size))
        step_dt = frame_duration / num_steps

        q = np.array(joint_positions, dtype=np.float64)
        qd = np.array(joint_velocities, dtype=np.float64)

        # jerk monitoring is meant for physical robots, so it uses wall-clock time (matching Kit)
        now = time.time()

        for _ in range(num_steps):
            if self.qdd is None:
                self.qdd = self._eval_speed_scaled_accel(q, qd)
                continue

            jerk_reduction_performed = False

            # reduce speed to the minimum when a big jerk is experienced
            if self.use_big_jerk_speed_scaling:
                is_first = True
                while True:
                    qdd_eval = self._eval_speed_scaled_accel(q, qd)
                    # only evaluate the reduction once; the loop re-evals qdd after the reduction
                    if not is_first:
                        break
                    if (
                        self.time_at_last_jerk_reduction is not None
                        and (now - self.time_at_last_jerk_reduction) < self.min_time_between_jerk_reductions
                    ):
                        break
                    jerk = np.linalg.norm(qdd_eval - self.qdd)
                    if jerk > self.big_jerk_limit:
                        self.speed_scalar = self.min_speed_scalar
                        jerk_reduction_performed = True
                    is_first = False

            # truncate transient medium jerks
            if self.use_medium_jerk_truncation:
                qdd_eval = self._eval_speed_scaled_accel(q, qd)
                jerk = np.linalg.norm(qdd_eval - self.qdd)
                if jerk > self.max_medium_jerk:
                    direction = (qdd_eval - self.qdd) / jerk
                    qdd_eval = self.qdd + self.max_medium_jerk * direction

            if jerk_reduction_performed:
                self.time_at_last_jerk_reduction = now

            self.qdd = qdd_eval

            a = self.speed_scalar_alpha_blend
            self.speed_scalar = a * self.speed_scalar + (1.0 - a) * self.desired_speed_scalar

            q = q + step_dt * qd
            qd = qd + step_dt * self.qdd

        return q, qd


class RmpFlowController:
    """Wraps around RMPFlow from IsaacSim for batched environments."""

    def __init__(self, cfg: RmpFlowControllerCfg, device: str):
        """Initialize the controller.

        Args:
            cfg: The configuration for the controller.
            device: The device to use for computation.
        """
        self.cfg = cfg
        self._device = device
        print(f"[INFO]: Loading RMPFlow controller URDF from: {self.cfg.urdf_file}")

    """
    Properties.
    """

    @property
    def num_actions(self) -> int:
        """Dimension of the action space of controller."""
        return 7

    """
    Operations.
    """

    def initialize(self, num_robots: int, joint_names: list[str]):
        """Initialize the controller.

        Args:
            num_robots: Number of robot instances (environments).
            joint_names: Ordered list of all joint names from the articulation.
        """
        physics_dt = sim_utils.SimulationContext.instance().get_physics_dt()
        self.num_robots = num_robots
        self._physics_dt = physics_dt

        if self.cfg.name == "rmp_flow":
            controller_cls = _LulaRmpFlow
        elif self.cfg.name == "rmp_flow_smoothed":
            controller_cls = _LulaRmpFlowSmoothed
        else:
            raise ValueError(f"Unsupported controller in Lula library: {self.cfg.name}")

        name_to_idx = {name: i for i, name in enumerate(joint_names)}

        self._rmpflow_policies: list[_LulaRmpFlow] = []
        self._active_indices: list[np.ndarray] = []
        self._watched_indices: list[np.ndarray] = []

        for _ in range(num_robots):
            local_urdf_file = retrieve_file_path(resolve_rmpflow_path(self.cfg.urdf_file), force_download=True)
            local_collision_file = retrieve_file_path(
                resolve_rmpflow_path(self.cfg.collision_file), force_download=True
            )
            local_config_file = retrieve_file_path(resolve_rmpflow_path(self.cfg.config_file), force_download=True)

            rmpflow = controller_cls(
                robot_description_path=local_collision_file,
                urdf_path=local_urdf_file,
                rmpflow_config_path=local_config_file,
                end_effector_frame_name=self.cfg.frame_name,
                maximum_substep_size=physics_dt / self.cfg.evaluations_per_frame,
                ignore_robot_state_updates=self.cfg.ignore_robot_state_updates,
            )

            active_indices = np.array([name_to_idx[n] for n in rmpflow.get_active_joints()], dtype=np.intp)
            watched_indices = np.array([name_to_idx[n] for n in rmpflow.get_watched_joints()], dtype=np.intp)

            self._rmpflow_policies.append(rmpflow)
            self._active_indices.append(active_indices)
            self._watched_indices.append(watched_indices)

        self.active_dof_names = self._rmpflow_policies[0].get_active_joints()
        self.num_dof = len(self.active_dof_names)

        self._command = torch.zeros(self.num_robots, self.num_actions, device=self._device)
        self.dof_pos_target = torch.zeros((self.num_robots, self.num_dof), device=self._device)
        self.dof_vel_target = torch.zeros((self.num_robots, self.num_dof), device=self._device)

    def reset_idx(self, robot_ids: torch.Tensor | None = None):
        """Reset the internals."""
        if robot_ids is None:
            robot_ids = torch.arange(self.num_robots, device=self._device)
        for index in robot_ids:
            self._rmpflow_policies[index].reset()

    def set_command(self, command: torch.Tensor):
        """Set target end-effector pose command."""
        self._command[:] = command

    def compute(
        self, joint_positions: torch.Tensor, joint_velocities: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs inference with the controller.

        Args:
            joint_positions: Current joint positions, shape ``[num_robots, num_joints]``.
            joint_velocities: Current joint velocities, shape ``[num_robots, num_joints]``.

        Returns:
            The target joint positions and velocity commands.
        """
        command = self._command.cpu().numpy()
        all_pos = joint_positions.cpu().numpy()
        all_vel = joint_velocities.cpu().numpy()

        for i, rmpflow in enumerate(self._rmpflow_policies):
            rmpflow.set_end_effector_target(target_position=command[i, 0:3], target_orientation=command[i, 3:7])
            active_pos = all_pos[i][self._active_indices[i]]
            active_vel = all_vel[i][self._active_indices[i]]
            watched_pos = all_pos[i][self._watched_indices[i]]
            watched_vel = all_vel[i][self._watched_indices[i]]

            pos_targets, vel_targets = rmpflow.compute_joint_targets(
                active_pos, active_vel, watched_pos, watched_vel, self._physics_dt
            )
            self.dof_pos_target[i, :] = torch.from_numpy(pos_targets[:]).to(self.dof_pos_target)
            self.dof_vel_target[i, :] = torch.from_numpy(vel_targets[:]).to(self.dof_vel_target)

        return self.dof_pos_target, self.dof_vel_target
