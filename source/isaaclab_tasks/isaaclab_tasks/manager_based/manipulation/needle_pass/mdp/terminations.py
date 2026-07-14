# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measured-contact phase state and terminations for dVRK needle pass."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

JAW_CONTACT_SENSOR_NAMES = (
    "left_jaw_1_needle_contact",
    "left_jaw_2_needle_contact",
    "right_jaw_1_needle_contact",
    "right_jaw_2_needle_contact",
)
"""Stable left-to-right order used by observations, phases, and recordings."""

JAW_BODY_REACTION_NORMALS_LOCAL = (
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
)
"""Link-local compressive reaction axes pointing from each jaw face into its solid.

The active convex inner-face surface normals point out of the jaw solids and
into the channel: jaw 1 ``-X``, jaw 2 ``+X``.  A needle contact force acting on
the sensor body has the opposite sign, so these body-reaction projection axes
are jaw 1 ``+X`` and jaw 2 ``-X``.  This preserves the contract
``max(0, dot(F_w, n_w))`` for physical compression.
"""


class HandoffPhase(IntEnum):
    """Ordered physical progress of one needle hand-off.

    ``INITIAL`` is the donor-held reset state awaiting a fresh measured-contact
    dwell.  It does not mean the needle is geometrically ungrasped.
    """

    INITIAL = 0
    DONOR_HOLD = 1
    CO_HOLD = 2
    RECEIVER_ONLY_HOLD = 3
    RETAINED_LIFT = 4


@configclass
class HandoffPhaseCfg:
    """Thresholds for the contact-driven hand-off state machine."""

    engage_force_n: float = 1.0e-4
    disengage_force_n: float = 5.0e-5
    opposed_normal_tolerance_rad: float = math.radians(20.0)
    donor_dwell_s: float = 8.0 / 240.0
    co_hold_dwell_s: float = 8.0 / 240.0
    receiver_only_dwell_s: float = 8.0 / 240.0
    retained_lift_dwell_s: float = 10.0 / 240.0
    receiver_relative_position_target_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    receiver_relative_orientation_target_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    receiver_relative_position_limit_m: float = 0.035
    receiver_relative_orientation_limit_rad: float = math.radians(60.0)
    maximum_linear_velocity_m_s: float = 0.10
    maximum_angular_velocity_rad_s: float = 5.0
    required_lift_delta_z_m: float = 0.015

    def __post_init__(self) -> None:
        if not 0.0 <= self.disengage_force_n < self.engage_force_n:
            raise ValueError("contact hysteresis requires 0 <= disengage < engage")
        if not 0.0 < self.opposed_normal_tolerance_rad < math.pi:
            raise ValueError("opposed_normal_tolerance_rad must lie in (0, pi)")
        receiver_position_target = torch.tensor(self.receiver_relative_position_target_m)
        receiver_orientation_target = torch.tensor(self.receiver_relative_orientation_target_wxyz)
        if receiver_position_target.shape != (3,) or not torch.isfinite(receiver_position_target).all():
            raise ValueError("receiver relative position target must be a finite three-vector")
        if (
            receiver_orientation_target.shape != (4,)
            or not torch.isfinite(receiver_orientation_target).all()
            or torch.linalg.vector_norm(receiver_orientation_target) <= 1.0e-9
        ):
            raise ValueError("receiver relative orientation target must be a normalisable wxyz quaternion")
        positive_values = (
            self.donor_dwell_s,
            self.co_hold_dwell_s,
            self.receiver_only_dwell_s,
            self.retained_lift_dwell_s,
            self.receiver_relative_position_limit_m,
            self.receiver_relative_orientation_limit_rad,
            self.maximum_linear_velocity_m_s,
            self.maximum_angular_velocity_rad_s,
            self.required_lift_delta_z_m,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive_values):
            raise ValueError("phase dwell, pose, velocity, and lift limits must be finite and positive")


@dataclass(slots=True)
class HandoffMeasurements:
    """One post-physics batch consumed by :class:`HandoffPhaseMachine`."""

    normal_forces_n: torch.Tensor
    reaction_normals_w: torch.Tensor
    needle_pose_w: torch.Tensor
    needle_velocity_w: torch.Tensor
    receiver_pose_w: torch.Tensor


class HandoffPhaseMachine:
    """Vectorised, stateful, measured-contact hand-off evaluator.

    The machine reads filtered normal contact loads and simulated poses.  It
    never reads the commanded action.  All counters and hysteresis state are
    per environment and support partial resets.
    """

    def __init__(self, num_envs: int, device: str, step_dt: float, cfg: HandoffPhaseCfg):
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        if not math.isfinite(step_dt) or step_dt <= 0.0:
            raise ValueError("step_dt must be finite and positive")
        self.num_envs = num_envs
        self.device = device
        self.step_dt = step_dt
        self.cfg = cfg
        self.phase = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._donor_engaged = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._receiver_engaged = torch.zeros_like(self._donor_engaged)
        self._donor_counter = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._co_hold_counter = torch.zeros_like(self._donor_counter)
        self._receiver_only_counter = torch.zeros_like(self._donor_counter)
        self._retained_lift_counter = torch.zeros_like(self._donor_counter)
        self.reset_needle_z_w = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self._last_step_token = torch.full((num_envs,), -1, dtype=torch.long, device=device)

    def _required_steps(self, duration_s: float) -> int:
        return max(1, math.ceil(duration_s / self.step_dt - 1.0e-12))

    def reset(
        self,
        env_ids: torch.Tensor,
        reset_needle_z_w: torch.Tensor,
        step_token: int | None = None,
    ) -> None:
        """Reset to donor-held INITIAL and await a fresh post-action contact sample."""

        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        reset_z = reset_needle_z_w.to(device=self.device, dtype=torch.float32).reshape(-1)
        if reset_z.shape[0] != env_ids.shape[0] or not torch.isfinite(reset_z).all():
            raise ValueError("reset heights must be one finite value per environment")
        self.phase[env_ids] = int(HandoffPhase.INITIAL)
        self._donor_engaged[env_ids] = False
        self._receiver_engaged[env_ids] = False
        self._donor_counter[env_ids] = 0
        self._co_hold_counter[env_ids] = 0
        self._receiver_only_counter[env_ids] = 0
        self._retained_lift_counter[env_ids] = 0
        self.reset_needle_z_w[env_ids] = reset_z
        self._last_step_token[env_ids] = -1 if step_token is None else int(step_token)

    def _bilateral_contact(
        self,
        loads: torch.Tensor,
        normals: torch.Tensor,
        engaged: torch.Tensor,
    ) -> torch.Tensor:
        threshold = torch.where(
            engaged,
            torch.full_like(loads[:, 0], self.cfg.disengage_force_n),
            torch.full_like(loads[:, 0], self.cfg.engage_force_n),
        )
        force_ok = torch.logical_and(loads[:, 0] >= threshold, loads[:, 1] >= threshold)
        unit_normals = torch.nn.functional.normalize(normals, dim=-1, eps=1.0e-12)
        normal_dot = torch.sum(unit_normals[:, 0] * unit_normals[:, 1], dim=-1)
        opposed = normal_dot <= -math.cos(self.cfg.opposed_normal_tolerance_rad)
        finite = torch.isfinite(loads).all(dim=-1) & torch.isfinite(normals).all(dim=(-2, -1))
        return finite & force_ok & opposed

    def _receiver_bounds(self, measurements: HandoffMeasurements) -> torch.Tensor:
        needle_pos_r, needle_quat_r = math_utils.subtract_frame_transforms(
            measurements.receiver_pose_w[:, :3],
            measurements.receiver_pose_w[:, 3:7],
            measurements.needle_pose_w[:, :3],
            measurements.needle_pose_w[:, 3:7],
        )
        position_target = torch.tensor(
            self.cfg.receiver_relative_position_target_m,
            dtype=needle_pos_r.dtype,
            device=needle_pos_r.device,
        )
        relative_position_ok = torch.linalg.vector_norm(needle_pos_r - position_target, dim=-1) <= (
            self.cfg.receiver_relative_position_limit_m
        )
        unit_quat = torch.nn.functional.normalize(needle_quat_r, dim=-1, eps=1.0e-12)
        orientation_target = torch.tensor(
            self.cfg.receiver_relative_orientation_target_wxyz,
            dtype=unit_quat.dtype,
            device=unit_quat.device,
        ).repeat(self.num_envs, 1)
        orientation_target = torch.nn.functional.normalize(orientation_target, dim=-1, eps=1.0e-12)
        relative_angle = math_utils.quat_error_magnitude(unit_quat, orientation_target)
        relative_orientation_ok = relative_angle <= self.cfg.receiver_relative_orientation_limit_rad
        linear_velocity_ok = torch.linalg.vector_norm(measurements.needle_velocity_w[:, :3], dim=-1) <= (
            self.cfg.maximum_linear_velocity_m_s
        )
        angular_velocity_ok = torch.linalg.vector_norm(measurements.needle_velocity_w[:, 3:6], dim=-1) <= (
            self.cfg.maximum_angular_velocity_rad_s
        )
        needle_quaternion_valid = torch.linalg.vector_norm(measurements.needle_pose_w[:, 3:7], dim=-1) > 1.0e-9
        receiver_quaternion_valid = torch.linalg.vector_norm(measurements.receiver_pose_w[:, 3:7], dim=-1) > 1.0e-9
        finite = torch.logical_and(
            torch.isfinite(measurements.needle_pose_w).all(dim=-1),
            torch.isfinite(measurements.needle_velocity_w).all(dim=-1),
        )
        finite = finite & torch.isfinite(measurements.receiver_pose_w).all(dim=-1)
        return (
            finite
            & needle_quaternion_valid
            & receiver_quaternion_valid
            & relative_position_ok
            & relative_orientation_ok
            & linear_velocity_ok
            & angular_velocity_ok
        )

    @staticmethod
    def _count_consecutive(counter: torch.Tensor, condition: torch.Tensor, mask: torch.Tensor) -> None:
        counter[mask] = torch.where(condition[mask], counter[mask] + 1, torch.zeros_like(counter[mask]))

    def _clear_progress(self, mask: torch.Tensor, *, after_phase: HandoffPhase) -> None:
        """Clear counters downstream of a physical rollback.

        Completed dwell counters are retained only when their corresponding
        measured phase remains established.  This prevents a partial dwell
        before contact loss from being combined with a later, disjoint dwell.
        """

        if after_phase < HandoffPhase.DONOR_HOLD:
            self._donor_counter[mask] = 0
        if after_phase < HandoffPhase.CO_HOLD:
            self._co_hold_counter[mask] = 0
        if after_phase < HandoffPhase.RECEIVER_ONLY_HOLD:
            self._receiver_only_counter[mask] = 0
        if after_phase < HandoffPhase.RETAINED_LIFT:
            self._retained_lift_counter[mask] = 0

    def advance(self, measurements: HandoffMeasurements, step_token: int) -> torch.Tensor:
        """Advance each environment at most once for one simulator step token."""

        if measurements.normal_forces_n.shape != (self.num_envs, 4):
            raise ValueError("normal_forces_n must have shape (num_envs, 4)")
        if measurements.reaction_normals_w.shape != (self.num_envs, 4, 3):
            raise ValueError("reaction_normals_w must have shape (num_envs, 4, 3)")
        if measurements.needle_pose_w.shape != (self.num_envs, 7):
            raise ValueError("needle_pose_w must have shape (num_envs, 7)")
        if measurements.needle_velocity_w.shape != (self.num_envs, 6):
            raise ValueError("needle_velocity_w must have shape (num_envs, 6)")
        if measurements.receiver_pose_w.shape != (self.num_envs, 7):
            raise ValueError("receiver_pose_w must have shape (num_envs, 7)")
        active = self._last_step_token != int(step_token)
        self._last_step_token[active] = int(step_token)

        donor = self._bilateral_contact(
            measurements.normal_forces_n[:, 0:2],
            measurements.reaction_normals_w[:, 0:2],
            self._donor_engaged,
        )
        receiver = self._bilateral_contact(
            measurements.normal_forces_n[:, 2:4],
            measurements.reaction_normals_w[:, 2:4],
            self._receiver_engaged,
        )
        self._donor_engaged[active] = donor[active]
        self._receiver_engaged[active] = receiver[active]
        receiver_bounds = self._receiver_bounds(measurements)

        initial = active & (self.phase == int(HandoffPhase.INITIAL))
        self._count_consecutive(self._donor_counter, donor, initial)
        donor_complete = initial & (self._donor_counter >= self._required_steps(self.cfg.donor_dwell_s))
        self.phase[donor_complete] = int(HandoffPhase.DONOR_HOLD)

        donor_phase = active & (self.phase == int(HandoffPhase.DONOR_HOLD)) & ~donor_complete
        donor_lost = donor_phase & ~donor
        self.phase[donor_lost] = int(HandoffPhase.INITIAL)
        self._clear_progress(donor_lost, after_phase=HandoffPhase.INITIAL)
        donor_phase = donor_phase & donor
        self._count_consecutive(self._co_hold_counter, donor & receiver, donor_phase)
        co_hold_complete = donor_phase & (self._co_hold_counter >= self._required_steps(self.cfg.co_hold_dwell_s))
        self.phase[co_hold_complete] = int(HandoffPhase.CO_HOLD)

        co_hold_phase = active & (self.phase == int(HandoffPhase.CO_HOLD)) & ~co_hold_complete
        receiver_lost = co_hold_phase & ~receiver
        receiver_lost_to_donor = receiver_lost & donor
        receiver_lost_to_initial = receiver_lost & ~donor
        self.phase[receiver_lost_to_donor] = int(HandoffPhase.DONOR_HOLD)
        self._clear_progress(receiver_lost_to_donor, after_phase=HandoffPhase.DONOR_HOLD)
        self.phase[receiver_lost_to_initial] = int(HandoffPhase.INITIAL)
        self._clear_progress(receiver_lost_to_initial, after_phase=HandoffPhase.INITIAL)
        receiver_only_condition = ~donor & receiver & receiver_bounds
        self._count_consecutive(self._receiver_only_counter, receiver_only_condition, co_hold_phase & receiver)
        receiver_only_complete = co_hold_phase & (
            self._receiver_only_counter >= self._required_steps(self.cfg.receiver_only_dwell_s)
        )
        self.phase[receiver_only_complete] = int(HandoffPhase.RECEIVER_ONLY_HOLD)

        receiver_phase = active & (self.phase == int(HandoffPhase.RECEIVER_ONLY_HOLD)) & ~receiver_only_complete
        # Receiver-only ownership is a contact fact. A transient pose or
        # velocity excursion while both recipient jaw loads remain bilateral
        # must not reclassify the free needle as unheld: the bounds still gate
        # retained-lift success below. Only a measured loss or reversal of
        # physical ownership rolls this phase back.
        receiver_regrasped_by_donor = receiver_phase & donor & receiver
        receiver_lost_to_donor = receiver_phase & donor & ~receiver
        receiver_lost_to_initial = receiver_phase & ~donor & ~receiver
        self.phase[receiver_regrasped_by_donor] = int(HandoffPhase.CO_HOLD)
        self._clear_progress(receiver_regrasped_by_donor, after_phase=HandoffPhase.CO_HOLD)
        self.phase[receiver_lost_to_donor] = int(HandoffPhase.DONOR_HOLD)
        self._clear_progress(receiver_lost_to_donor, after_phase=HandoffPhase.DONOR_HOLD)
        self.phase[receiver_lost_to_initial] = int(HandoffPhase.INITIAL)
        self._clear_progress(receiver_lost_to_initial, after_phase=HandoffPhase.INITIAL)
        lifted = measurements.needle_pose_w[:, 2] - self.reset_needle_z_w >= self.cfg.required_lift_delta_z_m
        retained_lift_condition = receiver_only_condition & lifted
        self._count_consecutive(self._retained_lift_counter, retained_lift_condition, receiver_phase)
        lift_complete = receiver_phase & (
            self._retained_lift_counter >= self._required_steps(self.cfg.retained_lift_dwell_s)
        )
        self.phase[lift_complete] = int(HandoffPhase.RETAINED_LIFT)
        return self.phase


def jaw_needle_contact_measurements(
    env: ManagerBasedRLEnv,
    sensor_names: tuple[str, str, str, str] = JAW_CONTACT_SENSOR_NAMES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read four needle-filtered force matrices and body-reaction axes.

    Returns projected compressive loads, world-space reaction axes pointing
    into the jaw solids, and the unmodified filtered world-force vectors.  The
    force matrix is the reaction acting on the jaw sensor body.  Each sensor
    must contain exactly one jaw body and exactly one needle filter.
    """

    if len(sensor_names) != 4:
        raise ValueError("needle pass requires exactly four jaw contact sensors")
    loads: list[torch.Tensor] = []
    normals: list[torch.Tensor] = []
    force_vectors: list[torch.Tensor] = []
    for sensor_name, local_normal in zip(sensor_names, JAW_BODY_REACTION_NORMALS_LOCAL, strict=True):
        sensor: ContactSensor = env.scene.sensors[sensor_name]
        force_matrix = sensor.data.force_matrix_w
        if force_matrix is None or force_matrix.shape != (env.num_envs, 1, 1, 3):
            actual_shape = None if force_matrix is None else tuple(force_matrix.shape)
            raise RuntimeError(
                f"contact sensor {sensor_name!r} must expose a (num_envs, 1, 1, 3) needle force matrix; "
                f"got {actual_shape}"
            )
        if sensor.data.quat_w is None or sensor.data.quat_w.shape != (env.num_envs, 1, 4):
            raise RuntimeError(f"contact sensor {sensor_name!r} must track its world pose")
        force_w = force_matrix[:, 0, 0, :]
        local_normal_tensor = torch.tensor(local_normal, dtype=force_w.dtype, device=force_w.device).repeat(
            env.num_envs, 1
        )
        body_reaction_normal_w = math_utils.quat_apply(sensor.data.quat_w[:, 0, :], local_normal_tensor)
        # F_w acts on the jaw body.  The body-reaction axis points from the
        # channel face into the jaw solid, so physical compression is exactly
        # F_n = max(0, dot(F_w, n_w)); J_n for one step is F_n * sim.dt.
        compressive_load = torch.clamp(torch.sum(force_w * body_reaction_normal_w, dim=-1), min=0.0)
        loads.append(compressive_load)
        normals.append(body_reaction_normal_w)
        force_vectors.append(force_w)
    return torch.stack(loads, dim=-1), torch.stack(normals, dim=1), torch.stack(force_vectors, dim=1)


def _asset_pose_w(asset: Articulation, body_name: str) -> torch.Tensor:
    body_ids, body_names = asset.find_bodies(body_name)
    if len(body_ids) != 1:
        raise RuntimeError(f"expected one {body_name!r} body, found {body_names}")
    return torch.cat((asset.data.body_pos_w[:, body_ids[0]], asset.data.body_quat_w[:, body_ids[0]]), dim=-1)


def get_handoff_phase_machine(env: ManagerBasedRLEnv, phase_cfg: HandoffPhaseCfg) -> HandoffPhaseMachine:
    """Return the environment-owned phase machine, constructing it once."""

    attribute_name = "_needle_pass_handoff_phase_machine"
    machine = getattr(env, attribute_name, None)
    if machine is None:
        machine = HandoffPhaseMachine(env.num_envs, env.device, env.step_dt, phase_cfg)
        setattr(env, attribute_name, machine)
    elif machine.cfg != phase_cfg:
        raise RuntimeError("needle-pass manager terms must share one HandoffPhaseCfg")
    return machine


def update_handoff_phase(
    env: ManagerBasedRLEnv,
    phase_cfg: HandoffPhaseCfg,
    needle_cfg: SceneEntityCfg = SceneEntityCfg("needle"),
    receiver_cfg: SceneEntityCfg = SceneEntityCfg("right_psm"),
    receiver_body_name: str = "psm_tool_tip_link",
) -> HandoffPhaseMachine:
    """Update the shared phase machine idempotently from post-physics buffers."""

    machine = get_handoff_phase_machine(env, phase_cfg)
    step_token = int(env.common_step_counter)
    cache_attribute = "_needle_pass_handoff_phase_sample_step_token"
    if getattr(env, cache_attribute, None) == step_token:
        return machine
    loads, normals, _ = jaw_needle_contact_measurements(env)
    needle: RigidObject = env.scene[needle_cfg.name]
    receiver: Articulation = env.scene[receiver_cfg.name]
    machine.advance(
        HandoffMeasurements(
            normal_forces_n=loads,
            reaction_normals_w=normals,
            needle_pose_w=torch.cat((needle.data.root_pos_w, needle.data.root_quat_w), dim=-1),
            needle_velocity_w=torch.cat((needle.data.root_lin_vel_w, needle.data.root_ang_vel_w), dim=-1),
            receiver_pose_w=_asset_pose_w(receiver, receiver_body_name),
        ),
        step_token=step_token,
    )
    setattr(env, cache_attribute, step_token)
    return machine


def reset_handoff_phase(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    reset_needle_z_w: torch.Tensor,
    phase_cfg: HandoffPhaseCfg,
) -> None:
    """Partially reset state and the reset-relative height reference."""

    get_handoff_phase_machine(env, phase_cfg).reset(
        env_ids,
        reset_needle_z_w,
        step_token=env.common_step_counter,
    )


def success(env: ManagerBasedRLEnv, phase_cfg: HandoffPhaseCfg) -> torch.Tensor:
    """Return true only after the measured retained-lift dwell completes."""

    machine = update_handoff_phase(env, phase_cfg)
    return machine.phase == int(HandoffPhase.RETAINED_LIFT)


def needle_dropped_or_out_of_bounds(
    env: ManagerBasedRLEnv,
    phase_cfg: HandoffPhaseCfg,
    needle_cfg: SceneEntityCfg = SceneEntityCfg("needle"),
    drop_distance_m: float = 0.12,
    horizontal_distance_m: float = 0.45,
) -> torch.Tensor:
    """Terminate a physically dropped needle separately from success."""

    if drop_distance_m <= 0.0 or horizontal_distance_m <= 0.0:
        raise ValueError("drop and horizontal bounds must be positive")
    machine = update_handoff_phase(env, phase_cfg)
    needle: RigidObject = env.scene[needle_cfg.name]
    dropped = needle.data.root_pos_w[:, 2] < machine.reset_needle_z_w - drop_distance_m
    horizontal_offset = needle.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    out_of_bounds = torch.linalg.vector_norm(horizontal_offset, dim=-1) > horizontal_distance_m
    non_finite = ~torch.isfinite(needle.data.root_state_w).all(dim=-1)
    return dropped | out_of_bounds | non_finite


__all__ = [
    "HandoffMeasurements",
    "HandoffPhase",
    "HandoffPhaseCfg",
    "HandoffPhaseMachine",
    "JAW_CONTACT_SENSOR_NAMES",
    "JAW_BODY_REACTION_NORMALS_LOCAL",
    "get_handoff_phase_machine",
    "jaw_needle_contact_measurements",
    "needle_dropped_or_out_of_bounds",
    "reset_handoff_phase",
    "success",
    "update_handoff_phase",
]
