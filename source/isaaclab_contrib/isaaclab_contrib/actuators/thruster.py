# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils

from isaaclab_contrib.utils.types import MultiRotorActions

if TYPE_CHECKING:
    from .thruster_cfg import ThrusterCfg


class Thruster:
    """Low-level motor/thruster dynamics with separate rise/fall time constants.

    Integration scheme is Euler or RK4. All internal buffers are shaped (num_envs, num_motors).
    Units: thrust [N], rates [N/s], time [s].
    """

    computed_thrust: torch.Tensor
    """The computed thrust for the actuator group. Shape is (num_envs, num_thrusters)."""

    applied_thrust: torch.Tensor
    """The applied thrust for the actuator group. Shape is (num_envs, num_thrusters).

    This is the thrust obtained after clipping the :attr:`computed_thrust` based on the
    actuator characteristics.
    """

    cfg: ThrusterCfg

    def __init__(
        self,
        cfg: ThrusterCfg,
        thruster_names: list[str],
        thruster_ids: slice | torch.Tensor,
        num_envs: int,
        device: str,
        init_thruster_rps: torch.Tensor,
    ):
        """Construct buffers and sample per-motor parameters.

        Args:
            cfg: Thruster configuration.
            thruster_names: List of thruster names belonging to this group.
            thruster_ids: Slice or tensor of indices into the articulation thruster array.
            num_envs: Number of parallel/vectorized environments.
            device: PyTorch device string or device identifier.
            init_thruster_rps: Initial per-thruster rotations-per-second tensor used when
                the configuration uses RPM-based thrust modelling.
        """
        self.cfg = cfg
        self._num_envs = num_envs
        self._device = device
        self._thruster_names = thruster_names
        self._thruster_indices = thruster_ids
        self._init_thruster_rps = init_thruster_rps

        # Range tensors, shaped (num_envs, 2, num_motors); [:,0,:]=min, [:,1,:]=max
        self.num_motors = len(thruster_names)
        self.thrust_r = torch.tensor(cfg.thrust_range).to(self._device)
        self.tau_inc_r = torch.tensor(cfg.tau_inc_range).to(self._device)
        self.tau_dec_r = torch.tensor(cfg.tau_dec_range).to(self._device)

        self.max_rate = torch.tensor(cfg.max_thrust_rate).expand(self._num_envs, self.num_motors).to(self._device)

        self.max_thrust = self.cfg.thrust_range[1]
        self.min_thrust = self.cfg.thrust_range[0]

        # State & randomized per-motor parameters
        self.tau_inc_s = math_utils.sample_uniform(*self.tau_inc_r, (self._num_envs, self.num_motors), self._device)
        self.tau_dec_s = math_utils.sample_uniform(*self.tau_dec_r, (self._num_envs, self.num_motors), self._device)
        self.thrust_const_r = torch.tensor(cfg.thrust_const_range, device=self._device, dtype=torch.float32)
        self.thrust_const = math_utils.sample_uniform(
            *self.thrust_const_r, (self._num_envs, self.num_motors), self._device
        ).clamp(min=1e-6)

        self.curr_thrust = self.thrust_const * (self._init_thruster_rps.to(self._device).float() ** 2)

        # Mixing factor (discrete vs continuous form)
        if self.cfg.use_discrete_approximation:
            self.mixing_factor_function = self.discrete_mixing_factor
        else:
            self.mixing_factor_function = self.continuous_mixing_factor

        # Choose stepping kernel once (avoids per-step branching)
        if self.cfg.integration_scheme == "euler":
            self._step_thrust = self.compute_thrust_with_rpm_time_constant
        elif self.cfg.integration_scheme == "rk4":
            self._step_thrust = self.compute_thrust_with_rpm_time_constant_rk4
        else:
            raise ValueError("integration scheme unknown")

    @property
    def num_thrusters(self) -> int:
        """Number of actuators in the group."""
        return len(self._thruster_names)

    @property
    def thruster_names(self) -> list[str]:
        """Articulation's thruster names that are part of the group."""
        return self._thruster_names

    @property
    def thruster_indices(self) -> slice | torch.Tensor:
        """Articulation's thruster indices that are part of the group.

        Note:
            If :obj:`slice(None)` is returned, then the group contains all the thrusters in the articulation.
            We do this to avoid unnecessary indexing of the thrusters for performance reasons.
        """
        return self._thruster_indices

    def compute(self, control_action: MultiRotorActions) -> MultiRotorActions:
        """Advance the thruster state one step.

        Applies saturation, chooses rise/fall tau per motor, computes mixing factor,
        and integrates with the selected kernel.

        Args:
            control_action: (num_envs, num_thrusters) commanded per-thruster thrust [N].

        Returns:
            (num_envs, num_thrusters) updated thrust state [N].

        """
        des_thrust = control_action.thrusts
        des_thrust = torch.clamp(des_thrust, *self.thrust_r)

        thrust_decrease_mask = torch.sign(self.curr_thrust) * torch.sign(des_thrust - self.curr_thrust)
        motor_tau = torch.where(thrust_decrease_mask < 0, self.tau_dec_s, self.tau_inc_s)
        mixing = self.mixing_factor_function(motor_tau)

        self.curr_thrust[:] = self._step_thrust(des_thrust, self.curr_thrust, mixing)

        self.computed_thrust = self.curr_thrust
        self.applied_thrust = torch.clamp(self.computed_thrust, self.min_thrust, self.max_thrust)

        control_action.thrusts = self.applied_thrust

        return control_action

    def reset_idx(self, env_ids=None) -> None:
        """Re-sample parameters and reinitialize state.

        Args:
            env_ids: Env indices to reset. If ``None``, resets all envs.
        """
        if env_ids is None:
            env_ids = slice(None)

        if isinstance(env_ids, slice):
            num_resets = self._num_envs
        else:
            num_resets = len(env_ids)

        self.tau_inc_s[env_ids] = math_utils.sample_uniform(
            *self.tau_inc_r,
            (num_resets, self.num_motors),
            self._device,
        )
        self.tau_dec_s[env_ids] = math_utils.sample_uniform(
            *self.tau_dec_r,
            (num_resets, self.num_motors),
            self._device,
        )
        self.thrust_const[env_ids] = math_utils.sample_uniform(
            *self.thrust_const_r,
            (num_resets, self.num_motors),
            self._device,
        )
        self.curr_thrust[env_ids] = self.thrust_const[env_ids] * self._init_thruster_rps[env_ids] ** 2

    def reset(self, env_ids: Sequence[int]) -> None:
        """Reset all envs."""
        self.reset_idx(env_ids)

    def motor_model_rate(self, error: torch.Tensor, mixing_factor: torch.Tensor):
        return torch.clamp(mixing_factor * (error), -self.max_rate, self.max_rate)

    def rk4_integration(self, error: torch.Tensor, mixing_factor: torch.Tensor):
        k1 = self.motor_model_rate(error, mixing_factor)
        k2 = self.motor_model_rate(error + 0.5 * self.cfg.dt * k1, mixing_factor)
        k3 = self.motor_model_rate(error + 0.5 * self.cfg.dt * k2, mixing_factor)
        k4 = self.motor_model_rate(error + self.cfg.dt * k3, mixing_factor)
        return (self.cfg.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def discrete_mixing_factor(self, time_constant: torch.Tensor):
        return 1.0 / (self.cfg.dt + time_constant)

    def continuous_mixing_factor(self, time_constant: torch.Tensor):
        return 1.0 / time_constant

    def compute_thrust_with_rpm_time_constant(
        self,
        des_thrust: torch.Tensor,
        curr_thrust: torch.Tensor,
        mixing_factor: torch.Tensor,
    ):
        # Avoid negative or NaN values inside sqrt by clamping the ratio to >= 0.
        current_ratio = torch.clamp(curr_thrust / self.thrust_const, min=0.0)
        desired_ratio = torch.clamp(des_thrust / self.thrust_const, min=0.0)
        current_rpm = torch.sqrt(current_ratio)
        desired_rpm = torch.sqrt(desired_ratio)
        rpm_error = desired_rpm - current_rpm
        current_rpm += self.motor_model_rate(rpm_error, mixing_factor) * self.cfg.dt
        return self.thrust_const * current_rpm**2

    def compute_thrust_with_rpm_time_constant_rk4(
        self,
        des_thrust: torch.Tensor,
        curr_thrust: torch.Tensor,
        mixing_factor: torch.Tensor,
    ) -> torch.Tensor:
        current_ratio = torch.clamp(curr_thrust / self.thrust_const, min=0.0)
        desired_ratio = torch.clamp(des_thrust / self.thrust_const, min=0.0)
        current_rpm = torch.sqrt(current_ratio)
        desired_rpm = torch.sqrt(desired_ratio)
        rpm_error = desired_rpm - current_rpm
        current_rpm += self.rk4_integration(rpm_error, mixing_factor)
        return self.thrust_const * current_rpm**2
