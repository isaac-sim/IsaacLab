# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.utils.math  import rand_range
from isaaclab.utils.types import ArticulationThrustActions

if TYPE_CHECKING:
    from .actuator_cfg import ThrusterCfg


class Thruster():
    """Low-level motor/thruster dynamics with separate rise/fall time constants.

    Supports two models:
      • Force domain (integrate thrust directly), or
      • Speed domain (integrate rotor speed ω, then F = k_f * ω²) when ``cfg.use_rps=True``.

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

    def __init__(self, 
                 cfg: ThrusterCfg, 
                 thruster_names: list[str],
                 thruster_ids: slice | torch.Tensor,
                 num_envs: int, 
                 device: str,
                 init_thruster_rps: torch.Tensor):
        """Construct buffers and sample per-motor parameters.

        Args:
            num_envs: Number of vectorized envs.
            cfg: Thruster configuration.
            device: PyTorch device string.
        """
        self.cfg = cfg
        self._num_envs = num_envs
        self._device = device
        self._thruster_names = thruster_names
        self._thruster_indices = thruster_ids
        self._init_thruster_rps = init_thruster_rps

        # Range tensors, shaped (num_envs, 2, num_motors); [:,0,:]=min, [:,1,:]=max
        target_size = (self._num_envs, 2, cfg.num_motors)
        self.thrust_r = torch.tensor(cfg.thrust_range).view(1, 2, 1).expand(target_size).to(self._device)
        self.tau_inc_r = torch.tensor(cfg.tau_inc_range).view(1, 2, 1).expand(target_size).to(self._device)
        self.tau_dec_r = torch.tensor(cfg.tau_dec_range).view(1, 2, 1).expand(target_size).to(self._device)

        self.max_rate = torch.tensor(cfg.max_thrust_rate).expand(self._num_envs, cfg.num_motors).to(self._device)
        
        self.max_thrust = self.cfg.thrust_range[1]
        self.min_thrust = self.cfg.thrust_range[0]

        # State & randomized per-motor parameters
        self.tau_inc_s = rand_range(self.tau_inc_r[:, 0], self.tau_inc_r[:, 1])
        self.tau_dec_s = rand_range(self.tau_dec_r[:, 0], self.tau_dec_r[:, 1])

        if cfg.use_rps:
            self.thrust_const_r = torch.tensor(cfg.thrust_const_range).view(1, 2, 1).expand(target_size).to(device)
            self.thrust_const = rand_range(self.thrust_const_r[:, 0], self.thrust_const_r[:, 1])
        
        self.curr_thrust = torch.ones(self._num_envs, cfg.num_motors, device=self._device, dtype=torch.float32)*self.thrust_const*self._init_thruster_rps**2 
        
        # Mixing factor (discrete vs continuous form)
        if self.cfg.use_discrete_approximation:
            self.mixing_factor_function = discrete_mixing_factor
        else:
            self.mixing_factor_function = continuous_mixing_factor

        # Choose stepping kernel once (avoids per-step branching)
        if self.cfg.integration_scheme not in ["euler", "rk4"]:
            raise ValueError("integration scheme unknown")

        if cfg.use_rps:
            if self.cfg.integration_scheme == "euler":
                self._step_thrust = compute_thrust_with_rpm_time_constant
            elif self.cfg.integration_scheme == "rk4":
                self._step_thrust = compute_thrust_with_rpm_time_constant_rk4
        else:
            if self.cfg.integration_scheme == "euler":
                self._step_thrust = compute_thrust_with_force_time_constant
            elif self.cfg.integration_scheme == "rk4":
                self._step_thrust = compute_thrust_with_force_time_constant_rk4
                
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

    def compute(self, control_action: ArticulationThrustActions) -> ArticulationThrustActions:
        """Advance the thruster state one step.

        Applies saturation, chooses rise/fall tau per motor, computes mixing factor,
        and integrates with the selected kernel.

        Args:
            control_action: (num_envs, num_motors) commanded per-motor thrust [N].

        Returns:
            (num_envs, num_motors) updated thrust state [N].
        
        """
        des_thrust = control_action.joint_thrusts
        des_thrust = torch.clamp(des_thrust, self.thrust_r[:, 0], self.thrust_r[:, 1])

        thrust_decrease_mask = torch.sign(self.curr_thrust) * torch.sign(des_thrust - self.curr_thrust)
        motor_tau = torch.where(thrust_decrease_mask < 0, self.tau_dec_s, self.tau_inc_s)
        mixing = self.mixing_factor_function(self.cfg.dt, motor_tau)

        if self.cfg.use_rps:
            thrust_args = (des_thrust, self.curr_thrust, mixing, self.thrust_const, self.max_rate, self.cfg.dt)
        else:
            thrust_args = (des_thrust, self.curr_thrust, mixing, self.max_rate, self.cfg.dt)

        self.curr_thrust[:] = self._step_thrust(*thrust_args)

        self.computed_thrust = self.curr_thrust
        self.applied_thrust = torch.clamp(self.computed_thrust, self.min_thrust, self.max_thrust)

        control_action.joint_thrusts = self.applied_thrust
        
        return control_action
    
    def reset_idx(self, env_ids=None) -> None:
        """Re-sample parameters and reinitialize state.

        Args:
            env_ids: Env indices to reset. If ``None``, resets all envs.
        """
        if env_ids is None:
            env_ids = slice(None)

        self.tau_inc_s[env_ids] = rand_range(self.tau_inc_r[env_ids, 0], self.tau_inc_r[env_ids, 1])
        self.tau_dec_s[env_ids] = rand_range(self.tau_dec_r[env_ids, 0], self.tau_dec_r[env_ids, 1])
        self.curr_thrust[env_ids] = rand_range(self.thrust_r[env_ids, 0], self.thrust_r[env_ids, 1])

        if self.cfg.use_rps:
            self.thrust_const[env_ids] = rand_range(self.thrust_const_r[:, 0], self.thrust_const_r[:, 1])[env_ids]

    def reset(self) -> None:
        """Reset all envs."""
        self.reset_idx()


@torch.jit.script
def motor_model_rate(error: torch.Tensor, mixing_factor: torch.Tensor, max_rate: torch.Tensor):
    return torch.clamp(mixing_factor * (error), -max_rate, max_rate)


@torch.jit.script
def rk4_integration(error: torch.Tensor, mixing_factor: torch.Tensor, max_rate: torch.Tensor, dt: float):
    k1 = motor_model_rate(error, mixing_factor, max_rate)
    k2 = motor_model_rate(error + 0.5 * dt * k1, mixing_factor, max_rate)
    k3 = motor_model_rate(error + 0.5 * dt * k2, mixing_factor, max_rate)
    k4 = motor_model_rate(error + dt * k3, mixing_factor, max_rate)
    return (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@torch.jit.script
def discrete_mixing_factor(dt: float, time_constant: torch.Tensor):
    return 1.0 / (dt + time_constant)


@torch.jit.script
def continuous_mixing_factor(dt: float, time_constant: torch.Tensor):http://thruster.py
    return 1.0 / time_constant


@torch.jit.script
def compute_thrust_with_rpm_time_constant(
    des_thrust: torch.Tensor,
    curr_thrust: torch.Tensor,
    mixing_factor: torch.Tensor,
    thrust_const: torch.Tensor,
    max_rate: torch.Tensor,
    dt: float,
):
    current_rpm = torch.sqrt(curr_thrust / thrust_const)
    desired_rpm = torch.sqrt(des_thrust / thrust_const)
    rpm_error = desired_rpm - current_rpm
    current_rpm += motor_model_rate(rpm_error, mixing_factor, max_rate) * dt
    return thrust_const * current_rpm**2


@torch.jit.script
def compute_thrust_with_rpm_time_constant_rk4(
    des_thrust: torch.Tensor,
    curr_thrust: torch.Tensor,
    mixing_factor: torch.Tensor,
    thrust_const: torch.Tensor,
    max_rate: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    current_rpm = torch.sqrt(curr_thrust / thrust_const)
    desired_rpm = torch.sqrt(des_thrust / thrust_const)
    rpm_error = desired_rpm - current_rpm
    current_rpm += rk4_integration(rpm_error, mixing_factor, max_rate, dt)
    return thrust_const * current_rpm**2


@torch.jit.script
def compute_thrust_with_force_time_constant(
    des_thrust: torch.Tensor, curr_thrust: torch.Tensor, mixing_factor: torch.Tensor, max_rate: torch.Tensor, dt: float
) -> torch.Tensor:
    thrust_error = des_thrust - curr_thrust
    curr_thrust[:] += motor_model_rate(thrust_error, mixing_factor, max_rate) * dt
    return curr_thrust


@torch.jit.script
def compute_thrust_with_force_time_constant_rk4(
    des_thrust: torch.Tensor, curr_thrust: torch.Tensor, mixing_factor: torch.Tensor, max_rate: torch.Tensor, dt: float
) -> torch.Tensor:
    thrust_error = des_thrust - curr_thrust
    curr_thrust[:] += rk4_integration(thrust_error, mixing_factor, max_rate, dt)
    return curr_thrust
