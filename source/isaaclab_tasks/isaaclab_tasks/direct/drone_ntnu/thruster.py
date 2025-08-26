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

from .utils import torch_rand_float

if TYPE_CHECKING:
    from .thruster_cfg import ThrusterCfg


class Thruster:

    cfg: ThrusterCfg

    def __init__(self, num_envs, cfg: ThrusterCfg, device="cpu"):
        self.cfg = cfg
        self.device = device

        self.max_thrust = self._f32(cfg.max_thrust).expand(num_envs, self.cfg.num_motors)
        self.min_thrust = self._f32(cfg.min_thrust).expand(num_envs, self.cfg.num_motors)

        self.tau_inc_min_s = self._f32(cfg.tau_inc_min).expand(num_envs, self.cfg.num_motors)
        self.tau_inc_max_s = self._f32(cfg.tau_inc_max).expand(num_envs, self.cfg.num_motors)
        self.tau_dec_min_s = self._f32(cfg.tau_dec_min).expand(num_envs, self.cfg.num_motors)
        self.tau_dec_max_s = self._f32(cfg.tau_dec_max).expand(num_envs, self.cfg.num_motors)

        self.max_rate = self._f32(cfg.max_thrust_rate).expand(num_envs, self.cfg.num_motors)


        self.curr_thrust = torch.zeros(num_envs, self.cfg.num_motors, device=self.device, dtype=torch.float32)
        self.tau_inc_s = torch_rand_float(self.tau_inc_min_s, self.tau_inc_max_s)
        self.tau_dec_s = torch_rand_float(self.tau_dec_min_s, self.tau_dec_max_s)

        if self.cfg.use_rps:
            ones = torch.ones(num_envs, self.cfg.num_motors, device=self.device, dtype=torch.float32)
            self.thrust_const_min = ones * float(self.cfg.thrust_const_min)
            self.thrust_const_max = ones * float(self.cfg.thrust_const_max)
            self.thrust_const = torch_rand_float(self.thrust_const_min, self.thrust_const_max)

        if self.cfg.use_discrete_approximation:
            self.mixing_factor_function = discrete_mixing_factor
        else:
            self.mixing_factor_function = continuous_mixing_factor
        
        
        if self.cfg.integration_scheme not in ['euler', 'rk4']:
            raise ValueError("integration scheme unknown")
    
        if cfg.use_rps:
            if self.cfg.integration_scheme == 'euler':
                self._step_thrust = compute_thrust_with_rpm_time_constant
            elif self.cfg.integration_scheme == 'rk4':
                self._step_thrust = compute_thrust_with_rpm_time_constant_rk4
        else:
            if self.cfg.integration_scheme == 'euler':
                self._step_thrust = compute_thrust_with_force_time_constant
            elif self.cfg.integration_scheme == 'rk4':
                self._step_thrust = compute_thrust_with_force_time_constant_rk4
            
    def update_motor_thrusts(self, ref_thrust):
        ref_thrust = torch.clamp(self._f32(ref_thrust), self.min_thrust, self.max_thrust)
        thrust_error_sign = torch.sign(ref_thrust - self.curr_thrust)
        motor_tau = torch.where(torch.sign(self.curr_thrust) * thrust_error_sign < 0, self.tau_dec_s, self.tau_inc_s)
        mixing = self.mixing_factor_function(self.cfg.dt, motor_tau)

        if self.cfg.use_rps:
            thrust_args = (ref_thrust, self.curr_thrust, mixing, self.thrust_const, self.max_rate, self.cfg.dt)
        else:
            thrust_args = (ref_thrust, self.curr_thrust, mixing, self.max_rate, self.cfg.dt)
        
        self.curr_thrust[:] = self._step_thrust(*thrust_args)
        return self.curr_thrust

    def reset_idx(self, env_ids = None):
        if env_ids is None:
            env_ids = slice(None)
        self.tau_inc_s[env_ids] = torch_rand_float(self.tau_inc_min_s, self.tau_inc_max_s)[env_ids]
        self.tau_dec_s[env_ids] = torch_rand_float(self.tau_dec_min_s, self.tau_dec_max_s)[env_ids]
        self.curr_thrust[env_ids] = torch_rand_float(self.min_thrust, self.max_thrust)[env_ids]
        if self.cfg.use_rps:
            self.thrust_const[env_ids] = torch_rand_float(self.thrust_const_min, self.thrust_const_max)[env_ids]

    def reset(self):
        self.reset_idx()

    def _f32(self, x):
        return torch.as_tensor(x, device=self.device, dtype=torch.float32)


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
def continuous_mixing_factor(dt: float, time_constant: torch.Tensor):
    return 1.0 / time_constant


@torch.jit.script
def compute_thrust_with_rpm_time_constant(
    ref_thrust: torch.Tensor,
    curr_thrust: torch.Tensor,
    mixing_factor: torch.Tensor,
    thrust_const: torch.Tensor,
    max_rate: torch.Tensor,
    dt: float,
):
    current_rpm = torch.sqrt(curr_thrust / thrust_const)
    desired_rpm = torch.sqrt(ref_thrust / thrust_const)
    rpm_error = desired_rpm - current_rpm
    current_rpm += motor_model_rate(rpm_error, mixing_factor, max_rate) * dt
    return thrust_const * current_rpm**2

@torch.jit.script
def compute_thrust_with_rpm_time_constant_rk4(
    ref_thrust: torch.Tensor,
    curr_thrust: torch.Tensor,
    mixing_factor: torch.Tensor,
    thrust_const: torch.Tensor,
    max_rate: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    current_rpm = torch.sqrt(curr_thrust / thrust_const)
    desired_rpm = torch.sqrt(ref_thrust / thrust_const)
    rpm_error = desired_rpm - current_rpm
    current_rpm += rk4_integration(rpm_error, mixing_factor, max_rate, dt)
    return thrust_const * current_rpm**2

@torch.jit.script
def compute_thrust_with_force_time_constant(
    ref_thrust: torch.Tensor, curr_thrust: torch.Tensor, mixing_factor: torch.Tensor, max_rate: torch.Tensor, dt: float
) -> torch.Tensor:
    thrust_error = ref_thrust - curr_thrust
    curr_thrust[:] += motor_model_rate(thrust_error, mixing_factor, max_rate) * dt
    return curr_thrust

@torch.jit.script
def compute_thrust_with_force_time_constant_rk4(
    ref_thrust: torch.Tensor, curr_thrust: torch.Tensor, mixing_factor: torch.Tensor, max_rate: torch.Tensor, dt: float
) -> torch.Tensor:
    thrust_error = ref_thrust - curr_thrust
    curr_thrust[:] += rk4_integration(thrust_error, mixing_factor, max_rate, dt)
    return curr_thrust
