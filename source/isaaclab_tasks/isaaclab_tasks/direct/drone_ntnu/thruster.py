from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from .utils import torch_rand_float_tensor

if TYPE_CHECKING:
    from .thruster_cfg import ThrusterCfg

class Thruster:
    
    cfg: ThrusterCfg
    
    def __init__(self, num_envs, cfg: ThrusterCfg, device="cuda:0"):
        self.num_envs = num_envs
        self.cfg = cfg
        self.device = device
        self.dt = cfg.dt
        self.num_motors_per_robot = self.cfg.num_motors
        self.integration_scheme = cfg.integration_scheme

        self.max_thrust = torch.tensor(self.cfg.max_thrust, device=self.device, dtype=torch.float32).expand(self.num_envs, self.num_motors_per_robot)
        self.min_thrust = torch.tensor(self.cfg.min_thrust, device=self.device, dtype=torch.float32).expand(self.num_envs, self.num_motors_per_robot)
        self.motor_time_constant_increasing_min = torch.tensor(self.cfg.motor_time_constant_increasing_min, device=self.device).expand(self.num_envs, self.num_motors_per_robot)
        self.motor_time_constant_increasing_max = torch.tensor(self.cfg.motor_time_constant_increasing_max, device=self.device).expand(self.num_envs, self.num_motors_per_robot)
        self.motor_time_constant_decreasing_min = torch.tensor(self.cfg.motor_time_constant_decreasing_min, device=self.device).expand(self.num_envs, self.num_motors_per_robot)
        self.motor_time_constant_decreasing_max = torch.tensor(self.cfg.motor_time_constant_decreasing_max, device=self.device).expand(self.num_envs, self.num_motors_per_robot)
        self.max_rate = torch.tensor(self.cfg.max_thrust_rate, device=self.device).expand(self.num_envs, self.num_motors_per_robot)
        self.init_tensors()

    def init_tensors(self):
        self.current_motor_thrust = torch_rand_float_tensor(
            torch.tensor(self.min_thrust, device=self.device, dtype=torch.float32).expand(self.num_envs, self.num_motors_per_robot),
            torch.tensor(self.max_thrust, device=self.device, dtype=torch.float32).expand(self.num_envs, self.num_motors_per_robot),
        )
        self.motor_time_constants_increasing = torch_rand_float_tensor(self.motor_time_constant_increasing_min, self.motor_time_constant_increasing_max)
        self.motor_time_constants_decreasing = torch_rand_float_tensor(self.motor_time_constant_decreasing_min, self.motor_time_constant_decreasing_max)
        self.motor_rate = torch.zeros((self.num_envs, self.num_motors_per_robot), device=self.device)
        if self.cfg.use_rps:
            self.motor_thrust_constant_min = torch.ones(self.num_envs, self.num_motors_per_robot, device=self.device, requires_grad=False) * self.cfg.motor_thrust_constant_min
            self.motor_thrust_constant_max = torch.ones(self.num_envs, self.num_motors_per_robot, device=self.device, requires_grad=False) * self.cfg.motor_thrust_constant_max
            self.motor_thrust_constant = torch_rand_float_tensor(self.motor_thrust_constant_min, self.motor_thrust_constant_max)

        if self.cfg.use_discrete_approximation:
            self.mixing_factor_function = discrete_mixing_factor
        else:
            self.mixing_factor_function = continuous_mixing_factor

    def update_motor_thrusts(self, ref_thrust):
        # clamp ref thrust so that it is within the min and max thrust
        ref_thrust = torch.clamp(ref_thrust, self.min_thrust, self.max_thrust)
        thrust_error = ref_thrust - self.current_motor_thrust
        motor_time_constants = torch.where(
            torch.sign(self.current_motor_thrust) * torch.sign(thrust_error) < 0, self.motor_time_constants_decreasing, self.motor_time_constants_increasing,
        )
        mixing_factor = self.mixing_factor_function(self.dt, motor_time_constants)
        if self.cfg.use_rps:
            if self.integration_scheme == "euler":
                self.current_motor_thrust[:] = compute_thrust_with_rpm_time_constant(ref_thrust, self.current_motor_thrust, mixing_factor, self.motor_thrust_constant, self.max_rate, self.dt)
            elif self.integration_scheme == "rk4":
                self.current_motor_thrust[:] = compute_thrust_with_rpm_time_constant_rk4(ref_thrust, self.current_motor_thrust, mixing_factor, self.motor_thrust_constant, self.max_rate, self.dt)
            else:
                raise ValueError("integration scheme unknown")
        else:
            if self.integration_scheme == "euler":
                self.current_motor_thrust[:] = compute_thrust_with_force_time_constant(ref_thrust, self.current_motor_thrust, mixing_factor, self.max_rate, self.dt)
            elif self.integration_scheme == "rk4":
                self.current_motor_thrust[:] = compute_thrust_with_force_time_constant_rk4(ref_thrust, self.current_motor_thrust, mixing_factor, self.max_rate, self.dt)
            else:
                raise ValueError("integration scheme unknown")
        return self.current_motor_thrust

    def reset_idx(self, env_ids):
        self.motor_time_constants_increasing[env_ids] = torch_rand_float_tensor(self.motor_time_constant_increasing_min, self.motor_time_constant_increasing_max)[env_ids]
        self.motor_time_constants_decreasing[env_ids] = torch_rand_float_tensor(self.motor_time_constant_decreasing_min, self.motor_time_constant_decreasing_max)[env_ids]
        self.current_motor_thrust[env_ids] = torch_rand_float_tensor(self.min_thrust, self.max_thrust)[env_ids]
        if self.cfg.use_rps:
            self.motor_thrust_constant[env_ids] = torch_rand_float_tensor(self.motor_thrust_constant_min, self.motor_thrust_constant_max)[env_ids]

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))


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
    ref_thrust: torch.Tensor, current_thrust: torch.Tensor, mixing_factor: torch.Tensor, thrust_constant: torch.Tensor, max_rate: torch.Tensor, dt: float
):
    current_rpm = torch.sqrt(current_thrust / thrust_constant)
    desired_rpm = torch.sqrt(ref_thrust / thrust_constant)
    rpm_error = desired_rpm - current_rpm
    current_rpm += motor_model_rate(rpm_error, mixing_factor, max_rate) * dt
    return thrust_constant * current_rpm**2


@torch.jit.script
def compute_thrust_with_force_time_constant(
    ref_thrust: torch.Tensor, current_thrust: torch.Tensor, mixing_factor: torch.Tensor, max_rate: torch.Tensor, dt: float
):
    thrust_error = ref_thrust - current_thrust
    current_thrust[:] += motor_model_rate(thrust_error, mixing_factor, max_rate) * dt
    return current_thrust

@torch.jit.script
def compute_thrust_with_rpm_time_constant_rk4(
    ref_thrust: torch.Tensor, current_thrust: torch.Tensor, mixing_factor: torch.Tensor, thrust_constant: torch.Tensor, max_rate: torch.Tensor, dt: float
):
    current_rpm = torch.sqrt(current_thrust / thrust_constant)
    desired_rpm = torch.sqrt(ref_thrust / thrust_constant)
    rpm_error = desired_rpm - current_rpm
    current_rpm += rk4_integration(rpm_error, mixing_factor, max_rate, dt)
    return thrust_constant * current_rpm**2


@torch.jit.script
def compute_thrust_with_force_time_constant_rk4(
    ref_thrust: torch.Tensor, current_thrust: torch.Tensor, mixing_factor: torch.Tensor, max_rate: torch.Tensor, dt: float
):
    thrust_error = ref_thrust - current_thrust
    current_thrust[:] += rk4_integration(thrust_error, mixing_factor, max_rate, dt)
    return current_thrust
