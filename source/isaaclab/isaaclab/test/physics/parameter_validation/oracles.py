# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure mechanics oracles used by simulation parameter-validation tests."""

import math
from dataclasses import dataclass

import torch

PROFILE_DOF_DT = 1.0 / 120.0
PROFILE_FREE_DT = 1.0 / 120.0
PROFILE_CONTACT_DT = 1.0 / 120.0


@dataclass(frozen=True)
class PhysicalCase:
    """Traceability information for one physical-behavior assertion."""

    parameter_id: str
    backend: str
    authoring_path: str
    profile: str
    dt: float
    substeps: int
    api: str
    rtol: float
    atol: float

    def message(self, measured: object, expected: object) -> str:
        """Format diagnostic context for a failed assertion."""
        return (
            f"{self.parameter_id}: backend={self.backend}, authoring={self.authoring_path}, "
            f"profile={self.profile}, dt={self.dt}, substeps={self.substeps}, api={self.api}, "
            f"measured={measured}, expected={expected}, "
            f"rtol={self.rtol}, atol={self.atol}"
        )


def predict_critical_incline_angle(friction: float) -> float:
    """Predict the Coulomb static-friction threshold angle [rad]."""
    return math.atan(friction)


def static_friction_angle_deadband(critical_angle: float) -> float:
    """Return the minimum static-friction threshold dead band [rad]."""
    return max(math.radians(2.0), 0.05 * critical_angle)


def predict_friction_stopping_distance(initial_speed: float, friction: float, gravity: float) -> float:
    """Predict Coulomb-friction stopping distance [m] on a level surface."""
    if friction <= 0.0:
        raise ValueError("Friction must be positive to predict a finite stopping distance.")
    if gravity == 0.0:
        raise ValueError("Gravity magnitude must be non-zero to predict a stopping distance.")
    return initial_speed * initial_speed / (2.0 * friction * abs(gravity))


def dynamic_friction_distance_atol(initial_speed: float, dt: float = PROFILE_CONTACT_DT) -> float:
    """Return the one-step stopping-distance uncertainty [m]."""
    return abs(initial_speed) * dt


def predict_rebound_height(drop_height: float, restitution: float) -> float:
    """Predict the first rebound height [m] for a Newton restitution coefficient."""
    return restitution * restitution * drop_height


def contact_separation_atol(approach_speed: float, dt: float = PROFILE_CONTACT_DT) -> float:
    """Return the first-contact separation tolerance [m]."""
    return max(2.0e-3, 2.0 * abs(approach_speed) * dt)


def predict_implicit_joint_step(
    *,
    stiffness: float,
    drive_damping: float,
    armature: float,
    position_target: float,
    body_inertia: float,
    effort: float = 0.0,
    velocity_target: float = 0.0,
    position: float = 0.0,
    velocity: float = 0.0,
    passive_damping: float = 0.0,
    dt: float = PROFILE_DOF_DT,
) -> tuple[float, float]:
    """Predict one implicit semi-Euler step for a fixed-base single-DOF joint."""
    effective_inertia = body_inertia + armature + dt * (drive_damping + passive_damping) + dt * dt * stiffness
    drive_effort = effort + stiffness * (position_target - position) + drive_damping * velocity_target
    velocity_next = ((body_inertia + armature) * velocity + dt * drive_effort) / effective_inertia
    return velocity_next, position + dt * velocity_next


def predict_implicitfast_joint_step(
    *,
    stiffness: float,
    drive_damping: float,
    armature: float,
    position_target: float,
    body_inertia: float,
    effort: float = 0.0,
    velocity_target: float = 0.0,
    position: float = 0.0,
    velocity: float = 0.0,
    passive_damping: float = 0.0,
    dt: float = PROFILE_DOF_DT,
) -> tuple[float, float]:
    """Predict one MJWarp implicitFast step for a fixed-base single-DOF joint."""
    effective_inertia = body_inertia + armature + dt * (drive_damping + passive_damping)
    drive_effort = effort + stiffness * (position_target - position) + drive_damping * velocity_target
    velocity_next = ((body_inertia + armature) * velocity + dt * drive_effort) / effective_inertia
    return velocity_next, position + dt * velocity_next


def predict_semi_implicit_motion(
    position: torch.Tensor,
    velocity: torch.Tensor,
    acceleration: torch.Tensor,
    *,
    dt: float,
    steps: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict pinned semi-implicit Euler translation after a number of steps."""
    position_next = position.clone()
    velocity_next = velocity.clone()
    for _ in range(steps):
        velocity_next = velocity_next + acceleration * dt
        position_next = position_next + velocity_next * dt
    return position_next, velocity_next


def predict_linear_wrench_step(
    velocity: torch.Tensor, force: torch.Tensor, mass: float, *, dt: float = PROFILE_FREE_DT
) -> torch.Tensor:
    """Predict center-of-mass linear velocity after one constant-force step."""
    return velocity + force * (dt / mass)


def predict_angular_wrench_step(
    angular_velocity: torch.Tensor,
    torque: torch.Tensor,
    inertia_world: torch.Tensor,
    *,
    dt: float = PROFILE_FREE_DT,
) -> torch.Tensor:
    """Predict angular velocity after one constant-torque step."""
    return angular_velocity + torch.linalg.solve(inertia_world, torque) * dt


def assert_physical_close(
    measured: torch.Tensor | float,
    expected: torch.Tensor | float,
    case: PhysicalCase,
) -> None:
    """Assert physical values with complete parameter-validation diagnostics."""
    measured_tensor = torch.as_tensor(measured)
    expected_tensor = torch.as_tensor(expected, device=measured_tensor.device, dtype=measured_tensor.dtype)
    absolute_error = torch.max(torch.abs(measured_tensor - expected_tensor)).item()
    relative_error = torch.max(
        torch.abs(measured_tensor - expected_tensor)
        / torch.clamp(torch.abs(expected_tensor), min=torch.finfo(measured_tensor.dtype).eps)
    ).item()
    torch.testing.assert_close(
        measured_tensor,
        expected_tensor,
        rtol=case.rtol,
        atol=case.atol,
        msg=lambda msg: (
            f"{case.message(measured, expected)}, max_absolute_error={absolute_error}, "
            f"max_relative_error={relative_error}\n{msg}"
        ),
    )
