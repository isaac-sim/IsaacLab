# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from mp_pytorch.basis_gn import (
    NormalizedRBFBasisGenerator,
    ProDMPBasisGenerator,
    ZeroPaddingNormalizedRBFBasisGenerator,
)
from mp_pytorch.mp import DMP, ProDMP, ProMP
from mp_pytorch.phase_gn import ExpDecayPhaseGenerator, LinearPhaseGenerator

# Minimal controller implementations for tracking desired trajectories.


class BaseController:
    """Interface for controllers that track desired position/velocity trajectories."""

    def get_action(self, des_pos, des_vel, cur_pos, cur_vel):
        raise NotImplementedError


class PDController(BaseController):
    """PD controller computing torques/commands from desired and current state."""

    def __init__(
        self,
        p_gains: float | torch.Tensor = 1.0,
        d_gains: float | torch.Tensor = 0.5,
        device=None,
        **_: float | torch.Tensor,
    ):
        self.p_gains = torch.as_tensor(p_gains, device=device)
        self.d_gains = torch.as_tensor(d_gains, device=device)

    def get_action(self, des_pos, des_vel, cur_pos, cur_vel):
        return self.p_gains * (des_pos - cur_pos) + self.d_gains * (des_vel - cur_vel)


class VelController(BaseController):
    """Velocity controller that forwards desired velocities as the action."""

    def __init__(self, device=None, **kwargs):
        _ = device
        _ = kwargs

    def get_action(self, des_pos, des_vel, cur_pos, cur_vel):
        return des_vel


class PosController(BaseController):
    """Position controller that forwards desired positions as the action."""

    def get_action(self, des_pos, des_vel, cur_pos, cur_vel):
        return des_pos


CONTROLLER_TYPES = {
    "motor": PDController,
    "velocity": VelController,
    "position": PosController,
}

PHASE_TYPES = {
    "linear": LinearPhaseGenerator,
    "exp": ExpDecayPhaseGenerator,
}

BASIS_TYPES = ["rbf", "zero_rbf", "prodmp"]
TRAJ_TYPES = ["promp", "dmp", "prodmp"]

MP_DEFAULTS = {
    "ProMP": {
        "trajectory_generator_kwargs": {"trajectory_generator_type": "promp"},
        "phase_generator_kwargs": {"phase_generator_type": "linear"},
        "controller_kwargs": {"controller_type": "motor", "p_gains": 1.0, "d_gains": 0.1},
        "basis_generator_kwargs": {"basis_generator_type": "zero_rbf", "num_basis": 5, "num_basis_zero_start": 1},
        "black_box_kwargs": {},
    },
    "DMP": {
        "trajectory_generator_kwargs": {"trajectory_generator_type": "dmp"},
        "phase_generator_kwargs": {"phase_generator_type": "exp"},
        "controller_kwargs": {"controller_type": "motor", "p_gains": 1.0, "d_gains": 0.1},
        "basis_generator_kwargs": {"basis_generator_type": "rbf", "num_basis": 5},
        "black_box_kwargs": {},
    },
    "ProDMP": {
        "trajectory_generator_kwargs": {
            "trajectory_generator_type": "prodmp",
            "duration": 2.0,
            "weights_scale": 1.0,
            "disable_goal": False,
        },
        "phase_generator_kwargs": {"phase_generator_type": "exp", "tau": 1.5},
        "controller_kwargs": {"controller_type": "motor", "p_gains": 1.0, "d_gains": 0.1},
        "basis_generator_kwargs": {"basis_generator_type": "prodmp", "alpha": 10, "num_basis": 5},
        "black_box_kwargs": {},
    },
}


def get_phase_generator(phase_generator_type: str, device: str | torch.device | None = None, **kwargs):
    """Instantiate a phase generator by type.

    Args:
        phase_generator_type (str): `"linear"` or `"exp"` (case-insensitive).
        device (str | torch.device | None): Device for internal tensors.
        **kwargs: Extra parameters forwarded to the phase generator ctor (e.g., `tau`).

    Returns:
        ExpDecayPhaseGenerator | LinearPhaseGenerator: Phase generator on `device`.

    Raises:
        ValueError: If the type is not supported.

    Notes:
        Phase generators output a 1D phase variable per environment; downstream basis
        generators assume matching batch/device.
    """
    ptype = phase_generator_type.lower()
    if ptype not in PHASE_TYPES:
        raise ValueError(f"Unsupported phase generator type '{phase_generator_type}'.")
    return PHASE_TYPES[ptype](device=device, **kwargs)


def get_basis_generator(basis_generator_type: str, phase_generator, device: str | torch.device | None = None, **kwargs):
    """Instantiate a basis generator compatible with the provided phase generator.

    Args:
        basis_generator_type (str): One of `"rbf"`, `"zero_rbf"`, `"prodmp"` (case-insensitive).
        phase_generator: Phase generator instance; must be `ExpDecayPhaseGenerator` for ProDMP.
        device (str | torch.device | None): Device for internal tensors and buffers.
        **kwargs: Extra parameters forwarded to the basis generator ctor (e.g., `num_basis`).

    Returns:
        NormalizedRBFBasisGenerator | ZeroPaddingNormalizedRBFBasisGenerator | ProDMPBasisGenerator.

    Raises:
        ValueError: On unknown type or when ProDMP is requested without an exponential phase.

    Notes:
        Basis generators produce features shaped `(batch, basis_dim)` aligned with the phase
        generator output and are kept on the same device to avoid cross-device copies.
    """
    btype = basis_generator_type.lower()
    if btype == "rbf":
        return NormalizedRBFBasisGenerator(phase_generator, device=device, **kwargs)
    if btype == "zero_rbf":
        return ZeroPaddingNormalizedRBFBasisGenerator(phase_generator, device=device, **kwargs)
    if btype == "prodmp":
        if not isinstance(phase_generator, ExpDecayPhaseGenerator):
            raise ValueError("ProDMP basis requires an ExpDecayPhaseGenerator.")
        return ProDMPBasisGenerator(phase_generator, device=device, **kwargs)
    raise ValueError(f"Unsupported basis generator type '{basis_generator_type}'.")


def get_trajectory_generator(
    trajectory_generator_type: str,
    action_dim: int,
    basis_generator,
    device: str | torch.device | None = None,
    **kwargs,
):
    """Instantiate a trajectory generator that maps MP parameters to state trajectories.

    Args:
        trajectory_generator_type (str): `"promp"`, `"dmp"`, or `"prodmp"` (case-insensitive).
        action_dim (int): Degrees of freedom of the controlled system.
        basis_generator: Basis generator instance; must be `ProDMPBasisGenerator` for ProDMP.
        device (str | torch.device | None): Device for parameters and outputs.
        **kwargs: Extra arguments forwarded to the generator ctor (e.g., `duration`).

    Returns:
        ProMP | DMP | ProDMP: Trajectory generator on the requested device.

    Raises:
        ValueError: On unknown type or mismatched basis/trajectory pairing.

    Notes:
        Generators expect MP parameters shaped `(batch, param_dim)` and emit trajectories
        shaped `(batch, horizon, action_dim)` on `device`.
    """
    ttype = trajectory_generator_type.lower()
    if ttype == "promp":
        return ProMP(basis_generator, action_dim, device=device, **kwargs)
    if ttype == "dmp":
        return DMP(basis_generator, action_dim, device=device, **kwargs)
    if ttype == "prodmp":
        if not isinstance(basis_generator, ProDMPBasisGenerator):
            raise ValueError("ProDMP trajectory generator requires a ProDMPBasisGenerator.")
        return ProDMP(basis_generator, action_dim, device=device, **kwargs)
    raise ValueError(f"Unsupported trajectory generator type '{trajectory_generator_type}'.")


def get_controller(controller_type: str, **kwargs) -> BaseController:
    """Instantiate a simple tracking controller.

    Args:
        controller_type (str): `"motor"` (PD), `"velocity"`, or `"position"`.
        **kwargs: Controller-specific parameters such as `p_gains`, `d_gains`, or `device`.

    Returns:
        BaseController: Concrete controller instance.

    Raises:
        ValueError: If the type is unsupported.

    Notes:
        Controllers accept batched desired/current positions and velocities and emit
        actions shaped `(batch, dof)` that `BlackBoxWrapper` will clamp to env bounds.
    """
    ctype = controller_type.lower()
    if ctype not in CONTROLLER_TYPES:
        raise ValueError(f"Unsupported controller type '{controller_type}'.")
    return CONTROLLER_TYPES[ctype](**kwargs)
