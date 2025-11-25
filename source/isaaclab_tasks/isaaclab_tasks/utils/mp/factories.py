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
    def get_action(self, des_pos, des_vel, cur_pos, cur_vel):
        raise NotImplementedError


class PDController(BaseController):
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
    def __init__(self, device=None, **kwargs):
        _ = device
        _ = kwargs

    def get_action(self, des_pos, des_vel, cur_pos, cur_vel):
        return des_vel


class PosController(BaseController):
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
    ptype = phase_generator_type.lower()
    if ptype not in PHASE_TYPES:
        raise ValueError(f"Unsupported phase generator type '{phase_generator_type}'.")
    return PHASE_TYPES[ptype](device=device, **kwargs)


def get_basis_generator(basis_generator_type: str, phase_generator, device: str | torch.device | None = None, **kwargs):
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
    ctype = controller_type.lower()
    if ctype not in CONTROLLER_TYPES:
        raise ValueError(f"Unsupported controller type '{controller_type}'.")
    return CONTROLLER_TYPES[ctype](**kwargs)
