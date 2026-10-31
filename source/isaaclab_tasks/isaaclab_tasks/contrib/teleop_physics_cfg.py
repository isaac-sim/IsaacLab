# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend selection for humanoid teleoperation tasks with task-owned preparation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab.physics import PhysxAutoCfg
from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnvCfg


def prepare_humanoid_teleop_environment(cfg: ManagerBasedRLEnvCfg) -> None:
    """Prepare humanoid task settings before simulation or environment construction."""
    physics = cfg.sim.physics
    if isinstance(physics, OvPhysxCfg):
        kitless_backend_prepared(cfg, "ovphysx")
        cfg.stabilize_for_kitless_ovphysx()
    elif isinstance(physics, NewtonCfg) and isinstance(physics.solver_cfg, MJWarpSolverCfg):
        kitless_backend_prepared(cfg, "newton_mjwarp")
        cfg.stabilize_for_kitless_newton_mjwarp()
    else:
        kitless_backend_prepared(cfg, None)


@configclass
class TeleopPhysicsCfg(PresetCfg):
    """Keep Kit as the default and expose the supported standalone backends."""

    isaacsim_physx = PhysxCfg()
    ovphysx = OvPhysxCfg()
    newton_mjwarp = NewtonCfg(solver_cfg=MJWarpSolverCfg())
    physx = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
    default = isaacsim_physx


def kitless_backend_prepared(env_cfg, backend: str | None) -> bool:
    """Keep completed task preparation stable across repeated callers and config copies."""
    prepared = getattr(env_cfg, "_prepared_kitless_backend", None)
    if prepared is not None and prepared != backend:
        raise ValueError(
            f"Task configuration was already prepared for {prepared}; resolve a fresh task configuration "
            "before selecting a different physics backend."
        )
    return prepared is not None
