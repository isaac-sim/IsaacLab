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

from isaaclab_tasks.utils import PresetCfg, resolve_presets

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnvCfg


def prepare_humanoid_teleop_environment(cfg: ManagerBasedRLEnvCfg) -> None:
    """Prepare humanoid task settings before simulation or environment construction.

    Dispatches to the configuration's ``stabilize_for_kitless_ovphysx`` or
    ``stabilize_for_kitless_newton_mjwarp`` for the selected standalone backend. Kit PhysX
    needs no preparation. Repeated calls on a prepared configuration are no-ops. A
    configuration constructed directly, without task resolution, uses the default of every
    remaining preset.

    Args:
        cfg: Resolved humanoid teleoperation task configuration.

    Raises:
        ValueError: If ``cfg`` was already prepared for a different physics backend.
    """
    resolve_presets(cfg)
    physics = cfg.sim.physics
    if isinstance(physics, OvPhysxCfg):
        cfg.stabilize_for_kitless_ovphysx()
    elif isinstance(physics, NewtonCfg) and isinstance(physics.solver_cfg, MJWarpSolverCfg):
        cfg.stabilize_for_kitless_newton_mjwarp()
    else:
        check_kitless_backend_preparation(cfg, None)


@configclass
class TeleopPhysicsCfg(PresetCfg):
    """Keep Kit as the default and expose the supported standalone backends."""

    isaacsim_physx = PhysxCfg()
    ovphysx = OvPhysxCfg()
    newton_mjwarp = NewtonCfg(solver_cfg=MJWarpSolverCfg())
    physx = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
    default = isaacsim_physx


def check_kitless_backend_preparation(env_cfg: ManagerBasedRLEnvCfg, backend: str | None) -> bool:
    """Check a task's kitless preparation against the selected backend.

    Preparation is recorded on the configuration, so it stays stable across repeated callers and
    configuration copies.

    Args:
        env_cfg: Task configuration that may carry ``_prepared_kitless_backend``.
        backend: Selected backend name (``"ovphysx"``, ``"newton_mjwarp"``) or ``None`` for Kit PhysX.

    Returns:
        Whether ``env_cfg`` is already prepared for ``backend``.

    Raises:
        ValueError: If ``env_cfg`` was prepared for a different backend.
    """
    prepared = getattr(env_cfg, "_prepared_kitless_backend", None)
    if prepared is not None and prepared != backend:
        raise ValueError(
            f"Task configuration was already prepared for {prepared}; resolve a fresh task configuration "
            "before selecting a different physics backend."
        )
    return prepared is not None
