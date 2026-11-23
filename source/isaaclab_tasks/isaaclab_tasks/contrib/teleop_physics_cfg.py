# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend selection for humanoid teleoperation tasks with task-owned preparation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_isaacsim.physics import PhysxCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_ov.physics import OvPhysxCfg

from isaaclab.physics import PhysxAutoCfg
from isaaclab.physics.physics_manager_cfg import _resolve_physx_auto_cfg
from isaaclab.utils import configclass, string_to_callable
from isaaclab.utils.version import has_kit

from isaaclab_tasks.utils import PresetCfg, resolve_presets

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnvCfg


def prepare_humanoid_teleop_environment(cfg: ManagerBasedRLEnvCfg) -> None:
    """Prepare humanoid task settings before simulation or environment construction.

    Dispatches to the preparation function that the configuration's ``backend_preparation``
    mapping names for the selected standalone backend (``"ovphysx"`` or ``"newton_mjwarp"``).
    Kit PhysX needs no preparation. Repeated calls on a prepared configuration are no-ops. A
    configuration constructed directly, without task resolution, uses the default of every
    remaining preset. The ``physx`` preset is resolved here as the simulation would resolve it:
    to Kit PhysX when Kit is available, otherwise to OvPhysX.

    Args:
        cfg: Resolved humanoid teleoperation task configuration.

    Raises:
        ValueError: If the selected physics backend is not one of the supported backends, or ``cfg``
            was already prepared for a different physics backend.
    """
    resolve_presets(cfg)
    if isinstance(cfg.sim.physics, PhysxAutoCfg):
        # A direct construction resolves this only inside SimulationContext, after preparation.
        cfg.sim.physics = _resolve_physx_auto_cfg(cfg.sim.physics, use_isaac_sim=has_kit())
    physics = cfg.sim.physics
    if isinstance(physics, OvPhysxCfg):
        backend = "ovphysx"
    elif isinstance(physics, NewtonCfg) and isinstance(physics.solver_cfg, MJWarpSolverCfg):
        backend = "newton_mjwarp"
    elif physics is None or isinstance(physics, PhysxCfg):
        # SimulationContext defaults a missing selection to Kit PhysX.
        check_backend_preparation(cfg, None)
        return
    else:
        raise ValueError(
            f"Humanoid teleoperation supports Kit PhysX, OvPhysX and Newton with MJWarp; got {type(physics).__name__}."
        )
    string_to_callable(cfg.backend_preparation[backend])(cfg)


@configclass
class TeleopPhysicsCfg(PresetCfg):
    """Keep Kit as the default and expose the supported standalone backends."""

    isaacsim_physx = PhysxCfg()
    ovphysx = OvPhysxCfg()
    newton_mjwarp = NewtonCfg(solver_cfg=MJWarpSolverCfg())
    physx = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
    default = isaacsim_physx


def check_backend_preparation(env_cfg: ManagerBasedRLEnvCfg, backend: str | None) -> bool:
    """Check a task's backend preparation against the selected backend.

    Preparation is recorded on the configuration, so it stays stable across repeated callers and
    configuration copies.

    Args:
        env_cfg: Task configuration that declares ``_prepared_physics_backend``.
        backend: Selected backend name (``"ovphysx"``, ``"newton_mjwarp"``) or ``None`` for Kit PhysX.

    Returns:
        Whether ``env_cfg`` is already prepared for ``backend``.

    Raises:
        ValueError: If ``env_cfg`` was prepared for a different backend.
    """
    prepared = env_cfg._prepared_physics_backend
    if prepared is not None and prepared != backend:
        raise ValueError(
            f"Task configuration was already prepared for {prepared}; resolve a fresh task configuration "
            "before selecting a different physics backend."
        )
    return prepared is not None
