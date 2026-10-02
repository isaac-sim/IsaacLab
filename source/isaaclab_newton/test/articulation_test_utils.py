# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Locally authored scenes shared by the kitless Newton asset and controller tests.

The small USD scenes live in ``test/assets/data`` and need neither Kit nor Nucleus.
Tests load them directly, so their bodies, joints, mass properties, and drives are readable without running
a Python scene generator.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import warp as wp
from isaaclab_newton.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.physics import NewtonManager as SimulationManager
from newton import ModelFlags

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.cloner import CloneCfg, clone_plan_from_env_0, replicate
from isaaclab.sim import SimulationCfg, SimulationContext

NUM_ENVS = 2
"""Environments per scene: enough to prove a partial write leaves the other environment untouched."""

WRIST_USD_STIFFNESS = (0.5, 0.4, 0.3)
"""Angular drive stiffness authored on the fixed chain's wrist joints [N·m/deg]; the damping is a tenth of it."""

Vec3 = tuple[float, float, float]

_DATA_DIR = Path(__file__).resolve().parent / "assets" / "data"
"""Directory holding the checked-in USD fixtures."""

_ENV_SPACING = 40.0
"""Distance between environment origins along x [m]."""


def local_usd(name: str, **kwargs: Any) -> sim_utils.UsdFileCfg:
    """Return a spawner for one of the local USD fixtures.

    Args:
        name: File name in ``test/assets/data``.
        **kwargs: Additional :class:`~isaaclab.sim.UsdFileCfg` fields.

    Returns:
        The spawner configuration.
    """
    return sim_utils.UsdFileCfg(usd_path=str(_DATA_DIR / name), **kwargs)


def newton_sim_cfg(
    device: str = "cpu",
    *,
    dt: float = 1.0 / 120.0,
    gravity: Vec3 = (0.0, 0.0, 0.0),
    use_newton_actuators: bool = True,
    newton_contacts: bool = False,
    solver_cfg: MJWarpSolverCfg | None = None,
    **newton_kwargs: Any,
) -> SimulationCfg:
    """Return an MJWarp simulation configuration, without CUDA-graph capture unless requested.

    Args:
        device: Simulation device.
        dt: Physics time step [s].
        gravity: Scene gravity [m/s^2]. Defaults to none, so free bodies only move when a test drives them.
        use_newton_actuators: Whether explicit actuator groups execute as Newton-native actuators.
        newton_contacts: Whether to use Newton's collision pipeline instead of MuJoCo contacts.
        solver_cfg: Solver configuration. Defaults to :class:`MJWarpSolverCfg` with the selected contacts.
        **newton_kwargs: Additional :class:`NewtonCfg` fields.

    Returns:
        The simulation configuration.
    """
    if solver_cfg is None:
        solver_cfg = MJWarpSolverCfg(use_mujoco_contacts=not newton_contacts)
    newton_kwargs.setdefault("use_cuda_graph", False)
    return SimulationCfg(
        device=device,
        dt=dt,
        gravity=gravity,
        use_newton_actuators=use_newton_actuators,
        physics=NewtonCfg(solver_cfg=solver_cfg, **newton_kwargs),
    )


def env_origins(device: str, num_envs: int = NUM_ENVS) -> torch.Tensor:
    """Return the world origins :func:`spawn_assets` places the environments at [m]."""
    origins = torch.zeros((num_envs, 3), device=device)
    origins[:, 0] = _ENV_SPACING * torch.arange(num_envs, device=device)
    return origins


def spawn_assets(
    asset_cfgs: Mapping[str, ArticulationCfg | RigidObjectCfg | RigidObjectCollectionCfg], num_envs: int = NUM_ENVS
) -> dict[str, Any]:
    """Plan, construct, and replicate assets into ``/World/Env_*`` of the active simulation.

    Call this inside a simulation context and before :meth:`~isaaclab.sim.SimulationContext.reset`.
    Each configuration's ``prim_path`` must address one asset per environment.

    Args:
        asset_cfgs: Asset configurations keyed by a name the caller uses to look the asset up.
        num_envs: Number of environments.

    Returns:
        The Newton assets keyed like ``asset_cfgs``.
    """
    positions = env_origins("cpu", num_envs).numpy()
    # a collection is planned as its member rigid objects
    planned_cfgs = []
    for cfg in asset_cfgs.values():
        planned_cfgs.extend(cfg.rigid_objects.values() if isinstance(cfg, RigidObjectCollectionCfg) else (cfg,))
    sim_utils.create_prim("/World/Env_0", "Xform")
    clone_plan_from_env_0(
        CloneCfg(clone_template="/World/Env_{}"), planned_cfgs, num_envs, _ENV_SPACING, positions=positions
    )
    assets = {}
    for name, cfg in asset_cfgs.items():
        if isinstance(cfg, ArticulationCfg):
            assets[name] = Articulation(cfg)
        elif isinstance(cfg, RigidObjectCollectionCfg):
            assets[name] = RigidObjectCollection(cfg)
        elif isinstance(cfg, RigidObjectCfg):
            assets[name] = RigidObject(cfg)
        else:
            raise TypeError(f"Unsupported asset configuration for '{name}': {type(cfg).__name__}.")
    replicate(SimulationContext.instance().get_clone_plan())
    return assets


@contextmanager
def world_gravity(gravity: Vec3) -> Iterator[None]:
    """Apply the same gravity to every world of the live Newton model, then restore the previous values.

    Args:
        gravity: Gravity applied to every world [m/s^2].
    """
    model = SimulationManager.get_model()
    world_gravity_view = wp.to_torch(model.gravity[: model.world_count])
    previous = world_gravity_view.clone()
    world_gravity_view.copy_(torch.tensor(gravity, device=previous.device))
    SimulationManager.add_model_change(ModelFlags.MODEL_PROPERTIES)
    try:
        yield
    finally:
        world_gravity_view.copy_(previous)
        SimulationManager.add_model_change(ModelFlags.MODEL_PROPERTIES)
