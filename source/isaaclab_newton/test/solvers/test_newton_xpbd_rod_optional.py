# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Optional tests for :class:`NewtonXPBDRodSolver` when Newton includes ``SolverXPBDRod``."""

from __future__ import annotations

import pytest
import torch


def _newton_has_xpbd_rod() -> bool:
    try:
        import newton.solvers

        return hasattr(newton.solvers, "SolverXPBDRod")
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _newton_has_xpbd_rod(),
    reason="Newton with SolverXPBDRod not installed (install PR #1981 branch or merged wheel)",
)


def test_newton_xpbd_rod_smoke():
    from isaaclab_newton.solvers import (
        NewtonXPBDRodSolver,
        RodConfig,
        RodGeometryConfig,
        RodMaterialConfig,
        RodSolverConfig,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = RodConfig(
        material=RodMaterialConfig(
            young_modulus=1.0e5,
            density=1000.0,
            damping=0.02,
            bend_stiffness=0.15,
            twist_stiffness=0.15,
        ),
        geometry=RodGeometryConfig(num_segments=16, rest_length=0.8, radius=0.002),
        solver=RodSolverConfig(dt=0.002, num_substeps=4, gravity=(0.0, 0.0, -9.81)),
        device=device,
    )
    solver = NewtonXPBDRodSolver(cfg, floor_z=None, initial_z=0.5)
    assert solver.positions.shape == (1, 16, 3)
    z0 = float(solver.positions[0, -1, 2].cpu())
    for _ in range(100):
        solver.step()
    z1 = float(solver.positions[0, -1, 2].cpu())
    assert z1 < z0, "tip should descend"


def test_newton_xpbd_num_envs_rejects_batch():
    from isaaclab_newton.solvers import NewtonXPBDRodSolver, RodConfig

    with pytest.raises(NotImplementedError):
        NewtonXPBDRodSolver(RodConfig(), num_envs=2)
