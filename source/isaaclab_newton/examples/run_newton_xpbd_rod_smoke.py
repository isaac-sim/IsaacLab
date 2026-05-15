#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke test: Newton ``SolverXPBDRod`` via :class:`NewtonXPBDRodSolver` (no Isaac Sim).

Requires a Newton install that includes ``SolverXPBDRod`` (see PR #1981)::

    pip install "newton @ git+https://github.com/newton-physics/newton.git@refs/pull/1981/head"

Run::

    ./isaaclab.sh -p source/isaaclab_newton/examples/run_newton_xpbd_rod_smoke.py

Or with conda env that has ``newton`` + ``warp``::

    python source/isaaclab_newton/examples/run_newton_xpbd_rod_smoke.py
"""

from __future__ import annotations

import torch

from isaaclab_newton.solvers import NewtonXPBDRodSolver, RodConfig, RodGeometryConfig, RodMaterialConfig, RodSolverConfig


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = RodConfig(
        material=RodMaterialConfig(
            young_modulus=1.0e6,
            density=2700.0,
            damping=0.01,
            bend_stiffness=0.2,
            twist_stiffness=0.2,
        ),
        geometry=RodGeometryConfig(num_segments=32, rest_length=1.6, radius=0.003),
        solver=RodSolverConfig(dt=1.0 / 120.0, num_substeps=2, gravity=(0.0, 0.0, -9.81)),
        device=device,
    )
    solver = NewtonXPBDRodSolver(cfg, num_envs=1, floor_z=None, initial_z=1.0)
    z0 = float(solver.positions[0, -1, 2].cpu())
    for _ in range(240):
        solver.step()
    z1 = float(solver.positions[0, -1, 2].cpu())
    print(f"NewtonXPBDRodSolver smoke: tip Z {z0:.4f} -> {z1:.4f} (expect drop under -Z gravity)")
    if z1 >= z0:
        raise SystemExit("Expected tip to move down under gravity")


if __name__ == "__main__":
    main()
