#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Headless smoke test for the self-contained XPBDRodSolver.

No Isaac Sim / Omniverse dependency — only warp + torch.
Prints tip position each frame and asserts the rod falls under gravity.

Usage:
    python source/isaaclab_newton/examples/run_xpbd_rod_smoke.py
"""

from __future__ import annotations

import sys
import time

import torch

sys.path.insert(0, "source/isaaclab_newton")

from isaaclab_newton.solvers import (
    RodConfig,
    RodGeometryConfig,
    RodMaterialConfig,
    RodSolverConfig,
    XPBDRodSolver,
    orientations_xyzw_along_polyline,
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    cfg = RodConfig(
        material=RodMaterialConfig(
            young_modulus=1e4,
            density=6450.0,
            damping=0.05,
            bend_stiffness=0.5,
            twist_stiffness=0.3,
        ),
        geometry=RodGeometryConfig(
            num_segments=20,
            rest_length=1.5,
            radius=0.015,
        ),
        solver=RodSolverConfig(
            dt=1.0 / 120.0,
            num_substeps=2,
            gravity=(0.0, 0.0, -9.81),
        ),
        device=device,
    )

    print(f"Creating XPBDRodSolver: {cfg.geometry.num_segments} segments, "
          f"rest_length={cfg.geometry.rest_length}m ...")

    solver = XPBDRodSolver(cfg, initial_height=0.8, floor_z=None)

    print(f"  num_points = {solver.num_points}")
    print(f"  num_edges  = {solver.num_edges}")

    initial_tip = solver.positions[-1].clone()
    print(f"  initial tip = ({initial_tip[0]:.4f}, {initial_tip[1]:.4f}, {initial_tip[2]:.4f})")
    print()

    dt = cfg.solver.dt
    n_frames = 300
    print(f"Stepping {n_frames} frames at dt={dt:.4f}s ...")

    t0 = time.perf_counter()
    for frame in range(n_frames):
        solver.step(dt)
        if frame % 60 == 0 or frame == n_frames - 1:
            tip = solver.positions[-1]
            ori = solver.orientations[-1]
            print(
                f"  frame {frame:4d}  tip=({tip[0]:.4f}, {tip[1]:.4f}, {tip[2]:.4f})  "
                f"q=({ori[0]:.3f}, {ori[1]:.3f}, {ori[2]:.3f}, {ori[3]:.3f})"
            )
    elapsed = time.perf_counter() - t0
    print(f"\n{n_frames} frames in {elapsed:.2f}s  ({n_frames / elapsed:.1f} fps)")

    final_tip = solver.positions[-1]
    dz = float(final_tip[2] - initial_tip[2])
    print(f"\nTip Z drop: {dz:.4f} m")

    quats = orientations_xyzw_along_polyline(solver.positions)
    print(f"orientations_xyzw_along_polyline shape: {quats.shape}")

    assert dz < -0.01, f"Expected tip to fall under gravity, but dz={dz:.4f}"
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
