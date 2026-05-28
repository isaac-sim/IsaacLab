# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script contains helper functions for the demos.
"""

from contextlib import contextmanager


@contextmanager
def resolve_backend_and_visualizer(args, newton_cfg=None):
    """Resolve physics + visualizer cfgs from ``--physics`` / ``--visualizer``.

    Yields ``(physics_cfg, visualizer_cfg)``. Kit is launched when required by
    either the visualizer or the physics backend, and closed automatically on exit.
    """

    # Resolve physics cfg
    if args.physics == "physx":
        from isaaclab_physx.physics import PhysxCfg

        physics_cfg = PhysxCfg()
    elif args.physics == "newton_mjwarp":
        from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

        DEFAULT_NEWTON_CFG = NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=70,
                nconmax=70,
                ls_iterations=40,
                cone="elliptic",
                impratio=100,
                ls_parallel=False,
                integrator="implicitfast",
            ),
            num_substeps=2,
        )
        physics_cfg = newton_cfg or DEFAULT_NEWTON_CFG
    else:
        raise ValueError(f"Unsupported --physics value: {args.physics}")

    # Resolve visualizer cfg
    if not isinstance(args.visualizer, list) or len(args.visualizer) != 1:
        raise ValueError("Demos support exactly one --visualizer value: kit or newton.")
    viz_type = args.visualizer[0]
    if viz_type == "kit":
        from isaaclab_visualizers.kit import KitVisualizerCfg

        visualizer_cfg = KitVisualizerCfg()
    elif viz_type == "newton":
        from isaaclab_visualizers.newton import NewtonVisualizerCfg

        visualizer_cfg = NewtonVisualizerCfg()
    else:
        raise ValueError(f"Unsupported --visualizer value: {viz_type}")

    # PhysX requires Isaac Sim Kit extensions even when rendering through a
    # standalone visualizer such as Newton.
    close_fn = None
    if viz_type == "kit" or args.physics == "physx":
        from isaaclab.app import AppLauncher

        args.visualizer = [viz_type]
        close_fn = AppLauncher(args).app.close

    try:
        yield physics_cfg, visualizer_cfg
    finally:
        # No-op for the newton visualizer; close Kit automatically upon exit
        if close_fn is not None:
            close_fn()


def has_no_alive_visualizer_window(sim) -> bool:
    """Check if there are no alive visualizer windows."""
    visualizers = sim.visualizers or ()
    return not any(v.is_running() and not v.is_closed for v in visualizers)
