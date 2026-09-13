# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Prove that the stock rough-terrain config collides against a native Newton heightfield.

Runs the unmodified Isaac Lab training CLI, but wraps
``NewtonManager._inject_terrain_heightfields`` so the realized terrain representation
is printed (and asserted) at solver-init time. Nothing about the task config is changed.

All arguments are forwarded verbatim to the training CLI.
"""

import sys

TAG = "HFPROBE:"


def _report_gate() -> None:
    """Print the all-or-nothing heightfield gate of the stock ``ROUGH_TERRAINS_CFG``."""
    from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

    gate = all(sub.convert_to_heightfield for sub in ROUGH_TERRAINS_CFG.sub_terrains.values())
    print(f"{TAG} stock gate all(convert_to_heightfield)={gate}", flush=True)
    for name, sub in ROUGH_TERRAINS_CFG.sub_terrains.items():
        print(f"{TAG}   {name}: convert_to_heightfield={sub.convert_to_heightfield}", flush=True)


def _install_probe() -> None:
    """Assert at solver-init time that the terrain became a Newton heightfield."""
    import isaaclab_newton.physics.newton_manager as newton_manager

    original = newton_manager.NewtonManager._inject_terrain_heightfields.__func__

    def checked_inject(cls, stage, builder):
        converted = original(cls, stage, builder)
        n_hfield = 0
        try:
            import newton

            n_hfield = sum(1 for t in builder.shape_type if int(t) == int(newton.GeoType.HFIELD))
        except Exception as exc:  # noqa: BLE001
            print(f"{TAG} WARN: could not count heightfield shapes: {exc}", flush=True)
        print(f"{TAG} converted={list(converted)} hfield_shapes={n_hfield}", flush=True)
        if not converted:
            raise SystemExit(f"{TAG} FATAL: terrain was NOT converted to a heightfield")
        print(f"{TAG} CONFIRMED native heightfield collider", flush=True)
        return converted

    newton_manager.NewtonManager._inject_terrain_heightfields = classmethod(checked_inject)


_report_gate()
_install_probe()

from isaaclab_rl.entrypoints import run_train_cli  # noqa: E402

raise SystemExit(run_train_cli(sys.argv[1:]))
