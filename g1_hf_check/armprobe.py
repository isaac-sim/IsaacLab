# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train wrapper that pins the rough-terrain collision representation and proves which one ran.

``ROUGH_TERRAINS_CFG`` becomes a native Newton heightfield only when *every* sub-terrain sets
``convert_to_heightfield``. The three mesh sub-terrains alone decide that all-or-nothing gate, so
flipping them selects the arm without touching anything else in the task config.

Environment variables:
    ARM: ``heightfield`` (stock default) or ``mesh``.
    MARGIN: override the rough-terrain Newton shape margin [m] (stock default 0.0).
    SUBSTEPS: override ``num_substeps`` (stock default 1; G1 and Cassie set 2 themselves).

Every other argument is forwarded verbatim to the training CLI.
"""

import os
import sys

TAG = "ARMPROBE:"
ARM = os.environ.get("ARM", "heightfield")
MARGIN = os.environ.get("MARGIN")
SUBSTEPS = os.environ.get("SUBSTEPS")
# Height-field sub-terrains default the flag to True, so these three alone flip the gate.
_MESH_SUB_TERRAINS = ("pyramid_stairs", "pyramid_stairs_inv", "boxes")


def _install_arm() -> bool:
    """Select the terrain arm and return whether a heightfield collider is expected."""
    if ARM not in ("heightfield", "mesh"):
        raise SystemExit(f"{TAG} FATAL: ARM must be 'heightfield' or 'mesh', got {ARM!r}")
    expect_heightfield = ARM == "heightfield"

    from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

    if not expect_heightfield:
        for name in _MESH_SUB_TERRAINS:
            ROUGH_TERRAINS_CFG.sub_terrains[name].convert_to_heightfield = False

    gate = all(sub.convert_to_heightfield for sub in ROUGH_TERRAINS_CFG.sub_terrains.values())
    print(f"{TAG} arm={ARM} gate={gate}", flush=True)
    if gate is not expect_heightfield:
        raise SystemExit(f"{TAG} FATAL: gate={gate} but arm={ARM} requires {expect_heightfield}")
    return expect_heightfield


def _install_conversion_check(expect_heightfield: bool) -> None:
    """Assert at solver-init time that the realized terrain matches the requested arm."""
    import isaaclab_newton.physics.newton_manager as newton_manager

    original = newton_manager.NewtonManager._inject_terrain_heightfields.__func__

    def checked_inject(cls, stage, builder):
        converted = original(cls, stage, builder)
        print(f"{TAG} converted={list(converted)}", flush=True)
        if expect_heightfield and not converted:
            raise SystemExit(f"{TAG} FATAL: arm=heightfield but the terrain was not converted")
        if not expect_heightfield and converted:
            raise SystemExit(f"{TAG} FATAL: arm=mesh but the terrain was converted: {list(converted)}")
        print(f"{TAG} CONFIRMED arm={ARM}", flush=True)
        return converted

    newton_manager.NewtonManager._inject_terrain_heightfields = classmethod(checked_inject)


def _install_margin() -> None:
    """Override the rough-terrain Newton shape margin the same way the task configs do.

    ``RoughPhysicsCfg`` is a ``configclass``, so its preset attributes are dataclass fields
    backed by a ``default_factory`` and cannot be patched on the class. Instead the override
    is applied inside ``__post_init__``, which is where G1 and Cassie already reach through
    ``self.sim.physics.newton_mjwarp`` to set ``num_substeps``.
    """
    if MARGIN is None and SUBSTEPS is None:
        return
    from isaaclab_tasks.core.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

    margin = None if MARGIN is None else float(MARGIN)
    substeps = None if SUBSTEPS is None else int(SUBSTEPS)
    original = LocomotionVelocityRoughEnvCfg.__post_init__

    def patched(self):
        original(self)
        newton = self.sim.physics.newton_mjwarp
        if margin is not None:
            newton.default_shape_cfg.margin = margin
        if substeps is not None:
            newton.num_substeps = substeps
        print(f"{TAG} margin={newton.default_shape_cfg.margin} num_substeps={newton.num_substeps}", flush=True)

    LocomotionVelocityRoughEnvCfg.__post_init__ = patched


_install_conversion_check(_install_arm())
_install_margin()

from isaaclab_rl.entrypoints import run_train_cli  # noqa: E402

raise SystemExit(run_train_cli(sys.argv[1:]))
