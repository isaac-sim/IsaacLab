# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load-time adapter for Mujoco Menagerie asset conversions.

The Menagerie source models (https://github.com/google-deepmind/mujoco_menagerie) are
correct and complete for MuJoCo. The USD conversions, however, drop guarantees that
MuJoCo provides implicitly and that other engines need authored explicitly. This module
is the single bridge for those conversion defects: every fix is applied to the composed
prims at spawn time, before the physics engine parses them, and is tagged with the
upstream asset/converter change that makes it deletable.

Faithful source content (base frames, masses, joint limits, naming) is intentionally
NOT touched here — task and robot configurations adapt to it instead.
"""

import os
from collections.abc import Callable
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from pxr import Usd

MENAGERIE_ASSET_ROOT = os.environ.get("MENAGERIE_ASSET_ROOT", f"{ISAAC_NUCLEUS_DIR}/Samples/Mujoco_Menagerie")
"""Root of the Mujoco Menagerie asset conversions.

Resolves through the Isaac asset root to the public production release (S3), so no
Nucleus authentication is required. Override with the ``MENAGERIE_ASSET_ROOT``
environment variable to point at a local mirror.
"""


def _author_static_friction(prim: "Usd.Prim") -> None:
    """Author ``physics:staticFriction`` on physics materials that lack it.

    UPSTREAM(asset): the converter authors only ``physics:dynamicFriction``; the static
    coefficient falls back to 0 and grasped objects slide out of a motionless hand.
    MuJoCo has a single sliding coefficient that plays both roles, so the faithful
    translation copies the dynamic value. Delete once the converter authors it.
    """
    from pxr import Usd, UsdPhysics

    for child in Usd.PrimRange(prim):
        if not child.HasAPI(UsdPhysics.MaterialAPI):
            continue
        material = UsdPhysics.MaterialAPI(child)
        static_attr = material.GetStaticFrictionAttr()
        if static_attr and static_attr.HasAuthoredValue():
            continue
        dynamic_attr = material.GetDynamicFrictionAttr()
        dynamic = dynamic_attr.Get() if dynamic_attr and dynamic_attr.HasAuthoredValue() else None
        if dynamic is not None:
            material.CreateStaticFrictionAttr(dynamic)


def _author_filtered_pairs(prim: "Usd.Prim", pairs: list[tuple[str, str]]) -> None:
    """Author collision filtered pairs between bodies named in ``pairs``.

    UPSTREAM(asset): the converter splits welded geoms (e.g. fingertips) out of their
    source body, manufacturing collision pairs that cannot exist in MuJoCo, where
    same-body geoms never collide. The durable fix is to not split (or to author these
    filters in the conversion); delete once that lands.
    """
    from pxr import Usd, UsdPhysics

    bodies = {child.GetName(): child for child in Usd.PrimRange(prim)}
    for name_a, name_b in pairs:
        body_a, body_b = bodies.get(name_a), bodies.get(name_b)
        if body_a is None or body_b is None:
            continue
        api = UsdPhysics.FilteredPairsAPI.Apply(body_a)
        api.GetFilteredPairsRel().AddTarget(body_b.GetPath())


def spawn_menagerie_asset(
    prim_path: str,
    cfg: "MenagerieUsdFileCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> "Usd.Prim":
    """Spawn a Menagerie asset conversion and apply the load-time conversion fixes."""
    # Deferred imports: this module is imported by the task-config registry before the
    # simulation app starts, when pxr and the spawner internals are not yet loadable.
    from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
    from isaaclab.sim.utils import clone

    @clone
    def _spawn(prim_path, cfg, translation=None, orientation=None, **inner_kwargs):
        prim = _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)
        if cfg.author_static_friction:
            _author_static_friction(prim)
        if cfg.filtered_body_pairs:
            _author_filtered_pairs(prim, cfg.filtered_body_pairs)
        for fixup in cfg.extra_fixups:
            fixup(prim)
        return prim

    return _spawn(prim_path, cfg, translation, orientation, **kwargs)


@configclass
class MenagerieUsdFileCfg(sim_utils.UsdFileCfg):
    """Spawn configuration for Menagerie conversions with load-time conversion fixes."""

    func: Callable = spawn_menagerie_asset

    author_static_friction: bool = True
    """Copy the material's dynamic friction to the unauthored static coefficient."""

    filtered_body_pairs: list[tuple[str, str]] = []
    """Body-name pairs to exclude from collision (pairs manufactured by the conversion)."""

    extra_fixups: list[Callable] = []
    """Additional per-asset fixups called with the spawned prim (e.g. tendon authoring)."""
