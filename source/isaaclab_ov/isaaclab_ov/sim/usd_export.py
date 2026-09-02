# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export a running OVPhysX articulation, as simulated, to USD.

Unlike the PhysX tensor view, an OVPhysX binding does not record the prim each link and degree of
freedom came from -- it reports names and the articulation prims it matched. The paths are therefore
resolved from the stage: the articulation root for the environment is taken from the view, and its
subtree is indexed by prim name, which is what the backend's body and joint names are.

Authoring the values is shared with the other stage-backed backends -- see
:mod:`isaaclab.sim.usd_export` for what is written and why the stage is patched rather than rebuilt.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pxr import Usd

from isaaclab.sim import usd_export as shared
from isaaclab.sim.usd_export import ArticulationPrimPaths

if TYPE_CHECKING:
    from isaaclab_ov.assets import Articulation

__all__ = ["export_articulation_to_usd", "resolve_articulation_prim_paths", "write_articulation_state_to_stage"]


def _index_subtree_by_name(stage: Usd.Stage, root_path: str) -> dict[str, str]:
    """Map prim name to prim path for everything under ``root_path``.

    Names are unique within an articulation -- the backend identifies bodies and joints by them --
    so a name-keyed index is enough to recover a path. Where a name does repeat, the shallowest prim
    wins, matching the one the articulation was built from rather than a nested duplicate.

    Args:
        stage: Stage holding the articulation.
        root_path: Prim path of the articulation root.

    Returns:
        A map of prim name to prim path.
    """
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise ValueError(f"Articulation root '{root_path}' is not a prim on the stage.")

    index: dict[str, str] = {}
    for prim in Usd.PrimRange(root):
        path = prim.GetPath().pathString
        name = prim.GetName()
        current = index.get(name)
        if current is None or path.count("/") < current.count("/"):
            index[name] = path
    return index


def resolve_articulation_prim_paths(articulation: Articulation, env_index: int = 0) -> ArticulationPrimPaths:
    """Resolve one environment's body and joint prim paths from the stage.

    Args:
        articulation: The articulation whose prims to resolve. It must be initialized.
        env_index: Environment whose paths to take. Defaults to ``0``.

    Returns:
        The environment's prim paths, in backend index order.

    Raises:
        ValueError: If the view holds no such environment, or a body or joint has no prim under the
            articulation root -- which would otherwise export a partial articulation.
    """
    roots = articulation.root_view.prim_paths
    if env_index >= len(roots):
        raise ValueError(f"Environment {env_index} is out of range for a view matching {len(roots)} articulations.")

    by_name = _index_subtree_by_name(articulation.stage, roots[env_index])

    def resolve(names: list[str], kind: str) -> list[str]:
        missing = [name for name in names if name not in by_name]
        if missing:
            raise ValueError(
                f"No prim under '{roots[env_index]}' for {kind}: {', '.join(missing)}. Exporting would"
                f" describe only part of the articulation."
            )
        return [by_name[name] for name in names]

    return ArticulationPrimPaths(
        bodies=resolve(articulation.backend_body_names, "bodies"),
        joints=resolve(articulation.backend_joint_names, "joints"),
    )


def write_articulation_state_to_stage(articulation: Articulation, env_index: int = 0) -> list[str]:
    """Author an OVPhysX articulation's simulated state onto the prims it was spawned from.

    Args:
        articulation: The articulation to read.
        env_index: Environment to write. Defaults to ``0``.

    Returns:
        The prim paths written, bodies first.
    """
    paths = resolve_articulation_prim_paths(articulation, env_index)
    return shared.write_articulation_state_to_stage(articulation, paths, env_index=env_index)


def export_articulation_to_usd(articulation: Articulation, usd_path: str, env_index: int = 0) -> str:
    """Export one environment's OVPhysX articulation, as simulated, to a USD file.

    Args:
        articulation: The articulation to export.
        usd_path: Destination path for the USD file.
        env_index: Environment to export. Defaults to ``0``.

    Returns:
        The path the stage was written to.
    """
    paths = resolve_articulation_prim_paths(articulation, env_index)
    return shared.export_articulation_to_usd(articulation, paths, usd_path, env_index=env_index)
