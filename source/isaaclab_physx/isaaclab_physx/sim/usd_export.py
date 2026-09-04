# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export a running PhysX articulation, as simulated, to USD.

PhysX records prim-path provenance on its tensor view: the view knows the prim every link and degree
of freedom was built from, so the paths are read straight off it. Authoring the values is shared
with the other stage-backed backends -- see :mod:`isaaclab.sim.usd_export` for what is written and
why the stage is patched rather than rebuilt.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pxr import Usd

from isaaclab.sim import usd_export as shared
from isaaclab.sim.usd_export import ArticulationPrimPaths

if TYPE_CHECKING:
    from isaaclab_physx.assets import Articulation

__all__ = ["export_articulation_to_usd", "resolve_articulation_prim_paths", "write_articulation_state_to_stage"]


def resolve_articulation_prim_paths(articulation: Articulation, env_index: int = 0) -> ArticulationPrimPaths:
    """Read one environment's body and joint prim paths off the PhysX tensor view.

    Args:
        articulation: The articulation whose view to read. It must be initialized.
        env_index: Environment whose paths to take. Defaults to ``0``.

    Returns:
        The environment's prim paths, in backend index order.

    Raises:
        ValueError: If the view holds no such environment.
    """
    view = articulation.root_view
    if env_index >= len(view.link_paths):
        raise ValueError(f"Environment {env_index} is out of range for a view with {len(view.link_paths)} rows.")
    return ArticulationPrimPaths(
        bodies=[str(path) for path in view.link_paths[env_index]],
        joints=[str(path) for path in view.dof_paths[env_index]],
    )


def write_articulation_state_to_stage(
    articulation: Articulation, env_index: int = 0, *, stage: Usd.Stage | None = None
) -> list[str]:
    """Author a PhysX articulation's simulated state onto the prims it was spawned from.

    Args:
        articulation: The articulation to read.
        env_index: Environment to write. Defaults to ``0``.
        stage: Stage to author onto; defaults to the live stage. See
            :func:`isaaclab.sim.usd_export.write_articulation_state_to_stage`.

    Returns:
        The prim paths written, bodies first.
    """
    paths = resolve_articulation_prim_paths(articulation, env_index)
    return shared.write_articulation_state_to_stage(articulation, paths, env_index=env_index, stage=stage)


def export_articulation_to_usd(articulation: Articulation, usd_path: str, env_index: int = 0) -> str:
    """Export one environment's PhysX articulation, as simulated, to a USD file.

    Args:
        articulation: The articulation to export.
        usd_path: Destination path for the USD file.
        env_index: Environment to export. Defaults to ``0``.

    Returns:
        The path the stage was written to.
    """
    paths = resolve_articulation_prim_paths(articulation, env_index)
    return shared.export_articulation_to_usd(articulation, paths, usd_path, env_index=env_index)
