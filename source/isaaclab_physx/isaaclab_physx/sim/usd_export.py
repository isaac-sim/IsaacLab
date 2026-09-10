# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export a running PhysX articulation, as simulated, to USD.

PhysX records prim-path provenance on its tensor view: the view knows the prim every link and degree
of freedom was built from, so the paths are read straight off it. Everything else is the shared
:class:`~isaaclab.sim.usd_export.ArticulationExporter` -- see :mod:`isaaclab.sim.usd_export` for what
is written and why the stage is patched rather than rebuilt.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pxr import Usd

from isaaclab.sim.usd_export import ArticulationExporter, ArticulationPrimPaths

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene

    from isaaclab_physx.assets import Articulation

__all__ = [
    "export_articulation_to_usd",
    "export_environment_to_usd",
    "exporter",
    "resolve_articulation_prim_paths",
    "write_articulation_state_to_stage",
]


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
    if not 0 <= env_index < len(view.link_paths):
        raise ValueError(f"Environment {env_index} is out of range for a view with {len(view.link_paths)} rows.")
    return ArticulationPrimPaths(
        bodies=[str(path) for path in view.link_paths[env_index]],
        joints=[str(path) for path in view.dof_paths[env_index]],
    )


def exporter(articulation: Articulation) -> ArticulationExporter:
    """Exporter for a PhysX articulation, resolving prim paths off its tensor view."""
    return ArticulationExporter(articulation, resolve_articulation_prim_paths)


def write_articulation_state_to_stage(
    articulation: Articulation, env_index: int = 0, *, stage: Usd.Stage | None = None
) -> list[str]:
    """Author a PhysX articulation's simulated state onto the prims it was spawned from.

    See :meth:`~isaaclab.sim.usd_export.ArticulationExporter.write_to_stage`.
    """
    return exporter(articulation).write_to_stage(env_index, stage=stage)


def export_articulation_to_usd(articulation: Articulation, usd_path: str, env_index: int = 0) -> str:
    """Export one environment's PhysX articulation, as simulated, to a USD file.

    See :meth:`~isaaclab.sim.usd_export.ArticulationExporter.export`.
    """
    return exporter(articulation).export(usd_path, env_index)


def export_environment_to_usd(scene: InteractiveScene, usd_path: str, env_index: int = 0) -> str:
    """Export one PhysX environment; see :func:`isaaclab.sim.usd_export.export_environment_to_usd`."""
    from isaaclab.sim.usd_export import export_stage_environment

    from isaaclab_physx.physics import PhysxManager

    return export_stage_environment(
        scene,
        usd_path,
        env_index,
        lambda asset, row, _stage: resolve_articulation_prim_paths(asset, row),
        _read_body_properties,
        PhysxManager.get_physics_sim_view().get_gravity(),
    )


def _read_body_properties(asset, row: int, articulation: bool):
    from isaaclab.sim.usd_export import RigidBodyExportProperties

    view = asset.root_view
    return RigidBodyExportProperties(
        *(
            getter().numpy()[row]
            for getter in (
                view.get_disable_gravities,
                view.get_material_properties,
                view.get_contact_offsets,
                view.get_rest_offsets,
            )
        )
    )
