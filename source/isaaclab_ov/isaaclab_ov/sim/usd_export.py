# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export a running OVPhysX articulation, as simulated, to USD.

Unlike the PhysX tensor view, an OVPhysX binding does not record the prim each link and degree of
freedom came from -- it reports names and the articulation prims it matched. The paths are therefore
resolved from the stage: the articulation root for the environment is taken from the view, and its
subtree is indexed by prim name, which is what the backend's body and joint names are.

Everything else is the shared :class:`~isaaclab.sim.usd_export.ArticulationExporter` -- see
:mod:`isaaclab.sim.usd_export` for what is written and why the stage is patched rather than rebuilt.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pxr import Usd, UsdPhysics

from isaaclab.sim.usd_export import ArticulationExporter, ArticulationPrimPaths

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene

    from isaaclab_ov.assets import Articulation

__all__ = [
    "export_articulation_to_usd",
    "export_environment_to_usd",
    "exporter",
    "resolve_articulation_prim_paths",
    "write_articulation_state_to_stage",
]


def _index_subtree_by_name(stage: Usd.Stage, root_path: str) -> tuple[dict[str, str], dict[str, str]]:
    """Map prim name to prim path under ``root_path``, separately for joints and for everything else.

    Bodies and joints are separate name spaces in the backend and may collide on the stage -- Ant has a
    body and a joint both called ``front_left_leg`` -- so a single index would send joint writes onto
    the body prim. Within a kind the shallowest prim wins, matching the one the articulation was built
    from rather than a nested duplicate.

    Args:
        stage: Stage holding the articulation.
        root_path: Prim path to index beneath.

    Returns:
        ``(bodies, joints)`` maps of prim name to prim path.
    """
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise ValueError(f"Articulation root '{root_path}' is not a prim on the stage.")

    bodies: dict[str, str] = {}
    joints: dict[str, str] = {}
    for prim in Usd.PrimRange(root):
        index = joints if prim.IsA(UsdPhysics.Joint) else bodies
        path = prim.GetPath().pathString
        current = index.get(prim.GetName())
        if current is None or path.count("/") < current.count("/"):
            index[prim.GetName()] = path
    return bodies, joints


def resolve_articulation_prim_paths(
    articulation: Articulation, env_index: int = 0, *, stage: Usd.Stage | None = None
) -> ArticulationPrimPaths:
    """Resolve one environment's body and joint prim paths from the stage.

    Args:
        articulation: The articulation whose prims to resolve. It must be initialized.
        env_index: Environment whose paths to take. Defaults to ``0``.
        stage: Optional materialized scene snapshot for resolving kitless clone paths.

    Returns:
        The environment's prim paths, in backend index order.

    Raises:
        ValueError: If the view holds no such environment, or a body or joint has no prim under the
            articulation root -- which would otherwise export a partial articulation.
    """
    roots = articulation.root_view.prim_paths
    if not 0 <= env_index < len(roots):
        raise ValueError(f"Environment {env_index} is out of range for a view matching {len(roots)} articulations.")

    # The view matches the prim carrying ArticulationRootAPI. Where that sits varies by asset: the
    # top-level Xform (Franka), or a link whose siblings are the other links (Ant's torso). Walk up
    # from it to the nearest ancestor whose subtree holds every body and joint name; that ancestor is
    # the articulation, whatever the asset called it.
    stage = articulation.stage if stage is None else stage
    body_names, joint_names = articulation.backend_body_names, articulation.backend_joint_names
    prim = stage.GetPrimAtPath(roots[env_index])
    while True:
        index_root = prim.GetPath().pathString
        body_paths, joint_paths = _index_subtree_by_name(stage, index_root)
        if set(body_names) <= set(body_paths) and set(joint_names) <= set(joint_paths):
            break
        if prim.GetParent().IsPseudoRoot() or not prim.GetParent().IsValid():
            break
        prim = prim.GetParent()

    def resolve(names: list[str], paths: dict[str, str], kind: str) -> list[str]:
        missing = [name for name in names if name not in paths]
        if missing:
            raise ValueError(
                f"No {kind} prim under '{index_root}' for: {', '.join(missing)}. Exporting would"
                f" describe only part of the articulation."
            )
        return [paths[name] for name in names]

    return ArticulationPrimPaths(
        bodies=resolve(body_names, body_paths, "body"),
        joints=resolve(joint_names, joint_paths, "joint"),
    )


def exporter(articulation: Articulation) -> ArticulationExporter:
    """Exporter for an OVPhysX articulation, resolving prim paths from the stage."""
    return ArticulationExporter(articulation, resolve_articulation_prim_paths)


def write_articulation_state_to_stage(
    articulation: Articulation, env_index: int = 0, *, stage: Usd.Stage | None = None
) -> list[str]:
    """Author an OVPhysX articulation's simulated state onto the prims it was spawned from.

    See :meth:`~isaaclab.sim.usd_export.ArticulationExporter.write_to_stage`.
    """
    return exporter(articulation).write_to_stage(env_index, stage=stage)


def export_articulation_to_usd(articulation: Articulation, usd_path: str, env_index: int = 0) -> str:
    """Export one environment's OVPhysX articulation, as simulated, to a USD file.

    See :meth:`~isaaclab.sim.usd_export.ArticulationExporter.export`.
    """
    return exporter(articulation).export(usd_path, env_index)


def export_environment_to_usd(scene: InteractiveScene, usd_path: str, env_index: int = 0) -> str:
    """Export one OVPhysX environment; see :func:`isaaclab.sim.usd_export.export_environment_to_usd`."""
    from isaaclab.sim.usd_export import export_stage_environment

    from isaaclab_ov.physics import OvPhysxManager

    return export_stage_environment(
        scene,
        usd_path,
        env_index,
        lambda asset, row, stage: resolve_articulation_prim_paths(asset, row, stage=stage),
        _read_body_properties,
        OvPhysxManager.get_gravity(),
    )


def _read_body_properties(asset, row: int, articulation: bool):
    from isaaclab.sim.usd_export import RigidBodyExportProperties

    from isaaclab_ov import tensor_types as TT

    types = (
        (TT.BODY_DISABLE_GRAVITY, TT.SHAPE_FRICTION_AND_RESTITUTION, TT.CONTACT_OFFSET, TT.REST_OFFSET)
        if articulation
        else (
            TT.RIGID_BODY_DISABLE_GRAVITY,
            TT.RIGID_BODY_SHAPE_FRICTION_AND_RESTITUTION,
            TT.RIGID_BODY_CONTACT_OFFSET,
            TT.RIGID_BODY_REST_OFFSET,
        )
    )
    return RigidBodyExportProperties(*(asset.root_view.get_attribute(token).numpy()[row] for token in types))
