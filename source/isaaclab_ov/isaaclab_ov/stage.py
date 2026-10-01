# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulation-owned OVStage resources and shared attribute-column helpers.

``ovstage`` is a hard dependency of ``isaaclab_ov``, so it is imported unconditionally here.
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING

import numpy as np
import ovstage
import warp as wp

from pxr import Sdf, Usd

from isaaclab.sim import BackendCfg
from isaaclab.utils import configclass

from isaaclab_ov.ovstage_compat import HIERARCHY_COMPUTATION_MODEL

if TYPE_CHECKING:
    from isaaclab.visualizers import VisualizerCfg

logger = logging.getLogger(__name__)

# DLDataType for a 4x4 double matrix (``omni:xform`` column). ovstage stores omni:xform as one
# 16-lane float64 element per prim; wp.mat44d maps to the same layout via __dlpack__.
OVSTAGE_XFORM_DTYPE = ovstage.DLDataType(code=ovstage.DLDataTypeCode.kDLFloat, bits=64, lanes=16)

# DLDataType for a float32 3-vector (``points`` column). ovstage stores ``point3f[] points`` as one
# 3-lane float32 element per vertex.
OVSTAGE_POINT_DTYPE = ovstage.DLDataType(code=ovstage.DLDataTypeCode.kDLFloat, bits=32, lanes=3)


def create_ovstage(name: str) -> ovstage.Stage:
    """Create an ovstage stage using Isaac Lab's process-wide stage configuration.

    ovstage's hierarchy computation model drives its automatic world-transform updates. It is
    process-scoped rather than per-stage: ovstage applies it when the first process reference is
    acquired and raises if a later stage asks for a conflicting model while another stage is live.
    Every Isaac Lab stage is therefore created through this helper so the whole process agrees on
    one model.

    The model is requested explicitly rather than left implicit, so the model in force is visible
    at the call site. Which one is requested depends on the installed ovstage version; see
    :mod:`isaaclab_ov.ovstage_compat`. A selected model the installed enum does not carry falls
    back to the host model, so a version gate that runs ahead of the runtime degrades rather than
    preventing stage creation.

    Args:
        name: Instance name used for ovstage diagnostics.

    Returns:
        The created :class:`ovstage.Stage`.
    """
    hierarchy_computation_model = getattr(ovstage.HierarchyComputationModel, HIERARCHY_COMPUTATION_MODEL, None)
    if hierarchy_computation_model is None:
        logger.warning(
            "This ovstage does not expose HierarchyComputationModel.%s; falling back to CPU_INCREMENTAL.",
            HIERARCHY_COMPUTATION_MODEL,
        )
        # Left unguarded: an ovstage without the host model is broken, and should say so loudly.
        hierarchy_computation_model = ovstage.HierarchyComputationModel.CPU_INCREMENTAL
    config = ovstage.StageConfig(runtime_default_hierarchy_computation_model=hierarchy_computation_model)
    return ovstage.Stage(name, config=config)


def xform_tensor_from_numpy(xforms: np.ndarray) -> ovstage.DLTensor:
    """Wrap a ``(N, 4, 4)`` float64 host array as a 16-lane DLTensor for ``omni:xform`` writes.

    Args:
        xforms: Array of shape ``(N, 4, 4)`` with dtype ``float64``.

    Returns:
        A :class:`ovstage.DLTensor` with shape ``[N]`` and ``lanes=16``.
    """
    flat = np.ascontiguousarray(xforms, dtype=np.float64).reshape(-1)
    return ovstage.make_dltensor(flat, dtype=OVSTAGE_XFORM_DTYPE, shape=[xforms.shape[0]])


def xform_tensor_from_warp(xforms: wp.array) -> ovstage.DLTensor:
    """Describe a warp ``mat44d`` array as a 16-lane DLTensor for ``omni:xform`` writes.

    The array is consumed zero-copy through DLPack: a warp ``mat44d`` exports as ``(N, 4, 4)``
    ``lanes=1``, and ovstage folds the trailing matrix axes into the ``lanes=16`` the column
    expects. A device array therefore reaches ovstage without a host round-trip.

    The caller owns the data: the returned tensor must stay alive until the consuming write
    completes, and that write must be ordered against the kernels that produced
    :paramref:`xforms` — pass their Warp stream as ``write_attribute(cuda_stream=...)``.

    Args:
        xforms: Warp array of shape ``[N]`` and dtype :class:`warp.mat44d`.

    Returns:
        A :class:`ovstage.DLTensor` with shape ``[N]`` and ``lanes=16``.
    """
    return ovstage.make_dltensor(xforms, dtype=OVSTAGE_XFORM_DTYPE)


def points_tensor_from_warp(points: wp.array) -> ovstage.DLTensor:
    """Describe a warp ``vec3f`` array as a 3-lane DLTensor for ``points`` writes.

    The array is consumed zero-copy through DLPack: a warp ``vec3f`` exports as ``(N, 3)``
    ``lanes=1``, and ovstage folds the trailing component axis into the ``lanes=3`` the
    ``point3f[]`` column expects. A device array therefore reaches ovstage without a host
    round-trip.

    The caller owns the data: the returned tensor must stay alive until the consuming write
    completes, and that write must be ordered against the kernels that produced
    :paramref:`points` — pass their Warp stream as ``write_attribute(cuda_stream=...)``.

    Args:
        points: Warp array of shape ``[N]`` and dtype :class:`warp.vec3f`.

    Returns:
        A :class:`ovstage.DLTensor` with shape ``[N]`` and ``lanes=3``.
    """
    return ovstage.make_dltensor(points, dtype=OVSTAGE_POINT_DTYPE)


@configclass
class OvstageBackendCfg(BackendCfg):
    """Configuration of a simulation-owned stage and its populated domains."""

    class_type: type[OvstageBackend] | str = "{DIR}.stage:OvstageBackend"
    scene_key: BackendCfg | VisualizerCfg | None = None
    """None selects the simulation scene; isolated consumers use their configuration as the key."""
    population_domains: ovstage.PopulationDomain = ovstage.PopulationDomain.RENDERING
    """USD domains to populate. Physics consumers require PHYSICS or ALL."""


class OvstageBackend:
    """Own a detached stage and its paths until the simulation has closed its consumers."""

    def __init__(self, cfg: OvstageBackendCfg):
        """Create the stage identified by its consumer configuration and population domains."""
        self.cfg = cfg
        self.ordinal = 0
        self.next_camera_id = 0
        self.clone_copies: list[tuple[str, list[str]]] = []
        self.clone_env_paths: list[str] = []
        self.clone_positions: np.ndarray | None = None
        with contextlib.ExitStack() as resources:
            self.stage = resources.enter_context(create_ovstage("isaaclab.scene"))
            self.paths = resources.enter_context(ovstage.PathDictionary(self.stage))
            self._resources = resources.pop_all()

    def populate(self, usda: str) -> None:
        """Import the exported prototypes and apply the clone context's prepared operations.

        Population commits ordinal 1. Consumers may then author render products and poses
        starting at ordinal 2, after every cloned camera path exists.

        Args:
            usda: USD scene containing the routed prototypes and their materials.
        """
        from isaaclab_ov.cloner import ovstage_replicate  # noqa: PLC0415

        if self.ordinal:
            return

        # Ordinal 0 is the empty state; population and cloning form the first committed write.
        ovstage.population.open_usd_from_string(self.stage, usda, ordinal=1, domains=self.cfg.population_domains)
        ovstage_replicate(
            self.stage, self.paths, self.clone_copies, self.clone_env_paths, self.clone_positions, ordinal=1
        )
        self.ordinal = 1
        self.commit()

    def commit(self) -> int:
        """Seal pending writes and return their ordinal; all consumers share the next write ordinal."""
        ordinal = self.ordinal
        self.stage.advance_write_floor(ordinal=ordinal).wait()
        self.ordinal += 1
        return ordinal

    def close(self) -> None:
        """Release the path dictionary and stage after their borrowers have closed."""
        self._resources.close()
        self.stage = self.paths = None


def _collect_prims_to_deactivate(parent_prim: Usd.Prim, source_paths: frozenset[Sdf.Path]) -> list[Sdf.Path]:
    """Collect child prims under ``parent_prim`` for deactivation.

    For each child:

    * If the child is a source, keep the full subtree and stop descending.
    * If the child is an ancestor of some source, recurse to deactivate non-source siblings deeper in the tree.
    * Otherwise, deactivate the child prim (including descendants).

    Args:
        parent_prim: Parent prim whose children are considered.
        source_paths: The paths to the cloning sources.

    Returns:
        Paths of prims to deactivate on the root layer.
    """
    prim_paths: list[Sdf.Path] = []

    for child in parent_prim.GetChildren():
        child_path = child.GetPath()

        # If the child is a source, keep it and stop walking down the tree.
        if child_path in source_paths:
            continue

        # If the child is an ancestor of some source, recurse to deactivate non-source siblings deeper in the tree.
        if any(source.HasPrefix(child_path) for source in source_paths):
            prim_paths.extend(_collect_prims_to_deactivate(child, source_paths))
            continue

        # Otherwise, deactivate the child prim (including descendants).
        if child.IsActive():
            prim_paths.append(child_path)

    return prim_paths


def export_stage_to_string(
    stage: Usd.Stage, num_envs: int, source_paths: tuple[str, ...], keep_env_roots: bool = True
) -> str:
    """Export routed prototypes without mutating the simulation's authored USD stage.

    An anonymous session layer hides non-source subtrees. Physics population needs typed
    environment ancestors, so keep environment roots when cloning their children; remove
    the roots only when they are themselves clone targets.

    Args:
        stage: USD stage to export.
        num_envs: Number of parallel environments. A single environment is exported unchanged.
        source_paths: Prototype subtrees retained with their materials and descendants.
        keep_env_roots: Whether to retain non-source environment roots and their transforms.

    Returns:
        USDA text containing the selected prototypes.
    """
    if num_envs <= 1:
        return stage.ExportToString()

    export_session = Sdf.Layer.CreateAnonymous()
    export_session.subLayerPaths = [stage.GetSessionLayer().identifier]
    export_stage = Usd.Stage.Open(stage.GetRootLayer(), export_session)
    envs_path = Sdf.Path("/World/envs")
    envs_prim = export_stage.GetPrimAtPath(envs_path)
    if not envs_prim.IsValid():
        raise RuntimeError(f"Failed to get prim at path: {envs_path}")

    source_path_set = frozenset(map(Sdf.Path, source_paths))
    prim_paths: list[Sdf.Path] = []

    if keep_env_roots:
        for child in envs_prim.GetChildren():
            # Retain authored ancestor types and transforms for asset-level clones.
            child_path = child.GetPath()
            if child_path not in source_path_set:
                prim_paths.extend(_collect_prims_to_deactivate(child, source_path_set))
    else:
        # Whole-environment copies recreate these roots.
        prim_paths = _collect_prims_to_deactivate(envs_prim, source_path_set)

    with Sdf.ChangeBlock():
        for prim_path in prim_paths:
            Sdf.CreatePrimInLayer(export_session, prim_path).active = False
            logger.debug("Deactivated prim: %s", prim_path)
    logger.info("Deactivated %d prims in total", len(prim_paths))
    return export_stage.ExportToString()
