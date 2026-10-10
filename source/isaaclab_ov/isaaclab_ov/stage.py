# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for creating ovstage stages and describing their attribute columns.

``ovstage`` is a hard dependency of ``isaaclab_ov``, so it is imported unconditionally here.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import numpy as np
import ovstage
import warp as wp

from pxr import Usd

from isaaclab.cloner import ClonePlan
from isaaclab.cloner.clone_plan import path as cloner_path
from isaaclab.sim import BackendCfg
from isaaclab.utils import configclass

from isaaclab_ov.ovstage_compat import HIERARCHY_COMPUTATION_MODEL
from isaaclab_ov.renderers.ovrtx_usd import export_stage_to_string

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


def _iter_clone_batches(plan: ClonePlan) -> Iterator[tuple[str, list[str]]]:
    """Yield native clone paths parent-first, omitting self-copies and children covered by their parent."""
    sources = cloner_path.get_asset_prototype_paths(plan)
    templates, starts, worlds, world_starts = cloner_path.get_world_prototype_asset_templates(
        plan, include_world_indices=True
    )
    copies = {}
    for group in np.flatnonzero(np.diff(world_starts)):
        start, end = starts[group : group + 2]
        targets = worlds[world_starts[group] : world_starts[group + 1]]
        references = [
            (sources[asset], templates[index])
            for index, asset in enumerate(plan.topology.world_prototypes[start:end], start)
        ]
        parents = cloner_path.get_parent_indices([target for _, target in references])
        for (source, target), parent in zip(references, parents, strict=True):
            if parent != -1:
                parent_source, parent_target = references[parent]
                if source == cloner_path.rebase(target, parent_target, parent_source):
                    continue
            copies.setdefault((source, target), []).append(targets)
    for source, template in sorted(copies, key=lambda copy: copy[1].count("/")):
        worlds = np.concatenate(copies[source, template])
        targets = [target for target in map(template.format, worlds) if target != source]
        if targets:
            yield source, targets


def ovstage_replicate(stage: ovstage.Stage, plan: ClonePlan, *, ordinal: int) -> None:
    """Clone the scene's prototypes and place environments on a native stage.

    Args:
        stage: Populated stage containing the authored prototypes.
        plan: Scene topology and environment positions [m].
        ordinal: Write ordinal for cloning and placement.
    """
    for source, targets in _iter_clone_batches(plan):
        stage.clone(source, targets, ordinal=ordinal)
    num_envs = len(plan.topology.world_prototype_layout)
    xforms = np.tile(np.eye(4, dtype=np.float64), (num_envs, 1, 1))
    xforms[:, 3, :3] = plan.positions
    with ovstage.PathDictionary(stage) as paths:
        env_paths = paths.create_path_list_from_strings([plan.env_template.format(i) for i in range(num_envs)])
        try:
            with stage.query_from_path_list(env_paths) as query:
                stage.write_attribute(
                    query,
                    "omni:xform",
                    ordinal=ordinal,
                    tensors=xform_tensor_from_numpy(xforms),
                    is_array=False,
                    semantic=ovstage.AttributeSemantic.MATRIX,
                ).wait()
        finally:
            paths.destroy_path_list(env_paths)


@configclass
class OvstageBackendCfg(BackendCfg):
    """Identify the simulation-owned rendering stage for one visualizer instance."""

    class_type: type[OvstageBackend] | str = "{DIR}.stage:OvstageBackend"
    viewer_id: int = 0


class OvstageBackend:
    """Own the native scene until the simulation has closed its borrowing viewers."""

    def __init__(self, cfg: OvstageBackendCfg):
        """Create an empty stage for the configured consumer."""
        self.stage = create_ovstage("isaaclab.viewer")
        self._populated = False

    def populate(self, stage: Usd.Stage, plan: ClonePlan | None) -> None:
        """Export and clone the authored scene before a viewer attaches to it."""
        if self._populated:
            return
        num_envs = len(plan.topology.world_prototype_layout) if plan is not None else 1
        sources = None
        if plan is not None:
            sources = tuple(source for source in cloner_path.get_asset_prototype_paths(plan) if source is not None)
        usda = export_stage_to_string(stage, num_envs, source_paths=sources, keep_env_roots=False)
        ovstage.population.open_usd_from_string(self.stage, usda, ordinal=1, domains=ovstage.PopulationDomain.RENDERING)
        if num_envs > 1:
            ovstage_replicate(self.stage, plan, ordinal=1)
        self.stage.advance_write_floor(1).wait()
        self._populated = True

    def close(self) -> None:
        """Release the stage after all viewers have detached."""
        if self.stage is not None:
            self.stage.destroy()
            self.stage = None
