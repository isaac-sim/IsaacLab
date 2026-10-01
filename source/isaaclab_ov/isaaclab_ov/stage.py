# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for creating ovstage stages and describing their attribute columns.

``ovstage`` is a hard dependency of ``isaaclab_ov``, so it is imported unconditionally here.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import ovstage
import warp as wp

from isaaclab_ov.ovstage_compat import HIERARCHY_COMPUTATION_MODEL

if TYPE_CHECKING:
    from pxr import Usd

    from isaaclab.cloner import ClonePlan

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


def create_render_ovstage(stage: Usd.Stage, plan: ClonePlan) -> ovstage.Stage:
    """Build a populated ovstage of the scene for a consumer that draws it, such as Newton's ``ViewerRTX``.

    The USD stage is trimmed to the clone plan's prototypes, exported into a new ovstage stage, and
    cloned onto every environment, so each environment keeps its own authored prims and visual
    materials. The caller owns the returned stage.

    Args:
        stage: The live USD stage the clone plan was published for.
        plan: The scene's completed clone plan.

    Returns:
        The populated stage, with all writes committed.

    Raises:
        RuntimeError: If the installed OVStage does not compute the prim hierarchy on the device, which a
            stage borrowed by ``ViewerRTX`` requires.
    """
    if HIERARCHY_COMPUTATION_MODEL != "GPU_INCREMENTAL":
        raise RuntimeError(
            "Rendering the simulation's USD stage needs OVStage 0.2 or newer, which computes the prim hierarchy"
            " on the device."
        )

    from isaaclab.cloner import path as cloner_path  # noqa: PLC0415

    from isaaclab_ov.cloner import ovstage_replicate  # noqa: PLC0415
    from isaaclab_ov.renderers.ovrtx_usd import export_stage_to_string  # noqa: PLC0415

    num_envs = len(plan.topology.world_prototype_layout)
    sources = tuple(source for source in cloner_path.get_asset_prototype_paths(plan) if source is not None)
    usda = export_stage_to_string(stage, num_envs, source_paths=sources, keep_env_roots=False)

    render_stage = create_ovstage("isaaclab.render")
    # Ordinal 0 is the empty state in ovstage; the first write must use >= 1.
    ordinal = 1
    ovstage.population.open_usd_from_string(
        render_stage, usda, ordinal=ordinal, domains=ovstage.PopulationDomain.RENDERING
    )
    ovstage.population.apply_usd_changes(render_stage, ordinal=ordinal)
    with ovstage.PathDictionary(render_stage) as paths:
        ovstage_replicate(render_stage, paths, plan, ordinal)
    render_stage.advance_write_floor(ordinal=ordinal).wait()
    return render_stage
