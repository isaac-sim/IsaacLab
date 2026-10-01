# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulation-owned OVStage resources and shared attribute-column helpers.

``ovstage`` is a hard dependency of ``isaaclab_ov``, so it is imported unconditionally here.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import numpy as np
import ovstage
import warp as wp

from isaaclab.sim import BackendCfg
from isaaclab.utils import configclass

from isaaclab_ov.ovstage_compat import HIERARCHY_COMPUTATION_MODEL

if TYPE_CHECKING:
    from pxr import Usd

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
    """Configuration of a simulation-owned stage borrowed by a visualizer."""

    class_type: type[OvstageBackend] | str = "{DIR}.stage:OvstageBackend"
    visualizer_cfg: VisualizerCfg = MISSING
    """Consumer settings; different viewer cameras and render products need separate stages."""


class OvstageBackend:
    """Own a detached rendering stage until the simulation has closed its consumers."""

    def __init__(self, cfg: OvstageBackendCfg):
        """Create the native stage with the device hierarchy required by borrowed-stage rendering.

        Args:
            cfg: Consumer configuration identifying this resource.

        Raises:
            RuntimeError: If the installed OVStage does not support the device hierarchy.
        """
        if HIERARCHY_COMPUTATION_MODEL != "GPU_INCREMENTAL":
            raise RuntimeError("Rendering a borrowed stage needs OVStage 0.2 or newer for its device hierarchy.")
        self.stage = create_ovstage("isaaclab.render")

    def populate(
        self,
        stage: Usd.Stage,
        copies: Sequence[tuple[str, Sequence[str]]],
        env_paths: Sequence[str],
        positions: np.ndarray | None,
    ) -> None:
        """Import routed prototypes and execute the clone context's prepared native operations.

        Args:
            stage: Live USD stage containing the prototypes and their materials.
            copies: Source and destination paths with the contract of :func:`isaaclab_ov.cloner.ovstage_replicate`.
            env_paths: Environment-root paths in placement order.
            positions: Optional environment positions [m], shape ``[len(env_paths), 3]``.
        """
        from isaaclab_ov.cloner import ovstage_replicate  # noqa: PLC0415
        from isaaclab_ov.renderers.ovrtx_usd import export_stage_to_string  # noqa: PLC0415

        sources = tuple(source for source, _ in copies)
        usda = export_stage_to_string(stage, len(env_paths), source_paths=sources, keep_env_roots=False)
        # Ordinal 0 is the empty state; population and cloning form the first committed write.
        ovstage.population.open_usd_from_string(self.stage, usda, ordinal=1, domains=ovstage.PopulationDomain.RENDERING)
        ovstage.population.apply_usd_changes(self.stage, ordinal=1)
        with ovstage.PathDictionary(self.stage) as paths:
            ovstage_replicate(self.stage, paths, copies, env_paths, positions, ordinal=1)
        self.stage.advance_write_floor(ordinal=1).wait()

    def close(self) -> None:
        """Release the native stage after its borrowers have closed."""
        self.stage.destroy()
