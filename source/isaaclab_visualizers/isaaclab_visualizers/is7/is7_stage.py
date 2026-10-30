# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Sim 7 stage bridge: borrow the viewer's OVStage and publish Isaac Lab body poses into it.

Isaac Sim 7 owns the OVStage that its renderers read. Isaac Lab owns physics and the world poses
it computes each step. The bridge joins them without a physics participant inside Isaac Sim:

1. The scene is exported from the authored USD stage into an Isaac Sim 7 ``SimulationContext``.
2. :func:`adopt_ovstage` wraps the context's native stage pointer in an :class:`ovstage.Stage`.
3. :meth:`IS7StageBridge.publish` writes world matrices from the scene data provider straight
   into ``omni:xform`` through DLPack, which is the same zero-copy path the OVRTX renderer uses.

Everything that depends on Isaac Sim 7 internals sits in this module so a library bump only
changes the adoption and write-ordinal helpers below.
"""

from __future__ import annotations

import contextlib
import ctypes
import logging
import math
from typing import TYPE_CHECKING

import numpy as np
import ovstage
import warp as wp

from pxr import Gf, Usd, UsdGeom

from isaaclab.scene_data import SceneDataFormat

if TYPE_CHECKING:
    from isaaclab.scene_data import SceneDataProvider

logger = logging.getLogger(__name__)

_XFORM_DTYPE = ovstage.DLDataType(code=ovstage.DLDataTypeCode.kDLFloat, bits=64, lanes=16)
"""ovstage stores ``omni:xform`` as one 16-lane float64 element per prim."""


def adopt_ovstage(address: int) -> tuple[ovstage.Stage, contextlib.AbstractContextManager[None]]:
    """Wrap a stage owned by another library in an :class:`ovstage.Stage`.

    Isaac Sim 7 publishes its stage as an integer address (``SimulationContext.stage_pointer``).
    The Python wrapper only needs the instance pointer and the process-wide runtime handles, so a
    throwaway stage supplies the handles and the borrowed pointer replaces its own.

    Args:
        address: Address of a live ``ovstage_instance_t``.

    Returns:
        The borrowed stage and a context manager that detaches it. Detach before the owner closes
        the stage so the wrapper's finalizer never destroys memory it does not own.
    """
    probe = ovstage.Stage("isaaclab.is7.probe")
    borrowed = object.__new__(ovstage.Stage)
    borrowed.__dict__.update(probe.__dict__)
    pointer_type = type(probe._inst)
    borrowed._inst = ctypes.cast(ctypes.c_void_p(address), pointer_type)
    borrowed._holds_process_ref = False

    @contextlib.contextmanager
    def detach():
        try:
            yield
        finally:
            borrowed._inst = pointer_type()
            probe.destroy()

    return borrowed, detach()


class IS7StageBridge:
    """Publish simulation poses into an Isaac Sim 7 stage that Isaac Lab does not own."""

    def __init__(self, native_address: int, usd_stage: Usd.Stage, provider: SceneDataProvider, device: str):
        """Bind body poses to the viewer stage.

        Args:
            native_address: ``SimulationContext.stage_pointer`` of a finalized context.
            usd_stage: Authored Isaac Lab stage, read for the bodies' authored scales.
            provider: Scene data provider that publishes simulation world transforms.
            device: Warp device that holds the published transforms.
        """
        self._provider = provider
        self._device = wp.get_device(device)
        self._resources = contextlib.ExitStack()
        self.stage, detach = adopt_ovstage(native_address)
        self._resources.enter_context(detach)

        paths = list(provider.backend.transform_paths)
        self.body_paths = paths
        self._scales = self._authored_scales(usd_stage, paths)
        self._pose_output = SceneDataFormat.TransposedMatrix44d()
        self._timestamp = -1
        self._bind(paths)

    @staticmethod
    def _authored_scales(usd_stage: Usd.Stage, paths: list[str]) -> wp.array | None:
        """Return per-body authored scales, or ``None`` when every body is unit scale.

        Simulation poses omit scale, and the viewer renders the body with an ``omni:resetXformStack``
        local matrix, so authored scale has to be folded back in.
        """
        cache = UsdGeom.XformCache()
        scales = []
        for path in paths:
            prim = usd_stage.GetPrimAtPath(path)
            if prim and prim.IsA(UsdGeom.Xformable):
                scales.append(tuple(map(float, Gf.Transform(cache.GetLocalToWorldTransform(prim)).GetScale())))
            else:
                scales.append((1.0, 1.0, 1.0))
        if all(math.isclose(axis, 1.0, rel_tol=1e-6, abs_tol=1e-6) for scale in scales for axis in scale):
            return None
        return wp.array(scales, dtype=wp.vec3f)

    def _bind(self, paths: list[str]) -> None:
        """Create the reusable path query and mark bodies as world-space so poses need no composition."""
        path_dictionary = self._resources.enter_context(ovstage.PathDictionary(self.stage))
        path_list = path_dictionary.create_path_list_from_strings(paths)
        self._resources.callback(path_dictionary.destroy_path_list, path_list)
        self._query = self._resources.enter_context(self.stage.query_from_path_list(path_list))
        ordinal = self._next_ordinal()
        self.stage.write_attribute(
            self._query,
            "omni:resetXformStack",
            ordinal=ordinal,
            tensors=np.ones(len(paths), dtype=np.bool_),
            is_array=False,
        ).wait()
        self.stage.advance_write_floor(ordinal=ordinal).wait()

    def _next_ordinal(self) -> int:
        """Return the first ordinal that is still writable.

        The Isaac Sim 7 renderer seals ordinals when it publishes a frame, so the stage's current
        write floor is read instead of counting locally.
        """
        query = self.stage.get_attribute_write_floor()
        try:
            query.wait()
            return int(query.fetch()) + 1
        finally:
            query.release()

    def publish(self) -> bool:
        """Write the latest simulation world transforms into the stage.

        Returns:
            ``True`` when new transforms were written, ``False`` when nothing changed.
        """
        timestamp = self._provider.backend.transforms_timestamp
        if timestamp == self._timestamp:
            return False
        if not self._provider.get_transforms(self._pose_output, allow_passthrough=True, scales=self._scales):
            return False
        matrices = self._pose_output.matrices
        ordinal = self._next_ordinal()
        stream = self._device.stream
        self.stage.write_attribute(
            self._query,
            "omni:xform",
            ordinal=ordinal,
            tensors=ovstage.make_dltensor(matrices, dtype=_XFORM_DTYPE),
            is_array=False,
            semantic=ovstage.AttributeSemantic.MATRIX,
            cuda_stream=stream.cuda_stream or 1,
        ).wait()
        self.stage.advance_write_floor(ordinal=ordinal).wait()
        self._timestamp = timestamp
        return True

    def close(self) -> None:
        """Release the query and path list, then detach from the borrowed stage."""
        self._resources.close()
