# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fabric stage and hierarchy cache, registered as a service on SimulationContext."""

from __future__ import annotations

from pxr import UsdUtils


class FabricStageCache:
    """Caches the usdrt stage attachment and IFabricHierarchy handles.

    Registered as a singleton service on :class:`~isaaclab.sim.SimulationContext` via
    ``set_service(FabricStageCache, ...)``.  Multiple
    :class:`~isaaclab_physx.sim.views.FabricFrameView` instances share a single
    hierarchy handle per Fabric attachment.

    The hierarchy cache is keyed by ``fabric_id_int`` (the stable ``.id`` integer from
    ``FabricId``).  Currently Isaac Lab always has exactly one Fabric attachment per
    stage, so this dict will hold at most one entry.  A dict is used rather than a plain
    attribute so the design naturally extends to multi-Fabric scenarios (e.g. multi-GPU
    support, where each GPU gets its own Fabric attachment) without an API change.
    """

    def __init__(self, usd_stage) -> None:
        import usdrt  # noqa: PLC0415

        stage_id = UsdUtils.StageCache.Get().GetId(usd_stage).ToLongInt()
        self._stage = usdrt.Usd.Stage.Attach(stage_id)
        self._stage.SynchronizeToFabric()
        self._hierarchy_cache: dict[int, object] = {}

    @property
    def stage(self):
        """The usdrt stage (already attached and synchronized)."""
        return self._stage

    def close(self) -> None:
        """Release cached handles.  Called by SimulationContext on teardown."""
        self._hierarchy_cache.clear()
        self._stage = None

    def get_hierarchy(self):
        """Return the IFabricHierarchy handle for the current Fabric attachment.

        Creates and caches the handle on first call.  Change-tracking is enabled
        for both local and world xforms.

        Returns:
            A tuple of ``(hierarchy_handle, fabric_id_int)``.
        """
        import usdrt  # noqa: PLC0415

        fabric_id = self._stage.GetFabricId()
        fabric_id_int = fabric_id.id

        if fabric_id_int not in self._hierarchy_cache:
            hierarchy = usdrt.hierarchy.IFabricHierarchy().get_fabric_hierarchy(
                fabric_id, self._stage.GetStageIdAsStageId()
            )
            hierarchy.track_local_xform_changes(True)
            hierarchy.track_world_xform_changes(True)
            self._hierarchy_cache[fabric_id_int] = hierarchy

        return self._hierarchy_cache[fabric_id_int], fabric_id_int
