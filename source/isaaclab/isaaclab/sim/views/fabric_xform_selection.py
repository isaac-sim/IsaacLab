# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fabric prim tagging, selections, and view-to-slot mapping shared by the backend frame views."""

from __future__ import annotations

import itertools
import logging

import warp as wp

from isaaclab.utils.warp import fabric as fabric_utils

logger = logging.getLogger(__name__)


def parent_path(prim_path: str) -> str:
    """Return the parent prim path, or ``"/"`` for a root-level prim."""
    parent = prim_path.rsplit("/", 1)[0]
    return parent if parent else "/"


class FabricXformSelection:
    """Tagged Fabric selections over a set of prims and their parents.

    Shared by :class:`~isaaclab_physx.sim.views.FabricFrameView`, which treats Fabric as its source of
    truth, and :class:`~isaaclab_newton.sim.views.NewtonSiteFrameView`, which only pushes Newton state
    onto it; :meth:`__init__` takes the flags that separate those two semantics.

    Selections are keyed on a per-instance index attribute, keeping their size proportional to the
    view rather than to the stage. Three live for the instance's lifetime: child read-only, child
    read-write (:attr:`read_write` selects between them), and parent world read-only. The view-index
    to Fabric-slot mapping is rebuilt from live Fabric data on every accessor call rather than cached,
    so a bucket reorder can never leave a stale mapping behind.
    """

    _WORLD_MATRIX_NAME = "omni:fabric:worldMatrix"
    _LOCAL_MATRIX_NAME = "omni:fabric:localMatrix"

    # Process-wide uid source for the per-instance attribute names. A monotonic counter (NOT
    # ``id(self)``) guarantees a name is never reused after an instance is garbage-collected, so a
    # dead instance's leftover attributes can never satisfy a live one's selection.
    _uid_counter = itertools.count()

    def __init__(
        self,
        prim_paths: list[str],
        device: str,
        *,
        owner: str,
        seed_from_usd: bool,
        skip_missing_prims: bool = False,
    ):
        """Tag the prims, build the selections, and allocate the mapping buffers.

        Args:
            prim_paths: Prim paths to select, in the owner's view order.
            device: Warp device for the index and slot buffers.
            owner: Owning class name, used in error messages and the attribute namespace.
            seed_from_usd: Whether to author and seed both matrices on every prim from USD. Use this
                when Fabric is the owner's source of truth and must start out populated. When
                ``False``, only genuinely absent attributes are created and seeded, and parents never
                receive a local matrix -- seeding a prim whose transform another system already
                drives would reset it, and authoring a local matrix on one lets the Fabric hierarchy
                pass overwrite that system's world matrix with ``parent * local``.
            skip_missing_prims: Whether prims absent from the Fabric stage are dropped instead of
                raising. Drops are reported through :attr:`kept_indices`.

        Raises:
            RuntimeError: If a required prim is missing from the Fabric stage.
        """
        import usdrt
        from usdrt import Rt

        try:
            from usdrt import hierarchy
        except ImportError:
            hierarchy = None

        from isaaclab.sim.utils import get_current_stage_id

        self._device = device
        self._owner = owner
        self._read_write = False
        self._tagged_prims: list[tuple[str, list]] = []
        self._sel_ro = None
        self._sel_rw = None
        self._sel_parent = None

        self._stage = usdrt.Usd.Stage.Attach(get_current_stage_id())
        fabric_id = self._stage.GetFabricId()
        self._fabric_hierarchy = (
            hierarchy.IFabricHierarchy().get_fabric_hierarchy(fabric_id, self._stage.GetStageIdAsStageId())
            if hierarchy is not None
            else None
        )

        uid = next(FabricXformSelection._uid_counter)
        self._child_index_attr = f"isaaclab:fabricXform:{uid}:index"
        self._parent_index_attr = f"isaaclab:fabricXform:{uid}:parentIndex"

        if skip_missing_prims:
            self._kept_indices = [i for i, path in enumerate(prim_paths) if self._stage.GetPrimAtPath(path).IsValid()]
            self._prim_paths = [prim_paths[i] for i in self._kept_indices]
        else:
            self._kept_indices = list(range(len(prim_paths)))
            self._prim_paths = list(prim_paths)
        self._count = len(self._prim_paths)
        if self._count == 0:
            self._unique_parent_paths = []
            return

        child_parent_paths = [parent_path(path) for path in self._prim_paths]
        self._unique_parent_paths = list(dict.fromkeys(child_parent_paths))
        parent_ordinal = {path: i for i, path in enumerate(self._unique_parent_paths)}

        # Tag children and parents with their per-instance ordinal, and make sure the matrices the
        # accessors below hand out actually exist. The index attribute doubles as the selection
        # filter, so a prim that is both a child and a parent here receives both attributes.
        for paths, index_attr, is_child in (
            (self._prim_paths, self._child_index_attr, True),
            (self._unique_parent_paths, self._parent_index_attr, False),
        ):
            group_prims: list = []
            for i, path in enumerate(paths):
                rt_prim = self._stage.GetPrimAtPath(path)
                if not rt_prim.IsValid():
                    raise RuntimeError(f"{owner}: prim '{path}' does not exist in the Fabric stage.")
                rt_xformable = Rt.Xformable(rt_prim)
                if seed_from_usd:
                    rt_xformable.CreateFabricHierarchyWorldMatrixAttr()
                    rt_xformable.CreateFabricHierarchyLocalMatrixAttr()
                    rt_xformable.SetLocalXformFromUsd()
                    rt_xformable.SetWorldXformFromUsd()
                else:
                    if not rt_prim.HasAttribute(self._WORLD_MATRIX_NAME):
                        rt_xformable.CreateFabricHierarchyWorldMatrixAttr()
                        rt_xformable.SetWorldXformFromUsd()
                    if is_child and not rt_prim.HasAttribute(self._LOCAL_MATRIX_NAME):
                        rt_xformable.CreateFabricHierarchyLocalMatrixAttr()
                        rt_xformable.SetLocalXformFromUsd()
                rt_prim.CreateAttribute(index_attr, usdrt.Sdf.ValueTypeNames.UInt, custom=True)
                rt_prim.GetAttribute(index_attr).Set(i)
                group_prims.append(rt_prim)
            self._tagged_prims.append((index_attr, group_prims))

        matrix = usdrt.Sdf.ValueTypeNames.Matrix4d
        uint_type = usdrt.Sdf.ValueTypeNames.UInt
        read = usdrt.Usd.Access.Read
        read_write = usdrt.Usd.Access.ReadWrite
        child_tag = (uint_type, self._child_index_attr, read)
        parent_tag = (uint_type, self._parent_index_attr, read)
        world_ro = (matrix, self._WORLD_MATRIX_NAME, read)
        local_ro = (matrix, self._LOCAL_MATRIX_NAME, read)
        world_rw = (matrix, self._WORLD_MATRIX_NAME, read_write)
        local_rw = (matrix, self._LOCAL_MATRIX_NAME, read_write)
        self._sel_ro = self._stage.SelectPrims(require_attrs=[child_tag, world_ro, local_ro], device=device)
        self._sel_rw = self._stage.SelectPrims(require_attrs=[child_tag, world_rw, local_rw], device=device)
        self._sel_parent = self._stage.SelectPrims(require_attrs=[parent_tag, world_ro], device=device)

        # ``_child_parent_map`` holds view-side indices (uint32, like the Fabric ``UInt`` index
        # attributes); the slot buffers hold Fabric slots and must be int32, the only dtype
        # ``wp.indexedfabricarray`` accepts for indices.
        self._view_indices = wp.array(list(range(self._count)), dtype=wp.uint32, device=device)
        self._parent_view_indices = wp.array(
            list(range(len(self._unique_parent_paths))), dtype=wp.uint32, device=device
        )
        self._child_parent_map = wp.array(
            [parent_ordinal[path] for path in child_parent_paths], dtype=wp.uint32, device=device
        )
        self._child_slots_buf = wp.empty((self._count,), dtype=wp.int32, device=device)
        self._parent_slots_buf = wp.empty((len(self._unique_parent_paths),), dtype=wp.int32, device=device)
        self._parent_slot_of_child_buf = wp.empty((self._count,), dtype=wp.int32, device=device)

    """
    Properties.
    """

    @property
    def count(self) -> int:
        """Number of selected prims."""
        return self._count

    @property
    def kept_indices(self) -> list[int]:
        """Index into the caller's original path list for each selected prim."""
        return self._kept_indices

    @property
    def unique_parent_paths(self) -> list[str]:
        """Parent prim paths, deduplicated in first-occurrence order."""
        return self._unique_parent_paths

    @property
    def stage(self):
        """The attached usdrt stage."""
        return self._stage

    @property
    def fabric_hierarchy(self):
        """Fabric hierarchy handle, or ``None`` when the bindings are unavailable."""
        return self._fabric_hierarchy

    @property
    def child_index_attr(self) -> str:
        """Per-instance index attribute authored on the selected prims."""
        return self._child_index_attr

    @property
    def view_indices(self) -> wp.array:
        """Dense ``uint32`` view indices, ``0..count-1``."""
        return self._view_indices

    @property
    def parent_view_indices(self) -> wp.array:
        """Dense ``uint32`` parent indices."""
        return self._parent_view_indices

    @property
    def sel_ro(self):
        """Persistent read-only child selection."""
        return self._sel_ro

    @property
    def sel_rw(self):
        """Persistent read-write child selection."""
        return self._sel_rw

    @property
    def sel_parent(self):
        """Persistent read-only parent world selection."""
        return self._sel_parent

    @property
    def read_write(self) -> bool:
        """Whether the child accessors resolve to the read-write selection."""
        return self._read_write

    @read_write.setter
    def read_write(self, value: bool) -> None:
        self._read_write = bool(value)

    """
    Operations.
    """

    def world_ifa(self) -> wp.indexedfabricarray:
        """Return the selected prims' world matrices in view order."""
        selection = self.refresh_child_selection()
        return wp.indexedfabricarray(
            fa=wp.fabricarray(selection, self._WORLD_MATRIX_NAME), indices=self._child_slots_buf
        )

    def local_ifa(self) -> wp.indexedfabricarray:
        """Return the selected prims' local matrices in view order."""
        selection = self.refresh_child_selection()
        return wp.indexedfabricarray(
            fa=wp.fabricarray(selection, self._LOCAL_MATRIX_NAME), indices=self._child_slots_buf
        )

    def child_ifas(self) -> tuple[wp.indexedfabricarray, wp.indexedfabricarray]:
        """Return ``(world, local)`` from one refresh; cheaper than calling both single accessors."""
        selection = self.refresh_child_selection()
        world = wp.fabricarray(selection, self._WORLD_MATRIX_NAME)
        local = wp.fabricarray(selection, self._LOCAL_MATRIX_NAME)
        return (
            wp.indexedfabricarray(fa=world, indices=self._child_slots_buf),
            wp.indexedfabricarray(fa=local, indices=self._child_slots_buf),
        )

    def parent_world_ifa(self) -> wp.indexedfabricarray:
        """Return each selected prim's parent world matrix, in child view order."""
        self.refresh_parent_selection()
        return wp.indexedfabricarray(
            fa=wp.fabricarray(self._sel_parent, self._WORLD_MATRIX_NAME), indices=self._parent_slot_of_child_buf
        )

    def parent_world_rw_ifa(self) -> wp.indexedfabricarray:
        """Return writable parent world matrices, via a one-off selection so :attr:`sel_parent` stays read-only."""
        import usdrt

        selection = self._stage.SelectPrims(
            require_attrs=[
                (usdrt.Sdf.ValueTypeNames.UInt, self._parent_index_attr, usdrt.Usd.Access.Read),
                (usdrt.Sdf.ValueTypeNames.Matrix4d, self._WORLD_MATRIX_NAME, usdrt.Usd.Access.ReadWrite),
            ],
            device=self._device,
        )
        num_parents = len(self._unique_parent_paths)
        self.check_count(selection.GetCount(), num_parents, self._parent_index_attr)
        wp.launch(
            kernel=fabric_utils.map_view_indices_to_fabric_slots,
            dim=num_parents,
            inputs=[wp.fabricarray(selection, self._parent_index_attr), self._parent_slots_buf],
            device=self._device,
        )
        return wp.indexedfabricarray(
            fa=wp.fabricarray(selection, self._WORLD_MATRIX_NAME), indices=self._parent_slots_buf
        )

    def check_count(self, found: int, expected: int, index_attr: str) -> None:
        """Raise if a selection stopped matching exactly the prims this instance tagged."""
        if found != expected:
            raise RuntimeError(
                f"{self._owner}: selection on '{index_attr}' matched {found} prims, expected {expected}. "
                "A prim managed by this view (or one of its Fabric matrix/index attributes) was removed "
                "from the Fabric stage; recreate the view."
            )

    def close(self) -> None:
        """Remove the index attributes authored by this instance. Safe to call more than once."""
        failed = total = 0
        for attr, prims in self._tagged_prims:
            total += len(prims)
            for prim in prims:
                try:
                    prim.RemoveProperty(attr)
                except Exception:  # noqa: BLE001 -- one bad handle must not strand the remaining tags
                    failed += 1
        self._tagged_prims = []
        if failed:
            logger.debug("%s: %d of %d tag removals failed", self._owner, failed, total)

    def refresh_child_selection(self):
        """Refresh the active child selection and rebuild its slot mapping on device.

        ``PrepareForReuse`` absorbs Fabric bucket changes, and notifies the renderer for the
        read-write selection -- which is why writers must select it, not just for access rights.
        """
        selection = self._sel_rw if self._read_write else self._sel_ro
        selection.PrepareForReuse()
        self.check_count(selection.GetCount(), self._count, self._child_index_attr)
        wp.launch(
            kernel=fabric_utils.map_view_indices_to_fabric_slots,
            dim=self._count,
            inputs=[wp.fabricarray(selection, self._child_index_attr), self._child_slots_buf],
            device=self._device,
        )
        return selection

    def refresh_parent_selection(self) -> None:
        """Refresh the parent selection and rebuild the per-child parent-slot mapping."""
        num_parents = self._parent_slots_buf.shape[0]
        self._sel_parent.PrepareForReuse()
        self.check_count(self._sel_parent.GetCount(), num_parents, self._parent_index_attr)
        wp.launch(
            kernel=fabric_utils.map_view_indices_to_fabric_slots,
            dim=num_parents,
            inputs=[wp.fabricarray(self._sel_parent, self._parent_index_attr), self._parent_slots_buf],
            device=self._device,
        )
        wp.launch(
            kernel=fabric_utils.gather_fabric_slots,
            dim=self._count,
            inputs=[self._parent_slots_buf, self._child_parent_map, self._parent_slot_of_child_buf],
            device=self._device,
        )
