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


def _parent_path(prim_path: str) -> str:
    """Parent prim path of ``prim_path``.

    Args:
        prim_path: Absolute prim path, so it always contains a separator.

    Raises:
        RuntimeError: If the prim is directly under the stage root and thus has no non-pseudoroot
            parent to read Fabric matrices from.
    """
    parent = prim_path.rsplit("/", 1)[0]
    if not parent:
        raise RuntimeError(
            f"Child prim '{prim_path}' is at the stage root and has no parent prim. "
            "A Fabric xform selection requires every prim to have a non-pseudoroot parent "
            "with Fabric world+local matrices."
        )
    return parent


class FabricXformSelection:
    """Tagged Fabric selections over a set of prims and their parents, shared by the backend views.

    Three selections -- child read-only, child read-write, and parent world -- live for the
    instance's lifetime, keyed on a per-instance index attribute so their size follows the view
    rather than the stage. Accessors rebuild the view-index to Fabric-slot mapping from live Fabric
    data, so a bucket reorder can never leave it stale.
    """

    _WORLD_MATRIX_NAME = "omni:fabric:worldMatrix"
    _LOCAL_MATRIX_NAME = "omni:fabric:localMatrix"

    # Uid source for the attribute names; monotonic (not ``id(self)``) so a dead instance's leftover
    # attributes can never satisfy a live one's selection.
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
            seed_from_usd: Whether to seed both matrices on every prim from USD, for an owner whose
                source of truth is Fabric. ``False`` seeds only absent attributes and gives parents
                no local matrix, so a transform another system drives is neither reset nor
                overwritten by the hierarchy pass.
            skip_missing_prims: Whether prims absent from the Fabric stage are dropped instead of
                raising, as reported by :attr:`kept_indices`.

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
        self.read_write = False
        self._tagged_prims: list[tuple[str, list]] = []
        self.sel_ro = None
        self.sel_rw = None
        self.sel_parent = None

        self.stage = usdrt.Usd.Stage.Attach(get_current_stage_id())
        fabric_id = self.stage.GetFabricId()
        self.fabric_hierarchy = (
            hierarchy.IFabricHierarchy().get_fabric_hierarchy(fabric_id, self.stage.GetStageIdAsStageId())
            if hierarchy is not None
            else None
        )

        uid = next(FabricXformSelection._uid_counter)
        self.child_index_attr = f"isaaclab:fabricXform:{uid}:index"
        self._parent_index_attr = f"isaaclab:fabricXform:{uid}:parentIndex"

        if skip_missing_prims:
            self.kept_indices = [i for i, path in enumerate(prim_paths) if self.stage.GetPrimAtPath(path).IsValid()]
            self._prim_paths = [prim_paths[i] for i in self.kept_indices]
        else:
            self.kept_indices = list(range(len(prim_paths)))
            self._prim_paths = list(prim_paths)
        self.count = len(self._prim_paths)
        if self.count == 0:
            self.unique_parent_paths = []
            return

        child_parent_paths = [_parent_path(path) for path in self._prim_paths]
        self.unique_parent_paths = list(dict.fromkeys(child_parent_paths))
        parent_ordinal = {path: i for i, path in enumerate(self.unique_parent_paths)}

        # Tag children and parents with their per-instance ordinal, and make sure the matrices the
        # accessors below hand out actually exist. The index attribute doubles as the selection
        # filter, so a prim that is both a child and a parent here receives both attributes.
        for paths, index_attr, is_child in (
            (self._prim_paths, self.child_index_attr, True),
            (self.unique_parent_paths, self._parent_index_attr, False),
        ):
            group_prims: list = []
            for i, path in enumerate(paths):
                rt_prim = self.stage.GetPrimAtPath(path)
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
        child_tag = (uint_type, self.child_index_attr, read)
        parent_tag = (uint_type, self._parent_index_attr, read)
        world_ro = (matrix, self._WORLD_MATRIX_NAME, read)
        local_ro = (matrix, self._LOCAL_MATRIX_NAME, read)
        world_rw = (matrix, self._WORLD_MATRIX_NAME, read_write)
        local_rw = (matrix, self._LOCAL_MATRIX_NAME, read_write)
        self.sel_ro = self.stage.SelectPrims(require_attrs=[child_tag, world_ro, local_ro], device=device)
        self.sel_rw = self.stage.SelectPrims(require_attrs=[child_tag, world_rw, local_rw], device=device)
        self.sel_parent = self.stage.SelectPrims(require_attrs=[parent_tag, world_ro], device=device)

        # ``_child_parent_map`` holds view-side indices (uint32, like the Fabric ``UInt`` index
        # attributes); the slot buffers hold Fabric slots and must be int32, the only dtype
        # ``wp.indexedfabricarray`` accepts for indices.
        self.view_indices = wp.array(list(range(self.count)), dtype=wp.uint32, device=device)
        self.parent_view_indices = wp.array(list(range(len(self.unique_parent_paths))), dtype=wp.uint32, device=device)
        self._child_parent_map = wp.array(
            [parent_ordinal[path] for path in child_parent_paths], dtype=wp.uint32, device=device
        )
        self._child_slots_buf = wp.empty((self.count,), dtype=wp.int32, device=device)
        self._parent_slots_buf = wp.empty((len(self.unique_parent_paths),), dtype=wp.int32, device=device)
        self._parent_slot_of_child_buf = wp.empty((self.count,), dtype=wp.int32, device=device)

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
            fa=wp.fabricarray(self.sel_parent, self._WORLD_MATRIX_NAME), indices=self._parent_slot_of_child_buf
        )

    def parent_world_rw_ifa(self) -> wp.indexedfabricarray:
        """Return writable parent world matrices, via a one-off selection so :attr:`sel_parent` stays read-only."""
        import usdrt

        selection = self.stage.SelectPrims(
            require_attrs=[
                (usdrt.Sdf.ValueTypeNames.UInt, self._parent_index_attr, usdrt.Usd.Access.Read),
                (usdrt.Sdf.ValueTypeNames.Matrix4d, self._WORLD_MATRIX_NAME, usdrt.Usd.Access.ReadWrite),
            ],
            device=self._device,
        )
        num_parents = len(self.unique_parent_paths)
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
        selection = self.sel_rw if self.read_write else self.sel_ro
        selection.PrepareForReuse()
        self.check_count(selection.GetCount(), self.count, self.child_index_attr)
        wp.launch(
            kernel=fabric_utils.map_view_indices_to_fabric_slots,
            dim=self.count,
            inputs=[wp.fabricarray(selection, self.child_index_attr), self._child_slots_buf],
            device=self._device,
        )
        return selection

    def refresh_parent_selection(self) -> None:
        """Refresh the parent selection and rebuild the per-child parent-slot mapping."""
        num_parents = self._parent_slots_buf.shape[0]
        self.sel_parent.PrepareForReuse()
        self.check_count(self.sel_parent.GetCount(), num_parents, self._parent_index_attr)
        wp.launch(
            kernel=fabric_utils.map_view_indices_to_fabric_slots,
            dim=num_parents,
            inputs=[wp.fabricarray(self.sel_parent, self._parent_index_attr), self._parent_slots_buf],
            device=self._device,
        )
        wp.launch(
            kernel=fabric_utils.gather_fabric_slots,
            dim=self.count,
            inputs=[self._parent_slots_buf, self._child_parent_map, self._parent_slot_of_child_buf],
            device=self._device,
        )
