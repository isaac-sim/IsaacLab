# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit/USD implementation for :class:`VisualizationMarkers`."""

from __future__ import annotations

import logging

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers.visualization_markers_cfg import VisualizationMarkersCfg
from isaaclab.utils.version import has_kit

logger = logging.getLogger(__name__)


class KitVisualizationMarkers:
    """USD PointInstancer backend for visualization markers."""

    def __init__(self, cfg: VisualizationMarkersCfg, visible: bool = True):
        self.cfg = cfg
        self.stage = sim_utils.get_current_stage()
        self.prim_path = sim_utils.get_next_free_prim_path(cfg.prim_path)
        self._is_visible = visible
        self._count = len(cfg.markers)

        from pxr import Gf, UsdGeom  # noqa: PLC0415

        self._instancer_manager = UsdGeom.PointInstancer.Define(self.stage, self.prim_path)
        self._add_markers_prototypes(self.cfg.markers)
        self._instancer_manager.GetProtoIndicesAttr().Set(list(range(len(self.cfg.markers))))
        self._instancer_manager.GetPositionsAttr().Set([Gf.Vec3f(0.0)] * len(self.cfg.markers))
        self.set_visibility(visible)

    @property
    def count(self) -> int:
        return self._count

    def set_visibility(self, visible: bool) -> None:
        """Set USD PointInstancer visibility."""
        from pxr import UsdGeom  # noqa: PLC0415

        self._is_visible = visible
        imageable = UsdGeom.Imageable(self._instancer_manager)
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()

    def is_visible(self) -> bool:
        """Return USD PointInstancer visibility."""
        from pxr import UsdGeom  # noqa: PLC0415

        return self._instancer_manager.GetVisibilityAttr().Get() != UsdGeom.Tokens.invisible

    def visualize(
        self,
        translations: torch.Tensor | None,
        orientations: torch.Tensor | None,
        scales: torch.Tensor | None,
        marker_indices: torch.Tensor | None,
    ) -> None:
        """Write marker transforms to USD PointInstancer attributes."""
        from pxr import Vt  # noqa: PLC0415

        num_markers = 0
        if translations is not None:
            translations_np = translations.detach().cpu().numpy()
            self._instancer_manager.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(translations_np))
            num_markers = translations_np.shape[0]
        if orientations is not None:
            orientations_np = orientations.detach().cpu().numpy()
            self._instancer_manager.GetOrientationsAttr().Set(Vt.QuathArray.FromNumpy(orientations_np))
            num_markers = orientations_np.shape[0]
        if scales is not None:
            scales_np = scales.detach().cpu().numpy()
            self._instancer_manager.GetScalesAttr().Set(Vt.Vec3fArray.FromNumpy(scales_np))
            num_markers = scales_np.shape[0]
        if marker_indices is not None or num_markers != self._count:
            if marker_indices is not None:
                marker_indices_np = marker_indices.detach().cpu().numpy()
                self._instancer_manager.GetProtoIndicesAttr().Set(Vt.IntArray.FromNumpy(marker_indices_np))
                num_markers = marker_indices_np.shape[0]
            elif num_markers != 0:
                self._instancer_manager.GetProtoIndicesAttr().Set([0] * num_markers)
        if num_markers != 0:
            self._count = num_markers

    def _add_markers_prototypes(self, markers_cfg: dict[str, sim_utils.SpawnerCfg]) -> None:
        """Create marker prototype prims under the PointInstancer."""
        for name, cfg in markers_cfg.items():
            marker_prim_path = f"{self.prim_path}/{name}"
            marker_prim = cfg.func(prim_path=marker_prim_path, cfg=cfg)
            self._process_prototype_prim(marker_prim)
            self._instancer_manager.GetPrototypesRel().AddTarget(marker_prim_path)

        prototypes = self._instancer_manager.GetPrototypesRel().GetTargets()
        if len(prototypes) != len(markers_cfg):
            raise RuntimeError(
                f"Failed to load all the prototypes. Expected: {len(markers_cfg)}. Received: {len(prototypes)}."
            )

    def _process_prototype_prim(self, prim) -> None:
        """Remove simulation APIs from marker prototypes and prepare them for instancing."""
        from pxr import Sdf, UsdGeom, UsdPhysics  # noqa: PLC0415

        if not prim.IsValid():
            raise ValueError(f"Prim at path '{prim.GetPrimAtPath()}' is not valid.")

        all_prims = [prim]
        while len(all_prims) > 0:
            child_prim = all_prims.pop(0)
            if child_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                child_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                child_prim.RemoveAppliedSchema("PhysxArticulationAPI")
            if child_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                child_prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                child_prim.RemoveAppliedSchema("PhysxRigidBodyAPI")
            if child_prim.IsA(UsdPhysics.Joint):
                child_prim.GetAttribute("physics:jointEnabled").Set(False)
            if child_prim.IsInstance():
                child_prim.SetInstanceable(False)
            if child_prim.IsA(UsdGeom.Gprim):
                sim_utils.change_prim_property(
                    prop_path=f"{child_prim.GetPrimPath().pathString}.primvars:invisibleToSecondaryRays",
                    value=True,
                    stage=prim.GetStage(),
                    type_to_create_if_not_exist=Sdf.ValueTypeNames.Bool,
                )
            all_prims += child_prim.GetChildren()

        if has_kit():
            import omni.physx.scripts.utils as physx_utils  # noqa: PLC0415

            physx_utils.removeRigidBodySubtree(prim)
