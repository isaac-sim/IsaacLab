# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit/USD implementation for :class:`VisualizationMarkers`.

This backend represents markers as :class:`UsdGeom.PointInstancer` prims in the
USD stage. Marker prototypes are created as child prims of the point instancer
and are instanced efficiently through prototype indices.

.. _UsdGeom.PointInstancer: https://graphics.pixar.com/usd/dev/api/class_usd_geom_point_instancer.html
"""

from __future__ import annotations

import logging

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers.visualization_markers_cfg import VisualizationMarkersCfg
from isaaclab.utils.version import has_kit

logger = logging.getLogger(__name__)


class KitVisualizationMarkers:
    """USD PointInstancer backend for visualization markers.

    This class wraps around the `UsdGeom.PointInstancer`_ for efficient
    handling of objects in the USD stage by instancing the created marker
    prototype prims.

    A marker prototype prim is a reusable template prim used for defining
    variations of objects in the scene. For example, a sphere prim can be used
    as a marker prototype prim to create multiple sphere prims at different
    locations. The marker prim path is resolved using the marker name from the
    :attr:`VisualizationMarkersCfg.markers` dictionary.

    .. _UsdGeom.PointInstancer: https://graphics.pixar.com/usd/dev/api/class_usd_geom_point_instancer.html
    """

    def __init__(self, cfg: VisualizationMarkersCfg, visible: bool = True):
        """Initialize the USD point instancer and register marker prototypes.

        When this backend is initialized, the :class:`UsdGeom.PointInstancer`
        is created in the stage and the marker prims are registered into it.

        .. note::
            If a prim already exists at the requested path, the next free path
            is used for the :class:`UsdGeom.PointInstancer` prim.
        """
        self.cfg = cfg
        self.stage = sim_utils.get_current_stage()
        # Resolve the next free prim path before creating the point instancer.
        self.prim_path = sim_utils.get_next_free_prim_path(cfg.prim_path)
        self._is_visible = visible
        self._count = len(cfg.markers)

        from pxr import Gf, UsdGeom  # noqa: PLC0415

        self._instancer_manager = UsdGeom.PointInstancer.Define(self.stage, self.prim_path)
        self._add_markers_prototypes(self.cfg.markers)
        # Note: We need to do this the first time to initialize the instancer.
        # Otherwise, the instancer is not fully "created" and USD instance
        # queries such as GetInstanceIndices() can fail.
        self._instancer_manager.GetProtoIndicesAttr().Set(list(range(len(self.cfg.markers))))
        self._instancer_manager.GetPositionsAttr().Set([Gf.Vec3f(0.0)] * len(self.cfg.markers))
        self.set_visibility(visible)

    @property
    def count(self) -> int:
        return self._count

    def set_visibility(self, visible: bool) -> None:
        """Set USD PointInstancer visibility.

        The method does this through the USD API.
        """
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
        """Write marker transforms to USD PointInstancer attributes.

        Args:
            translations: Translations w.r.t. parent prim frame. Shape is
                (M, 3).
            orientations: Quaternion orientations (x, y, z, w) w.r.t. parent
                prim frame. Shape is (M, 4).
            scales: Scale applied before any rotation is applied. Shape is
                (M, 3).
            marker_indices: Decides which marker prototype to visualize. Shape
                is (M).
        """
        from pxr import Vt  # noqa: PLC0415

        num_markers = 0
        if translations is not None:
            translations_np = translations.detach().cpu().numpy()
            # Apply translations.
            self._instancer_manager.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(translations_np))
            num_markers = translations_np.shape[0]
        if orientations is not None:
            orientations_np = orientations.detach().cpu().numpy()
            # Apply orientations. USD expects quaternion data in xyzw format.
            self._instancer_manager.GetOrientationsAttr().Set(Vt.QuathArray.FromNumpy(orientations_np))
            num_markers = orientations_np.shape[0]
        if scales is not None:
            scales_np = scales.detach().cpu().numpy()
            # Apply scales.
            self._instancer_manager.GetScalesAttr().Set(Vt.Vec3fArray.FromNumpy(scales_np))
            num_markers = scales_np.shape[0]
        if marker_indices is not None or num_markers != self._count:
            if marker_indices is not None:
                marker_indices_np = marker_indices.detach().cpu().numpy()
                # Apply prototype indices.
                self._instancer_manager.GetProtoIndicesAttr().Set(Vt.IntArray.FromNumpy(marker_indices_np))
                num_markers = marker_indices_np.shape[0]
            elif num_markers != 0:
                # Set all markers to the first prototype when the marker count
                # changes and explicit marker indices are not provided.
                self._instancer_manager.GetProtoIndicesAttr().Set([0] * num_markers)
        if num_markers != 0:
            self._count = num_markers

    def _add_markers_prototypes(self, markers_cfg: dict[str, sim_utils.SpawnerCfg]) -> None:
        """Add marker prototypes to the scene and register them with the point instancer."""
        # Add markers based on config.
        for name, cfg in markers_cfg.items():
            # Resolve prim path from the marker name.
            marker_prim_path = f"{self.prim_path}/{name}"
            # Create a child prim for the marker.
            marker_prim = cfg.func(prim_path=marker_prim_path, cfg=cfg)
            # Make the asset uninstanceable in case it is already instanced.
            # Point instancer defines its own prototypes, so already-instanced
            # assets cannot be used directly.
            self._process_prototype_prim(marker_prim)
            # Add child reference to point instancer.
            self._instancer_manager.GetPrototypesRel().AddTarget(marker_prim_path)

        # Check that all prototypes were loaded.
        prototypes = self._instancer_manager.GetPrototypesRel().GetTargets()
        if len(prototypes) != len(markers_cfg):
            raise RuntimeError(
                f"Failed to load all the prototypes. Expected: {len(markers_cfg)}. Received: {len(prototypes)}."
            )

    def _process_prototype_prim(self, prim) -> None:
        """Process a prim and its descendants to make them suitable for prototypes.

        Point instancer defines its own prototypes, so if an asset is already
        instanced, this does not work. This function checks if the prim and its
        descendants are instanced. If so, it makes the respective prim
        uninstanceable by disabling instancing on the prim.

        Additionally, it makes the prim invisible to secondary rays. This is
        useful when marker prims should not appear in camera images.

        Args:
            prim: The prim to process.
        """
        from pxr import Sdf, UsdGeom, UsdPhysics  # noqa: PLC0415

        if not prim.IsValid():
            raise ValueError(f"Prim at path '{prim.GetPrimAtPath()}' is not valid.")

        # Iterate over all prims under the marker prim path.
        all_prims = [prim]
        while len(all_prims) > 0:
            child_prim = all_prims.pop(0)
            # Remove physics from marker prototypes because they are only for
            # visualization.
            if child_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                child_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                child_prim.RemoveAppliedSchema("PhysxArticulationAPI")
            if child_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                child_prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                child_prim.RemoveAppliedSchema("PhysxRigidBodyAPI")
            if child_prim.IsA(UsdPhysics.Joint):
                child_prim.GetAttribute("physics:jointEnabled").Set(False)
            # Point instancer defines its own instancing, so nested instances
            # must be made uninstanceable.
            if child_prim.IsInstance():
                child_prim.SetInstanceable(False)
            # Make renderable prims invisible to secondary rays such as depth
            # images.
            if child_prim.IsA(UsdGeom.Gprim):
                sim_utils.change_prim_property(
                    prop_path=f"{child_prim.GetPrimPath().pathString}.primvars:invisibleToSecondaryRays",
                    value=True,
                    stage=prim.GetStage(),
                    type_to_create_if_not_exist=Sdf.ValueTypeNames.Bool,
                )
            all_prims += child_prim.GetChildren()

        # Remove any remaining physics on the markers because they are only for
        # visualization.
        if has_kit():
            import omni.physx.scripts.utils as physx_utils  # noqa: PLC0415

            physx_utils.removeRigidBodySubtree(prim)
