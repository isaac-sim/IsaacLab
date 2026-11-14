# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab visualization markers as point instancers in the shared stage.

Isaac Lab hands each marker group's state (translations, orientations, scales and prototype indices) to a
Newton-family marker backend, because the Isaac Sim 7 viewer does not pump Kit. Each group becomes a
``UsdGeom.PointInstancer`` that is added to the shared stage when the group first appears, with the group's own
spawner configurations as prototypes, as Isaac Lab's Kit marker backend builds them. Every frame then writes the
instancer's arrays at Isaac Lab's open ordinal.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .isaac_sim_shared_stage import SharedStage


def _instancer_usd(markers_cfg: dict[str, Any], root: str) -> str:
    """Author a point instancer whose prototypes are the marker spawners, and return it as USD text."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt

    import isaaclab.sim as sim_utils

    stage = Usd.Stage.CreateInMemory()
    with sim_utils.use_stage(stage):
        instancer = UsdGeom.PointInstancer.Define(stage, root)
        stage.SetDefaultPrim(instancer.GetPrim())
        # OVRTX also draws a prototype where it is authored. Its parent's visibility does not reach the instances,
        # so an invisible parent hides only that copy.
        UsdGeom.Scope.Define(stage, f"{root}/Prototypes").MakeInvisible()
        for name, cfg in markers_cfg.items():
            prim = cfg.func(prim_path=f"{root}/Prototypes/{name}", cfg=cfg)
            material = getattr(cfg, "visual_material", None)
            material_path = getattr(cfg, "visual_material_path", None)
            if material is not None and material_path is not None:
                # File spawners apply a separate material only with Kit; the material itself is plain UsdShade.
                if not material_path.startswith("/"):
                    material_path = f"{prim.GetPath()}/{material_path}"
                material.func(material_path, material)
                sim_utils.bind_visual_material(str(prim.GetPath()), material_path, stage=stage)
            # Prototypes are display-only: no physics, and no nested instancing under the point instancer.
            for child in Usd.PrimRange(prim):
                child.SetInstanceable(False)
                for api in (UsdPhysics.ArticulationRootAPI, UsdPhysics.RigidBodyAPI, UsdPhysics.CollisionAPI):
                    if child.HasAPI(api):
                        child.RemoveAPI(api)
            instancer.GetPrototypesRel().AddTarget(prim.GetPath())
    count = len(markers_cfg)
    # Author every array so the stage knows their types before the first runtime write.
    instancer.GetProtoIndicesAttr().Set(list(range(count)))
    instancer.GetPositionsAttr().Set([Gf.Vec3f(0.0, 0.0, -1.0e6)] * count)
    instancer.GetOrientationsAttr().Set(Vt.QuathArray([Gf.Quath(1.0, 0.0, 0.0, 0.0)] * count))
    instancer.GetScalesAttr().Set([Gf.Vec3f(1.0)] * count)
    return stage.GetRootLayer().ExportToString()


class SharedMarkers:
    """Point instancers in the shared stage that mirror Isaac Lab's visualization marker groups."""

    def __init__(self):
        self._roots: dict[str, str] = {}
        self._pending: set[str] = set()
        self._names = (f"/IsaacLabMarkers_{index}" for index in itertools.count())

    def sync(self, stage: SharedStage, registry: Any) -> None:
        """Add new marker groups and write every group's latest state at the stage's open ordinal.

        A group added at this ordinal receives its state from the next one, once its instancer is populated.

        Args:
            stage: The shared stage.
            registry: The simulation's visualization marker registry.
        """
        import ovstage

        from isaaclab_visualizers.newton.newton_visualization_markers import NewtonVisualizationMarkers

        groups = {
            group_id: group
            for group_id, group in registry.get_groups().items()
            if isinstance(group, NewtonVisualizationMarkers)
        }
        added = False
        for group_id, group in groups.items():
            root = self._roots.get(group_id)
            if root is None:
                root = self._roots[group_id] = next(self._names)
                ovstage.population.add_usd_reference_from_string(
                    stage.stage, _instancer_usd(group.cfg.markers, root), root
                )
                self._pending.add(group_id)
                added = True
            elif group_id in self._pending:
                self._pending.discard(group_id)
            else:
                self._write(stage, root, group)
        if added:
            ovstage.population.apply_usd_changes(stage.stage, ordinal=stage.open)
        for group_id in set(self._roots) - set(groups):
            # A removed group keeps its prims; an empty instancer draws nothing.
            self._write(stage, self._roots.pop(group_id), None)

    def _write(self, stage: SharedStage, root: str, group: Any) -> None:
        """Write one group's instancer arrays, or empty arrays when it is hidden or removed."""
        import ovstage
        from isaaclab_ov.stage import OVSTAGE_POINT_DTYPE

        count = 0 if group is None or not group.is_visible() else int(group.count)

        def column(value: Any, width: int, fill: np.ndarray, dtype: type) -> np.ndarray:
            if value is None or count == 0:
                return np.broadcast_to(fill, (count, width) if width > 1 else (count,)).astype(dtype)
            array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
            return np.ascontiguousarray(array[:count], dtype=dtype)

        positions = column(group.translations if count else None, 3, np.zeros(3), np.float32)
        orientations = column(group.orientations if count else None, 4, np.array([0.0, 0.0, 0.0, 1.0]), np.float16)
        scales = column(group.scales if count else None, 3, np.ones(3), np.float32)
        indices = column(group.marker_indices if count else None, 1, np.zeros(1), np.int32)
        half4 = ovstage.DLDataType(code=ovstage.DLDataTypeCode.kDLFloat, bits=16, lanes=4)
        float3 = OVSTAGE_POINT_DTYPE
        import warp as wp

        for attribute, values, vector, dtype in (
            ("positions", positions, wp.vec3f, float3),
            # USD point instancers store orientations as (x, y, z, w) half quaternions, as Isaac Lab passes them.
            ("orientations", orientations, wp.vec4h, half4),
            ("scales", scales, wp.vec3f, float3),
            ("protoIndices", indices, None, None),
        ):
            tensor = values
            if vector is not None:
                # ovstage folds the vector components into the column's lanes, as for Isaac Lab's ``points`` writes.
                tensor = ovstage.make_dltensor(wp.array(values, dtype=vector, device="cpu"), dtype=dtype)
            stage.stage.write_attribute(stage._query([root]), attribute, stage.open, [tensor], is_array=True).wait()
