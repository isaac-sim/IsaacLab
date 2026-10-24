# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared collision preparation for the packing-table USD asset."""

from isaaclab.assets import AssetBaseCfg


def configure_packing_table_collisions(table_cfg: AssetBaseCfg, *, physx_offsets: bool = False) -> None:
    """Move the asset's non-geometric collision anchor onto its existing meshes.

    The source USD attaches a bounding-cube collision approximation to an Xform.
    Both standalone backends need concrete geometry. Preserve the authored mesh
    transforms and approximation, then use the common schema writers for each
    mesh. The five bin colliders are left intact; no replacement geometry or
    backend-specific cooking implementation is needed.
    """
    if table_cfg.spawn is None:
        return
    table_cfg.spawn._packing_table_physx_offsets = physx_offsets
    if getattr(table_cfg.spawn, "_packing_table_collision_prepared", False):
        return
    original_spawn = table_cfg.spawn.func

    def _spawn_with_mesh_collisions(prim_path, cfg, *args, **kwargs):
        from pxr import Usd, UsdGeom, UsdPhysics

        import isaaclab.sim as sim_utils

        prim = original_spawn(prim_path, cfg, *args, **kwargs)
        stage = prim.GetStage()
        sim_utils.make_uninstanceable(str(prim.GetPath()), stage=stage)
        for child in list(Usd.PrimRange(prim)):
            if not child.HasAPI(UsdPhysics.MeshCollisionAPI) or child.IsA(UsdGeom.Gprim):
                continue
            approximation = UsdPhysics.MeshCollisionAPI(child).GetApproximationAttr().Get()
            child.RemoveAPI(UsdPhysics.MeshCollisionAPI)
            child.RemoveAPI(UsdPhysics.CollisionAPI)
            for mesh in Usd.PrimRange(child):
                if mesh.IsA(UsdGeom.Mesh):
                    path = str(mesh.GetPath())
                    sim_utils.apply_collision_properties(
                        path,
                        [sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)],
                        stage=stage,
                        create_if_missing=True,
                    )
                    sim_utils.apply_mesh_collision_properties(
                        path,
                        [sim_utils.UsdPhysicsMeshCollisionCfg(mesh_approximation_name=approximation or "boundingCube")],
                        stage=stage,
                    )
        if cfg._packing_table_physx_offsets:
            from isaaclab_physx.sim.schemas import PhysxCollisionCfg

            sim_utils.apply_collision_properties(
                str(prim.GetPath()) + "/.*",
                [PhysxCollisionCfg(contact_offset=0.001, rest_offset=0.0)],
                stage=stage,
            )
        return prim

    table_cfg.spawn.func = _spawn_with_mesh_collisions
    table_cfg.spawn._packing_table_collision_prepared = True


def condition_packing_table_for_kitless_ovphysx(table_cfg: AssetBaseCfg) -> None:
    """Prepare shared mesh collisions with the accepted PhysX contact offsets."""
    configure_packing_table_collisions(table_cfg, physx_offsets=True)
