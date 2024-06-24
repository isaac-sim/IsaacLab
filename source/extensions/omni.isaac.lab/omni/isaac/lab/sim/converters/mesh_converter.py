# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
import os

import omni
import omni.kit.commands
import omni.usd
from omni.isaac.core.utils.extensions import enable_extension
from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils

from isaaclab.sim.converters.asset_converter_base import AssetConverterBase
from isaaclab.sim.converters.mesh_converter_cfg import MeshConverterCfg
from isaaclab.sim.schemas import schemas
from isaaclab.sim.utils import export_prim_to_file


class MeshConverter(AssetConverterBase):
    """Converter for a mesh file in OBJ / STL / FBX format to a USD file.

    This class wraps around the `omni.kit.asset_converter`_ extension to provide a lazy implementation
    for mesh to USD conversion. It stores the output USD file in an instanceable format since that is
    what is typically used in all learning related applications.

    To make the asset instanceable, we must follow a certain structure dictated by how USD scene-graph
    instancing and physics work. The rigid body component must be added to each instance and not the
    referenced asset (i.e. the prototype prim itself). This is because the rigid body component defines
    properties that are specific to each instance and cannot be shared under the referenced asset. For
    more information, please check the `documentation <https://docs.omniverse.nvidia.com/extensions/latest/ext_physics/rigid-bodies.html#instancing-rigid-bodies>`_.

    Due to the above, we follow the following structure:

    * ``{prim_path}`` - The root prim that is an Xform with the rigid body and mass APIs if configured.
    * ``{prim_path}/geometry`` - The prim that contains the mesh and optionally the materials if configured.
      If instancing is enabled, this prim will be an instanceable reference to the prototype prim.

    .. _omni.kit.asset_converter: https://docs.omniverse.nvidia.com/extensions/latest/ext_asset-converter.html

    .. caution::
        When converting STL files, Z-up convention is assumed, even though this is not the default for many CAD
        export programs. Asset orientation convention can either be modified directly in the CAD program's export
        process or an offset can be added within the config in Isaac Lab.

    """

    cfg: MeshConverterCfg
    """The configuration instance for mesh to USD conversion."""

    def __init__(self, cfg: MeshConverterCfg):
        """Initializes the class.

        Args:
            cfg: The configuration instance for mesh to USD conversion.
        """
        super().__init__(cfg=cfg)

    """
    Implementation specific methods.
    """

    def _convert_asset(self, cfg: MeshConverterCfg):
        """Generate USD from OBJ, STL or FBX.

        It stores the asset in the following format:

        /file_name (default prim)
          |- /geometry <- Made instanceable if requested
            |- /Looks
            |- /mesh

        Args:
            cfg: The configuration for conversion of mesh to USD.

        Raises:
            RuntimeError: If the conversion using the Omniverse asset converter fails.
        """
        # resolve mesh name and format
        mesh_file_basename, mesh_file_format = os.path.basename(cfg.asset_path).split(".")
        mesh_file_format = mesh_file_format.lower()

        # Convert USD
        asyncio.get_event_loop().run_until_complete(
            self._convert_mesh_to_usd(
                in_file=cfg.asset_path, out_file=self.usd_path, prim_path=f"/{mesh_file_basename}"
            )
        )
        # Open converted USD stage
        # note: This opens a new stage and does not use the stage created earlier by the user
        # create a new stage
        stage = Usd.Stage.Open(self.usd_path)
        # add USD to stage cache
        stage_id = UsdUtils.StageCache.Get().Insert(stage)
        # Get the default prim (which is the root prim) -- "/{mesh_file_basename}"
        xform_prim = stage.GetDefaultPrim()
        geom_prim = stage.GetPrimAtPath(f"/{mesh_file_basename}/geometry")
        # Move all meshes to underneath new Xform
        for child_mesh_prim in geom_prim.GetChildren():
            if child_mesh_prim.GetTypeName() == "Mesh":
                # Apply collider properties to mesh
                if cfg.collision_props is not None:
                    # -- Collision approximation to mesh
                    # TODO: https://github.com/isaac-orbit/orbit/issues/163 Move this to a new Schema
                    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(child_mesh_prim)
                    mesh_collision_api.GetApproximationAttr().Set(cfg.collision_approximation)
                    # -- Collider properties such as offset, scale, etc.
                    schemas.define_collision_properties(
                        prim_path=child_mesh_prim.GetPath(), cfg=cfg.collision_props, stage=stage
                    )
        # Delete the old Xform and make the new Xform the default prim
        stage.SetDefaultPrim(xform_prim)
        # Handle instanceable
        # Create a new Xform prim that will be the prototype prim
        if cfg.make_instanceable:
            # Export Xform to a file so we can reference it from all instances
            export_prim_to_file(
                path=os.path.join(self.usd_dir, self.usd_instanceable_meshes_path),
                source_prim_path=geom_prim.GetPath(),
                stage=stage,
            )
            # Delete the original prim that will now be a reference
            geom_prim_path = geom_prim.GetPath().pathString
            omni.kit.commands.execute("DeletePrims", paths=[geom_prim_path], stage=stage)
            # Update references to exported Xform and make it instanceable
            geom_undef_prim = stage.DefinePrim(geom_prim_path)
            geom_undef_prim.GetReferences().AddReference(self.usd_instanceable_meshes_path, primPath=geom_prim_path)
            geom_undef_prim.SetInstanceable(True)

        # Apply mass and rigid body properties after everything else
        # Properties are applied to the top level prim to avoid the case where all instances of this
        #   asset unintentionally share the same rigid body properties
        # apply mass properties
        if cfg.mass_props is not None:
            schemas.define_mass_properties(prim_path=xform_prim.GetPath(), cfg=cfg.mass_props, stage=stage)
        # apply rigid body properties
        if cfg.rigid_props is not None:
            schemas.define_rigid_body_properties(prim_path=xform_prim.GetPath(), cfg=cfg.rigid_props, stage=stage)

        # Save changes to USD stage
        stage.Save()
        if stage_id is not None:
            UsdUtils.StageCache.Get().Erase(stage_id)

    """
    Helper methods.
    """

    @staticmethod
    async def _convert_mesh_to_usd(
        in_file: str, out_file: str, prim_path: str = "/World", load_materials: bool = True
    ) -> bool:
        """Convert mesh from supported file types to USD.

        This function uses the Omniverse Asset Converter extension to convert a mesh file to USD.
        It is an asynchronous function and should be called using `asyncio.get_event_loop().run_until_complete()`.

        The converted asset is stored in the USD format in the specified output file.
        The USD file has Y-up axis and is scaled to meters.

        The asset hierarchy is arranged as follows:

        .. code-block:: none
            prim_path (default prim)
                |- /geometry/Looks
                |- /geometry/mesh

        Args:
            in_file: The file to convert.
            out_file: The path to store the output file.
            prim_path: The prim path of the mesh.
            load_materials: Set to True to enable attaching materials defined in the input file
                to the generated USD mesh. Defaults to True.

        Returns:
            True if the conversion succeeds.
        """
        enable_extension("omni.kit.asset_converter")
        enable_extension("omni.usd.metrics.assembler")

        import omni.kit.asset_converter
        import omni.usd
        from omni.metrics.assembler.core import get_metrics_assembler_interface

        # Create converter context
        converter_context = omni.kit.asset_converter.AssetConverterContext()
        # Set up converter settings
        # Don't import/export materials
        converter_context.ignore_materials = not load_materials
        converter_context.ignore_animations = True
        converter_context.ignore_camera = True
        converter_context.ignore_light = True
        # Merge all meshes into one
        converter_context.merge_all_meshes = True
        # Sets world units to meters, this will also scale asset if it's centimeters model.
        # This does not work right now :(, so we need to scale the mesh manually
        converter_context.use_meter_as_world_unit = True
        converter_context.baking_scales = True
        # Uses double precision for all transform ops.
        converter_context.use_double_precision_to_usd_transform_op = True

        # Create converter task
        instance = omni.kit.asset_converter.get_instance()
        out_file_non_metric = out_file.replace(".usd", "_non_metric.usd")
        task = instance.create_converter_task(in_file, out_file_non_metric, None, converter_context)
        # Start conversion task and wait for it to finish
        success = True
        while True:
            success = await task.wait_until_finished()
            if not success:
                await asyncio.sleep(0.1)
            else:
                break

        temp_stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(temp_stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(temp_stage, 1.0)
        UsdPhysics.SetStageKilogramsPerUnit(temp_stage, 1.0)

        base_prim = temp_stage.DefinePrim(prim_path, "Xform")
        prim = temp_stage.DefinePrim(f"{prim_path}/geometry", "Xform")
        prim.GetReferences().AddReference(out_file_non_metric)
        cache = UsdUtils.StageCache.Get()
        cache.Insert(temp_stage)
        stage_id = cache.GetId(temp_stage).ToLongInt()
        get_metrics_assembler_interface().resolve_stage(stage_id)
        temp_stage.SetDefaultPrim(base_prim)
        temp_stage.Export(out_file)
        return success
