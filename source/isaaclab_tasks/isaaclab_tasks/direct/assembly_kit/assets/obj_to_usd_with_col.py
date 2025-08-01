# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import bpy
import math
import os

remesh = False  # set to False to skip remeshing
kit = False  # set to False to convert models instead of kits

# ─── 3) Define your input and output directories ───────────────────────────────
input_folder = (
    r"/home/johann/.maniskill/data/tasks/assembling_kits/kits"
    if kit
    else r"/home/johann/.maniskill/data/tasks/assembling_kits/models/visual"
)

if remesh:
    output_folder = (
        r"/home/johann/Downloads/assembly_kit/kits" if kit else r"/home/johann/Downloads/assembly_kit/models"
    )
else:
    output_folder = (
        r"/home/johann/Downloads/assembly_kit_noremesh/kits"
        if kit
        else r"/home/johann/Downloads/assembly_kit_noremesh/models"
    )

# ─── 4) Iterate over every .obj in the input folder ────────────────────────────
print(os.listdir(input_folder))
for fn in os.listdir(input_folder):
    if not fn.lower().endswith(".obj"):
        continue

    kit_name = os.path.splitext(fn)[0]
    input_path = os.path.join(input_folder, fn)

    # Make a same-named subfolder in the output folder
    output_sub = os.path.join(output_folder, kit_name)
    os.makedirs(output_sub, exist_ok=True)

    # Where to write the .usda
    output_path = os.path.join(output_sub, f"{kit_name}.usda")

    # ─── 4) Reset scene to empty ───────────────────────────────────────────
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # ─── 2) Ensure USD add‐on is enabled ────────────────────────────────────────────
    prefs = bpy.context.preferences
    if "usd_hook_collisions" not in prefs.addons:
        bpy.ops.preferences.addon_enable(module="usd_hook_collisions")

    # ─── 5) Import the OBJ ───────────────────────────────────────────────
    if bpy.app.version[0] >= 4:
        bpy.ops.wm.obj_import(
            filepath=input_path,
        )

    else:
        bpy.ops.import_scene.obj(
            filepath=input_path,
            use_edges=True,
            use_smooth_groups=True,
            use_split_objects=True,
            axis_forward="X",
            axis_up="Z",
        )

    # ─── 6) (Optional) Post-import tweaks go here ──────────────────────────

    # after you import:
    for obj in bpy.context.selected_objects:
        if len(bpy.context.selected_objects) != 1:
            raise ValueError(
                "Please select only one object to export. "
                "If you want to export multiple objects, "
                "please select them all before running this script."
            )
        # obj.name = kit_name
        obj.rotation_euler = (
            math.radians(0),
            0,
            math.pi if kit else 0,
        )  # rotate 90° on X and 180° on Z
        # bake that rotation into the mesh data:
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(rotation=True)

        # ─── 6.1) Clean up the mesh: remesh to get a watertight, uniform topology ──
        bpy.context.view_layer.objects.active = obj

        obj.scale = (1, 1, 1) if kit else (0.003, 0.003, 0.001)
        bpy.ops.object.transform_apply(scale=True)

        # # Option A) Voxel Remesh modifier
        if remesh:
            remesh_mod = obj.modifiers.new(name="Remesh", type="REMESH")
            remesh_mod.mode = "VOXEL"  # 'VOXEL' fills holes best; 'BLOCKS' also available
            # remesh_mod.adaptivity = 0.05
            # remesh_mod.voxel_size = 0.0005  # tune this: smaller = higher resolution
            remesh_mod.adaptivity = 0.05
            remesh_mod.voxel_size = 0.0002  # tune this: smaller = higher resolution
            remesh_mod.use_smooth_shade = False  # keep flat facets (good for collision)
            remesh_mod.use_remove_disconnected = True  # keep all pieces
            # bpy.ops.object.modifier_apply(modifier=remesh_mod.name)

    # ─── 7) Export to USD ────────────────────────────────────────────────
    bpy.ops.wm.usd_export(
        filepath=output_path,
        check_existing=True,
        export_materials=False,
        generate_preview_surface=False,
        generate_materialx_network=False,
        convert_orientation=True,
    )

    print(f"✓ {fn} → {kit_name}.usd")
