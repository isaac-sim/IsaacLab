# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import bpy
import os

# ─── USER SETTINGS ──────────────────────────────────────────────────────────────

# Output directory for USD files
output_folder = "/home/johann/Downloads/peg_insertion_side"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Number of samples along each dimension
num_lengths = 5
num_radii = 5

# ManiSkill sampling ranges for peg half-length and radius
min_length = 0.085  # half-length range
max_length = 0.125
min_radius = 0.015
max_radius = 0.025

# Clearance between peg and hole
clearance = 0.003

# Small overlap to hide seam at center
epsilon = 1e-4

# ─── INITIALIZATION ─────────────────────────────────────────────────────────────


def ensure_usd_addon():
    prefs = bpy.context.preferences
    if "usd_hook_collisions" not in prefs.addons:
        bpy.ops.preferences.addon_enable(module="usd_hook_collisions")


ensure_usd_addon()


# Helper: reset scene to empty and re-enable addon
def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    ensure_usd_addon()
    bpy.context.scene.unit_settings.system = "METRIC"
    bpy.context.scene.unit_settings.scale_length = 1.0


# Helper: export current objects to USD without materials
def export_usd(name):
    path = os.path.join(output_folder, f"{name}.usda")
    bpy.ops.wm.usd_export(
        filepath=path,
        check_existing=True,
        export_materials=False,
        generate_preview_surface=False,
        generate_materialx_network=False,
        convert_orientation=True,
    )
    print(f"✓ Exported {path}")


# ─── SAMPLING GRIDS ──────────────────────────────────────────────────────────────

lengths = [min_length + i * (max_length - min_length) / (num_lengths - 1) for i in range(num_lengths)]
radii = [min_radius + j * (max_radius - min_radius) / (num_radii - 1) for j in range(num_radii)]

# ─── MAIN LOOP ─────────────────────────────────────────────────────────────────

for length in lengths:
    for radius in radii:
        inner_r = radius + clearance
        outer_r = length
        depth = length
        thickness = 0.5 * (outer_r - inner_r)
        offset = thickness + inner_r

        # PEG: create two halves and join into single prim, then center origin & rotate
        reset_scene()

        # head half
        bpy.ops.mesh.primitive_cube_add(size=2, location=(length / 2 - epsilon, 0, 0))
        head = bpy.context.active_object
        head.name = "peg_head"
        head.scale = (length / 2 + epsilon, radius + epsilon, radius + epsilon)

        # tail half
        bpy.ops.mesh.primitive_cube_add(size=2, location=(-length / 2 + epsilon, 0, 0))
        tail = bpy.context.active_object
        tail.name = "peg_tail"
        tail.scale = (length / 2 + epsilon, radius + epsilon, radius + epsilon)

        # join into single peg prim
        bpy.ops.object.select_all(action="DESELECT")
        head.select_set(True)
        tail.select_set(True)
        bpy.context.view_layer.objects.active = head
        bpy.ops.object.join()
        peg = bpy.context.active_object
        peg.name = "peg"
        peg["collision"] = True
        # set origin to geometric center along long axis
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN")
        # rotate peg 90 degrees around Z axis
        bpy.ops.object.transform_apply(rotation=True)

        export_usd(f"peg_L{length:.3f}_R{radius:.3f}")

        # BOX WITH HOLE: walls joined then centered & rotated
        reset_scene()
        wall_objs = []
        half_sizes = [
            (depth, thickness, outer_r),
            (depth, thickness, outer_r),
            (depth, outer_r, thickness),
            (depth, outer_r, thickness),
        ]
        positions = [
            (0, offset, 0),
            (0, -offset, 0),
            (0, 0, offset),
            (0, 0, -offset),
        ]
        for hs, loc in zip(half_sizes, positions):
            bpy.ops.mesh.primitive_cube_add(size=2, location=loc)
            wall = bpy.context.active_object
            wall.scale = hs
            wall_objs.append(wall)

        # join walls into single box prim
        bpy.ops.object.select_all(action="DESELECT")
        for w in wall_objs:
            w.select_set(True)
        bpy.context.view_layer.objects.active = wall_objs[0]
        bpy.ops.object.join()
        box = bpy.context.active_object
        box.name = "box"
        box["collision"] = True
        # set origin to geometric center
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN")
        # rotate box 90 degrees around Z axis
        bpy.ops.object.transform_apply(rotation=True)

        export_usd(f"box_L{length:.3f}_R{radius:.3f}")

print("All done!")
