# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import shutil

# Define source and destination directories
source_base = "/home/bingjie/repos/shared/isaacgymenvs/assets/automate/mesh"
destination_base = "/home/bingjie/Downloads/assembly_asset"

# List of assembly IDs
assembly_ids = [
    "01053",
    "00470",
    "00499",
    "00652",
    "00346",
    "00004",
    "00681",
    "00783",
    "00768",
    "00192",
    "01026",
    "00007",
    "00308",
    "00437",
    "00446",
    "00016",
    "00133",
    "00855",
    "00471",
    "00030",
    "00186",
    "01132",
    "00863",
    "00426",
    "00077",
    "00703",
    "00028",
    "00110",
    "00015",
    "00329",
    "01125",
    "00103",
    "00444",
    "00014",
    "00615",
    "00078",
    "00187",
    "01029",
    "00021",
    "01041",
    "00755",
    "01102",
    "00614",
    "00597",
    "00686",
    "01129",
    "00074",
    "00293",
    "00649",
    "00638",
    "00143",
    "00345",
    "00537",
    "00648",
    "00388",
    "00163",
    "00117",
    "01079",
    "00340",
    "00506",
    "00741",
    "00271",
    "00301",
    "00062",
    "00210",
    "00731",
    "00042",
    "00559",
    "00514",
    "00726",
    "00553",
    "01136",
    "00083",
    "00319",
    "00417",
    "00175",
    "00700",
    "00360",
    "00318",
    "00320",
    "00581",
    "01092",
    "00410",
    "00213",
    "00659",
    "00831",
    "00422",
    "00296",
    "00138",
    "01036",
    "00486",
    "00860",
    "00211",
    "00480",
    "00081",
    "00256",
    "00190",
    "00032",
    "00141",
    "00255",
]  # Add more IDs as needed


def copy_and_rename_files(assembly_id):
    # Define source and destination paths
    source_mesh_dir = f"/home/bingjie/repos/shared/isaacgymenvs/assets/automate/mesh/{assembly_id}/"
    dest_dir = f"/home/bingjie/Downloads/assembly_asset/{assembly_id}/"

    source_traj_file = f"/home/bingjie/repos/shared/isaacgymenvs/isaacgymenvs/tasks/automate/data/asset_{assembly_id}_disassemble_traj.json"
    dest_traj_file = f"{dest_dir}/disassemble_traj.json"

    # Ensure destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    # Copy and rename mesh files
    for src_file, dest_file in [("asset_plug.obj", "plug.obj"), ("asset_socket.obj", "socket.obj")]:
        src_path = os.path.join(source_mesh_dir, src_file)
        dest_path = os.path.join(dest_dir, dest_file)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)
            print(f"Copied {src_path} to {dest_path}")
        else:
            print(f"Warning: {src_path} does not exist")

    # Copy disassembly trajectory file
    if os.path.exists(source_traj_file):
        shutil.copy2(source_traj_file, dest_traj_file)
        print(f"Copied {source_traj_file} to {dest_traj_file}")
    else:
        print(f"Warning: {source_traj_file} does not exist")


for assembly_id in assembly_ids:
    copy_and_rename_files(assembly_id)

# print("Done!")

# from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf
# import os

# BASE_PATH = "/home/bingjie/Downloads/assembly_asset"

# # Define color values for plug and socket
# COLORS = {
#     "plug": {
#         "Ka": Gf.Vec3f(0.5, 0.5, 0.5),
#         "Kd": Gf.Vec3f(0.952941, 0.870588, 0.741176),
#     },
#     "socket": {
#         "Ka": Gf.Vec3f(0.5, 0.5, 0.5),
#         "Kd": Gf.Vec3f(0.243137, 0.592157, 0.509804),
#     }
# }

# def apply_material(stage, diffuse_color, material_name):
#     """Creates a USD material and applies it to all meshes in the USD stage."""
#     material_path = f"/{material_name}"
#     material = UsdShade.Material.Define(stage, material_path)

#     # Create a USD Preview Surface shader
#     shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
#     shader.CreateIdAttr("UsdPreviewSurface")

#     # Set the diffuse color
#     shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(diffuse_color)

#     # Connect the shader to the material
#     # material.CreateSurfaceOutput("mdl").ConnectToSource(shader, "surface")
#     material.CreateSurfaceOutput("UsdPreviewSurface").ConnectToSource(shader.ConnectableAPI(), "surface")

#     # Bind the material to all meshes in the stage
#     for prim in stage.Traverse():

#         # print(f"- {prim.GetPath()} ({prim.GetTypeName()})")
#         if prim.GetName() == "visuals":  # Only enter "visuals" groups
#             print(f"- {prim.GetPath()} ({prim.GetTypeName()})")
#             for child in prim.GetChildren():
#                 # if child.IsA(UsdGeom.Mesh):
#                 #     print(f"✅ Applying material to {child.GetPath()}")
#                 #     UsdShade.MaterialBindingAPI.Apply(child).Bind(material)
#                 print(f"- {child.GetPath()} ({child.GetTypeName()})")
#         # if prim.IsA(UsdGeom.Mesh):

#             # UsdShade.MaterialBindingAPI(prim).Bind(material)

#             # print("applying GetDisplayColorAttr")
#             # mesh = UsdGeom.Mesh(prim)
#             # mesh.GetDisplayColorAttr().Set([diffuse_color])  # Fallback color

#     return stage

# def process_usd(file_path, diffuse_color, material_name):
#     """Load USD file, apply color, and save changes."""
#     if not os.path.exists(file_path):
#         print(f"File not found: {file_path}")
#         return

#     # Open the USD stage
#     stage = Usd.Stage.Open(file_path)

#     # Apply the material
#     stage = apply_material(stage, diffuse_color, material_name)

#     # Save changes
#     stage.GetRootLayer().Save()
#     # print(f"Updated USD file: {file_path} with material {material_name}")

# selected_ids = [
#     # '00681',
#     # '00320',
#     # '00768',
#     # '00015',
#     # '00731',
#     # '01036',
#     # '00340',
#     # '01041',
#     # '00296',
#     '01129'
# ]

# # Iterate over all assembly IDs
# for assembly_id in selected_ids:
#     assembly_path = os.path.join(BASE_PATH, assembly_id)

#     plug_file = os.path.join(assembly_path, "plug.usd")
#     socket_file = os.path.join(assembly_path, "socket.usd")

#     print(f"Processing assembly {assembly_id}...")

#     # Process plug and socket
#     process_usd(plug_file, COLORS["plug"]["Kd"], f"PlugMaterial_{assembly_id}")
#     process_usd(socket_file, COLORS["socket"]["Kd"], f"SocketMaterial_{assembly_id}")

# print("Processing complete for all assemblies.")
