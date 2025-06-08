import os
import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Utility to convert a mesh file into USD format.")
parser.add_argument("--input", type=str, default="/home/octipus/data/Props/Dextrah/Objects", help="The path to the input USD file.")
parser.add_argument("--output", type=str, default="source/isaaclab_assets/data/Props/Dextrah/Objects", help="The path to store the USD file.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.headless=True
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaacsim.core.utils.stage as stage_utils
from pxr import PhysxSchema, UsdPhysics, Sdf
import isaaclab.sim as sim_utils


def process_usd(input_usd: str, output_usd: str):
    stage_utils.open_stage(input_usd)
    stage = stage_utils.get_current_stage()
    layer = stage.GetRootLayer()

    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        raise RuntimeError("No defaultPrim! Make sure your USD file has one set.")
    orig_name = default_prim.GetName()
    orig_path = default_prim.GetPath()

    # 1) remove articulation APIs
    for prim in sim_utils.get_all_matching_child_prims(
        orig_path,
        predicate=lambda prim: prim.HasAPI(UsdPhysics.ArticulationRootAPI),
    ):
        prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)

    # 2) clear material bindings
    for prim in sim_utils.get_all_matching_child_prims(
        orig_path,
        predicate=lambda p: p.HasRelationship("material:binding"),
    ):
        rel = prim.GetRelationship("material:binding")
        rel.ClearTargets(True)

    # 3) remove Looks scopes
    for prim in sim_utils.get_all_matching_child_prims(
        orig_path,
        predicate=lambda p: p.GetName() == "Looks" or p.GetTypeName() == "Scope",
    ):
        stage.RemovePrim(prim.GetPath())

    # collect mesh & rigid-body prims
    mesh_prims = sim_utils.get_all_matching_child_prims(
        orig_path,
        predicate=lambda p: p.GetTypeName() == "Mesh"
    )
    rigidbody_prims = sim_utils.get_all_matching_child_prims(
        orig_path,
        predicate=lambda p: p.HasAPI(UsdPhysics.RigidBodyAPI)
    )
    if not rigidbody_prims:
        raise RuntimeError(f"No RigidBodyAPI prim found under {orig_path}")
    rigidbody_prim = rigidbody_prims[0]

    # 4) copy the rigid-body prim to a temp path
    temp_path = Sdf.Path("/__temp__")
    Sdf.CopySpec(layer, rigidbody_prim.GetPath(), layer, temp_path)
    # strip its children
    for child in list(stage.GetPrimAtPath(temp_path).GetChildren()):
        stage.RemovePrim(child.GetPath())

    # 5) copy meshes under the temp prim
    for mesh in mesh_prims:
        dst = temp_path.AppendChild(mesh.GetName())
        Sdf.CopySpec(layer, mesh.GetPath(), layer, dst)

    # 6) remove the old original subtree
    stage.RemovePrim(orig_path)

    # 7) rename /__temp__ → /<orig_name>
    final_path = Sdf.Path(f"/{orig_name}")
    Sdf.CopySpec(layer, temp_path, layer, final_path)
    stage.SetDefaultPrim(stage.GetPrimAtPath(final_path))
    stage.RemovePrim(temp_path)

    # ensure output directory exists
    os.makedirs(os.path.dirname(output_usd), exist_ok=True)
    # export
    layer.Export(output_usd)
    print(f"Processed {input_usd} → {output_usd}")


# for root, _, files in os.walk(args_cli.input):
#     for fname in files:
#         if not fname.lower().endswith(".usd"):
#             continue
#         in_path = os.path.join(root, fname)
#         rel = os.path.relpath(in_path, args_cli.input)
#         out_path = os.path.join(args_cli.output, rel)
#         process_usd(in_path, out_path)

process_usd("/home/octipus/data/Props/Dextrah/table.usd", "source/isaaclab_assets/data/Props/Dextrah/table.usd")