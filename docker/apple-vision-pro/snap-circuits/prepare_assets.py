# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert and arrange the Snap Circuits demo assets into a table-top USD."""

"""Launch Isaac Sim Simulator first."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source-dir", type=Path, required=True, help="Unpacked source asset directory.")
parser.add_argument("--output-dir", type=Path, required=True, help="Directory for converted assets and scene USD.")
parser.add_argument(
    "--s3-scale",
    type=float,
    default=1.0,
    help="Scale applied to private-S3 assets. Downloaded ZIP meshes always use 0.001 (mm to m).",
)
parser.add_argument(
    "--asset-set",
    choices=("demo", "catalog"),
    default="demo",
    help="Use the curated demo bench (default) or every discovered catalog asset.",
)
parser.add_argument(
    "--piece-scale",
    type=float,
    default=1.5,
    help="Uniform scale applied to each prepared object without changing its table position (default: 1.5).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import re

from pxr import Gf, Usd, UsdGeom

from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg

_MESH_PRIORITY = {".obj": 0, ".stl": 1, ".fbx": 2}
_USD_SUFFIXES = {".usd", ".usda", ".usdc"}
_USD_DEPENDENCY_DIRECTORIES = {"configuration", "materials", "payloads", "textures", ".thumbs"}
_DEMO_ASSET_STEMS = {
    # Authoritative S3 parts.
    "dl_snapcircuit_basegrid_02",
    "dl_snapcircuit_batteryholder_loaded_02",
    "dl_snapcircuit_lamp_02",
    "dl_snapcircuit_led_02",
    "dl_snapcircuit_motor_fan_02",
    "dl_snapcircuit_slideswitch_02",
    "dl_snapcircuit_wire_1_02",
    "dl_snapcircuit_wire_2_02",
    "dl_snapcircuit_wire_3_02",
    "dl_snapcircuit_wire_4_02",
    # Useful SC-100 parts not represented in the S3 set.
    "08_photoresistor",
    "10_press_switch",
    "12_speaker",
    "14_music_ic",
    "16_alarm_ic",
    # The two explicitly requested labware objects.
    "test_tube_16x150mm_round_bottom",
    "test_tube_rack_4x6_18mm",
}


def _slug(path: Path, source_dir: Path) -> str:
    relative = path.relative_to(source_dir).with_suffix("")
    return re.sub(r"[^a-zA-Z0-9_]+", "_", "_".join(relative.parts)).strip("_").lower()


def _select_sources(source_dir: Path) -> list[Path]:
    """Select one preferred representation per physical asset."""
    candidates: dict[tuple[Path, str], Path] = {}
    usd_assets: dict[tuple[Path, str], Path] = {}
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in _USD_SUFFIXES:
            if not _USD_DEPENDENCY_DIRECTORIES.intersection(part.lower() for part in path.parts):
                # The private prefix contains original and newer ``_02``
                # revisions of the same part. Prefer the newer revision while
                # keeping standalone assets that do not have one.
                logical_stem = re.sub(r"_02$", "", path.stem.lower())
                key = (path.parent, logical_stem)
                current = usd_assets.get(key)
                if current is None or (path.stem.lower().endswith("_02") and not current.stem.lower().endswith("_02")):
                    usd_assets[key] = path
            continue
        if suffix not in _MESH_PRIORITY:
            continue
        # The SC-100 archive stores identical assets in sibling obj/stl/glb
        # directories. Collapse those representation directories before
        # choosing the preferred input so each physical component appears once.
        key_parent = path.parent.parent if path.parent.name.lower() in {"obj", "stl", "fbx"} else path.parent
        key = (key_parent, path.stem.lower())
        current = candidates.get(key)
        if current is None or _MESH_PRIORITY[suffix] < _MESH_PRIORITY[current.suffix.lower()]:
            candidates[key] = path

    selected = list(candidates.values())
    selected.extend(usd_assets.values())
    return sorted(selected)


def _is_static(path: Path) -> bool:
    name = path.stem.lower()
    return "base_grid" in name or "basegrid" in name or "baseplate" in name or "rack" in name


def _is_zip_bundle_asset(path: Path) -> bool:
    path_text = path.as_posix().lower()
    return "sc100_mesh_bundle" in path_text or "test_tube" in path_text


def _is_demo_asset(path: Path) -> bool:
    return path.stem.lower() in _DEMO_ASSET_STEMS


def _convert_mesh(path: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    static = _is_static(path)
    mass = None if static else (0.01 if "tube" in path.stem.lower() else 0.02)
    cfg = MeshConverterCfg(
        asset_path=str(path),
        force_usd_conversion=True,
        usd_dir=str(destination.parent),
        usd_file_name=destination.name,
        make_instanceable=True,
        collision_props=schemas_cfg.CollisionPropertiesCfg(collision_enabled=True),
        mesh_collision_props=schemas_cfg.ConvexHullPropertiesCfg(),
        mass_props=schemas_cfg.MassPropertiesCfg(mass=mass) if mass is not None else None,
        rigid_props=schemas_cfg.RigidBodyPropertiesCfg() if mass is not None else None,
    )
    return Path(MeshConverter(cfg).usd_path)


def _layout_positions(count: int) -> list[tuple[float, float, float]]:
    """Generate collision-free rows across the packing-table work surface."""
    positions = []
    columns = 8
    for index in range(count):
        row, column = divmod(index, columns)
        positions.append((-0.38 + 0.105 * column, -0.22 + 0.105 * row, 0.025))
    return positions


def _write_scene(references: list[tuple[Path, float, bool]], output_path: Path, piece_scale: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/SnapCircuitsTable")
    stage.SetDefaultPrim(root.GetPrim())

    dynamic_positions = _layout_positions(sum(not static for _, _, static in references))
    dynamic_index = 0
    for index, (asset_path, scale, static) in enumerate(references):
        name = re.sub(r"[^a-zA-Z0-9_]", "_", asset_path.stem)
        xform = UsdGeom.Xform.Define(stage, f"/SnapCircuitsTable/asset_{index:03d}_{name}")
        relative_asset = os.path.relpath(asset_path, output_path.parent)
        source = UsdGeom.Xform.Define(stage, f"{xform.GetPrim().GetPath()}/source")
        source.GetPrim().GetReferences().AddReference(relative_asset)

        asset_name = asset_path.stem.lower()
        if "base_grid" in asset_name or "basegrid" in asset_name:
            position = (-0.18, 0.02, 0.0)
        elif "baseplate" in asset_name:
            position = (0.24, 0.02, 0.0)
        elif "rack" in asset_path.stem.lower():
            position = (0.38, 0.12, 0.0)
        else:
            position = dynamic_positions[dynamic_index]
            dynamic_index += 1

        xform.AddTranslateOp().Set(Gf.Vec3d(*position))
        final_scale = scale * piece_scale
        xform.AddScaleOp().Set(Gf.Vec3f(final_scale, final_scale, final_scale))

    stage.GetRootLayer().Save()


def main() -> None:
    source_dir = args_cli.source_dir.resolve()
    output_dir = args_cli.output_dir.resolve()
    converted_dir = output_dir / "converted"
    selected_sources = _select_sources(source_dir)
    if args_cli.asset_set == "demo":
        selected_sources = [path for path in selected_sources if _is_demo_asset(path)]
    if not selected_sources:
        raise RuntimeError(f"No supported OBJ, STL, FBX, or USD assets found under {source_dir}")

    references: list[tuple[Path, float, bool]] = []
    for path in selected_sources:
        scale = 0.001 if _is_zip_bundle_asset(path) else args_cli.s3_scale
        if path.suffix.lower() in _USD_SUFFIXES:
            asset_path = path
        else:
            asset_path = _convert_mesh(path, converted_dir / f"{_slug(path, source_dir)}.usd")
        references.append((asset_path, scale, _is_static(path)))
        print(f"Prepared {path} -> {asset_path} (scale={scale})")

    scene_path = output_dir / "snap_circuits_table.usda"
    _write_scene(references, scene_path, args_cli.piece_scale)
    print(f"Prepared {len(references)} {args_cli.asset_set} table assets: {scene_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
