# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""NVBug 7071342 standalone probe: Valiant Gaussian-splat capture in Isaac Sim RTX, without Isaac Lab.

Every iteration builds a fresh stage that references Isaac Sim's public Valiant sample, freezes its camera rig at the
first time sample, creates one tiled render product on the sample's PPISP camera with HdrColor, LdrColor and
distance_to_image_plane annotators, renders until all three outputs are populated, classifies the capture, renders
--extra-frames more frames and classifies again, then destroys the render product. Isaac Lab's rendering test builds
one scene per test case the same way inside one Kit process.

    <isaac-sim>/python.sh valiant_teal_probe.py --iterations 100 --out out

--profile isaaclab (default) launches Kit with the renderer settings of Isaac Lab's headless camera experience;
--profile isaacsim uses Isaac Sim's defaults. The script exits with status 1 when any capture is classified as teal.
"""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--iterations", type=int, default=50)
parser.add_argument("--out", default="valiant-teal-probe-out")
parser.add_argument("--profile", choices=("isaaclab", "isaacsim"), default="isaaclab")
parser.add_argument("--extra-frames", type=int, default=5, help="frames rendered after the first populated capture")
parser.add_argument("--max-warmup-frames", type=int, default=20)
parser.add_argument("--resolution", default="128x96")
parser.add_argument("--no-exposure-overrides", action="store_true", help="skip Isaac Lab's PPISP exposure overrides")
parser.add_argument("--skip-tonemapping", action="store_true", help="leave Gaussian skip-tonemapping enabled")
parser.add_argument(
    "--usd",
    default="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2/Isaac/Samples/"
    "Scene_ParticleField/valiant_auto.usdz",
)
parser.add_argument("--camera", default="rig_trajectories/sensor_rig_0/camera1_ppisp")
parser.add_argument("--rig", default="rig_trajectories/sensor_rig_0")
args = parser.parse_args()

# Settings from Isaac Lab's apps/isaaclab.python.headless.rendering.kit that touch rendering.
ISAACLAB_SETTINGS = [
    "--/app/useFabricSceneDelegate=true",
    "--/rtx/hydra/readTransformsFromFabricInRenderDelegate=true",
    "--/renderer/scenePartitioning/enabled=true",
    "--/rtx/translucency/enabled=true",
    "--/rtx/reflections/enabled=true",
    "--/rtx/indirectDiffuse/enabled=true",
    "--/rtx-transient/dlssg/enabled=false",
    "--/rtx/directLighting/sampledLighting/enabled=true",
    "--/rtx/directLighting/sampledLighting/samplesPerPixel=1",
    "--/rtx/sceneDb/ambientLightIntensity=1.0",
    "--/rtx/shadows/enabled=true",
    "--/rtx/rtpt/maxBounces=3",
    "--/rtx/rtpt/cached/enabled=false",
    "--/rtx/rtpt/lightcache/cached/enabled=false",
    "--/rtx/rtpt/translucency/virtualMotion/enabled=false",
    "--/rtx/rtpt/splitRoughReflection=true",
    "--/rtx/rtpt/adaptiveSampling/disocclusion/enabled=true",
    "--/rtx/rtpt/adaptiveSampling/disocclusion/spp=4",
    "--/rtx/pathtracing/maxSamplesPerLaunch=1000000",
    "--/rtx/viewTile/limit=1000000",
    "--/rtx/raytracing/cached/enabled=true",
    "--/rtx/ambientOcclusion/enabled=true",
    "--/rtx/ambientOcclusion/denoiserMode=0",
    "--/rtx/raytracing/subpixel/mode=1",
    "--/rtx/post/dlss/execMode=2",
    "--/omni/replicator/asyncRendering=false",
    "--/app/updateOrder/checkForHydraRenderComplete=1000",
    "--/app/renderer/waitIdle=true",
    "--/app/hydraEngine/waitIdle=true",
    "--/omni/rtx/nre/compositing/rendererHints=3",
    "--/exts/omni.replicator.core/Orchestrator/enabled=false",
]
SYNC_SETTINGS = [
    "--/omni/replicator/asyncRendering=false",
    "--/app/renderer/waitIdle=true",
    "--/app/hydraEngine/waitIdle=true",
]

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp(
    {
        "headless": True,
        "renderer": "RealTimePathTracing",
        "extra_args": ISAACLAB_SETTINGS if args.profile == "isaaclab" else SYNC_SETTINGS,
    }
)

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.kit.app  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Sdf, Usd, UsdGeom  # noqa: E402

omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.replicator.core", True)
import omni.replicator.core as rep  # noqa: E402

WIDTH, HEIGHT = (int(v) for v in args.resolution.split("x"))
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
settings = carb.settings.get_settings()
kit = omni.kit.app.get_app()
rep.AnnotatorRegistry.register_annotator_from_aov(aov="HdrColor", output_data_type=np.float32, output_channels=4)


def apply_exposure_overrides(prim: Usd.Prim) -> None:
    """Isaac Lab's isaaclab_ppisp.apply_rtx_exposure_overrides: neutral RTX exposure when PPISP owns processing."""
    prim.AddAppliedSchema("OmniRtxCameraAutoExposureAPI_1")
    prim.AddAppliedSchema("OmniRtxCameraExposureAPI_1")
    for name, value in (
        ("exposure", 0.0),
        ("exposure:fStop", 1.0),
        ("exposure:iso", 0.0),
        ("exposure:responsivity", 1.0),
        ("exposure:time", 1.0),
    ):
        prim.CreateAttribute(name, Sdf.ValueTypeNames.Float, custom=False).Set(value)
    prim.CreateAttribute("omni:rtx:autoExposure:enabled", Sdf.ValueTypeNames.Bool, custom=False).Set(False)


def ldr_stats(rgba: np.ndarray) -> dict:
    rgb = rgba[..., :3].astype(np.float32)
    mean = rgb.reshape(-1, 3).mean(0)
    std = rgb.reshape(-1, 3).std(0)
    flat = bool(std.mean() < 30.0)
    return {
        "mean": [round(float(v), 1) for v in mean],
        "std": [round(float(v), 1) for v in std],
        "flat": flat,
        "teal": bool(flat and mean[1] > mean[0] + 20.0),
        "zero": bool(not rgb.any()),
    }


def hdr_stats(hdr: np.ndarray) -> dict:
    rgb = hdr[..., :3].astype(np.float64)
    finite = np.isfinite(rgb)
    values = np.where(finite, rgb, 0.0)
    luma = values @ np.array([0.2126, 0.7152, 0.0722])
    mean = values.reshape(-1, 3).mean(0)
    contrast = float(luma.std() / max(luma.mean(), 1e-12))
    return {
        "mean": [float(f"{v:.5g}") for v in mean],
        "luma_contrast": round(contrast, 4),
        "nonfinite": int((~finite).sum()),
        "negative": int((values < 0).sum()),
        "greenish": bool(mean[1] > 1.2 * mean[0]),
    }


def depth_stats(depth: np.ndarray) -> dict:
    d = depth.astype(np.float64)
    valid = np.isfinite(d) & (d > 0)
    return {
        "valid_frac": round(float(valid.mean()), 4),
        "median": float(f"{np.median(d[valid]):.5g}") if valid.any() else None,
    }


def save_png(array: np.ndarray, path: Path) -> None:
    Image.fromarray(np.ascontiguousarray(array[..., :3]).astype(np.uint8)).save(path)


def save_hdr_png(hdr: np.ndarray, path: Path) -> None:
    rgb = np.nan_to_num(hdr[..., :3].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0).clip(min=0.0)
    save_png((255.0 * (rgb / (1.0 + rgb)) ** (1.0 / 2.2)).clip(0, 255), path)


def capture(annotators: dict) -> dict:
    return {name: np.asarray(annotator.get_data()) for name, annotator in annotators.items()}


def render() -> None:
    # Isaac Lab disables the Replicator orchestrator, so app updates deliver annotator data; with Isaac Sim's
    # defaults the orchestrator owns captures.
    if args.profile == "isaaclab":
        kit.update()
    else:
        rep.orchestrator.step()


def populated(data: dict) -> bool:
    return all(array.size and np.count_nonzero(array) for array in data.values())


records = []
context = omni.usd.get_context()
started = time.time()
for iteration in range(args.iterations):
    t0 = time.time()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.Xform.Define(stage, "/World")
    scene = stage.DefinePrim("/World/envs/env_0/Scene", "Xform")
    scene.GetReferences().AddReference(args.usd)
    while context.get_stage_loading_status()[2]:
        kit.update()

    rig = stage.GetPrimAtPath(f"/World/envs/env_0/Scene/{args.rig}")
    if rig.IsValid():
        (op,) = UsdGeom.Xformable(rig).GetOrderedXformOps()
        op.GetAttr().Set(op.GetAttr().Get(Usd.TimeCode(0.0)))
    camera_path = f"/World/envs/env_0/Scene/{args.camera}"
    camera = stage.GetPrimAtPath(camera_path)
    if not camera.IsValid():
        sys.exit(f"camera prim {camera_path} not found")
    if not args.no_exposure_overrides:
        apply_exposure_overrides(camera)
    if not args.skip_tonemapping:
        settings.set_bool("/rtx/rtpt/gaussian/skipTonemapping/enabled", False)

    name = f"rp_{uuid.uuid4().hex}"
    if hasattr(rep.create, "render_product_tiled"):
        product = rep.create.render_product_tiled(cameras=[camera_path], tile_resolution=(WIDTH, HEIGHT), name=name)
    else:
        product = rep.create.render_product(camera_path, (WIDTH, HEIGHT), name=name)
    if not args.skip_tonemapping:
        product_prim = stage.GetPrimAtPath(product.path)
        product_prim.CreateAttribute("omni:rtx:gaussian:skipTonemapping:enabled", Sdf.ValueTypeNames.Bool).Set(False)
    annotators = {
        "hdr": rep.AnnotatorRegistry.get_annotator("HdrColor"),
        "ldr": rep.AnnotatorRegistry.get_annotator("rgb"),
        "depth": rep.AnnotatorRegistry.get_annotator("distance_to_image_plane"),
    }
    for annotator in annotators.values():
        annotator.attach([product.path])

    warmup = 0
    data = capture(annotators)
    while not populated(data) and warmup < args.max_warmup_frames:
        render()
        warmup += 1
        data = capture(annotators)
    if not populated(data):
        sys.exit(f"iteration {iteration}: outputs still empty after {warmup} frames: { {k: v.shape for k, v in data.items()} }")
    first = {"ldr": ldr_stats(data["ldr"]), "hdr": hdr_stats(data["hdr"]), "depth": depth_stats(data["depth"])}
    first_data = data
    for _ in range(args.extra_frames):
        render()
    data = capture(annotators)
    later = {"ldr": ldr_stats(data["ldr"]), "hdr": hdr_stats(data["hdr"]), "depth": depth_stats(data["depth"])}

    tag = f"{iteration:04d}"
    save_png(first_data["ldr"], OUT / f"{tag}-first-ldr.png")
    save_hdr_png(first_data["hdr"], OUT / f"{tag}-first-hdr.png")
    if first["ldr"]["teal"] or later["ldr"]["teal"]:
        np.save(OUT / f"{tag}-first-hdr.npy", first_data["hdr"])
        save_png(data["ldr"], OUT / f"{tag}-later-ldr.png")
    record = {
        "iteration": iteration,
        "warmup_frames": warmup,
        "first": first,
        "later": later,
        "seconds": round(time.time() - t0, 2),
    }
    records.append(record)
    print(
        f"[probe] {tag} warmup={warmup} first_ldr_mean={first['ldr']['mean']} std={first['ldr']['std']}"
        f" teal={first['ldr']['teal']} later_teal={later['ldr']['teal']} hdr_greenish={first['hdr']['greenish']}"
        f" depth_valid={first['depth']['valid_frac']} t={record['seconds']}s",
        flush=True,
    )

    for annotator in annotators.values():
        annotator.detach([product.path])
    product.destroy()
    kit.update()

teal_first = sum(r["first"]["ldr"]["teal"] for r in records)
teal_later = sum(r["later"]["ldr"]["teal"] for r in records)
summary = {
    "kit": kit.get_kit_version(),
    "profile": args.profile,
    "iterations": len(records),
    "teal_first": teal_first,
    "teal_later": teal_later,
    "exposure_overrides": not args.no_exposure_overrides,
    "skip_tonemapping": args.skip_tonemapping,
    "extra_frames": args.extra_frames,
    "resolution": [WIDTH, HEIGHT],
    "wall_seconds": round(time.time() - started, 1),
    "gpu": os.environ.get("PROBE_GPU_NAME", ""),
    "records": records,
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
print(
    f"[probe] Kit {summary['kit']} profile={args.profile}: teal first captures {teal_first}/{len(records)},"
    f" teal after {args.extra_frames} more frames {teal_later}/{len(records)}",
    flush=True,
)
app.close(exit_code=1 if teal_first or teal_later else 0)
