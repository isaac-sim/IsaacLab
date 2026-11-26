# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Temporary NVBug 7071342 diagnostics for the Valiant Isaac RTX teal frame. Do not merge.

Each iteration builds the Valiant scene the way ``run_rendering_case`` does, with RGB, HDR and depth on one camera,
classifies the first populated capture, then attaches RTX's own ``LdrColor`` and re-captures after 1, 5 and 30 more
renders. Teal iterations (and the first clean ones, as references) save every image plus a settings snapshot.
"""

from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(enable_cameras=True)

import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from rendering_runtime import build_rendering_scene  # noqa: E402
from rendering_scene_cfgs import make_rendering_scene_spec  # noqa: E402

from isaaclab_physx.renderers import IsaacRtxRendererCfg  # noqa: E402

from isaaclab.renderers import RenderBufferKind  # noqa: E402
from isaaclab.utils.seed import configure_seed  # noqa: E402

pytestmark = pytest.mark.isaacsim_ci

_OUT = Path.cwd() / "tests" / "comparison-images" / "teal-diagnostics"
_ITERATIONS = int(os.environ.get("TEAL_PROBE_ITERATIONS", "30"))
_CLEAN_REFERENCES = 3
_AOVS = (RenderBufferKind.RGB, RenderBufferKind.RGB_HDR, RenderBufferKind.DISTANCE_TO_IMAGE_PLANE)
_SETTING_ROOTS = (
    "/rtx/rtpt/gaussian",
    "/rtx/rendermode",
    "/rtx/sdg",
    "/rtx/post/tonemap",
    "/rtx/post/histogram",
    "/rtx/post/dlss",
    "/rtx/post/aa",
    "/omni/rtx/nre",
    "/rtx/nre",
    "/rtx/hydra",
    "/rtx-transient/hydra",
    "/rtx/materialDb",
    "/rtx-transient/psoDb",
    "/renderer",
    "/app/renderer",
    "/app/hydraEngine",
)
_SUMMARY: list[dict] = []


def _ldr_stats(rgb: np.ndarray) -> dict:
    flat = rgb[..., :3].reshape(-1, 3).astype(np.float32)
    mean, std = flat.mean(0), flat.std(0)
    return {
        "mean": [round(float(v), 1) for v in mean],
        "std": [round(float(v), 1) for v in std],
        "teal": bool(std.mean() < 30.0 and mean[1] > mean[0] + 20.0),
    }


def _hdr_stats(hdr: np.ndarray) -> dict:
    rgb = np.nan_to_num(hdr[..., :3].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    luma = rgb @ np.array([0.2126, 0.7152, 0.0722])
    mean = rgb.reshape(-1, 3).mean(0)
    return {
        "mean": [float(f"{v:.5g}") for v in mean],
        "luma_contrast": round(float(luma.std() / max(luma.mean(), 1e-12)), 4),
        "nonfinite": int((~np.isfinite(hdr[..., :3])).sum()),
        "greenish": bool(mean[1] > 1.2 * mean[0]),
    }


def _depth_stats(depth: np.ndarray) -> dict:
    d = depth.astype(np.float64)
    valid = np.isfinite(d) & (d > 0)
    return {"valid_frac": round(float(valid.mean()), 4), "median": float(np.median(d[valid])) if valid.any() else None}


def _snapshot(outputs: dict[str, torch.Tensor], ldr: np.ndarray | None) -> dict:
    stats = {
        "rgb": _ldr_stats(outputs[RenderBufferKind.RGB][0].cpu().numpy()),
        "rgb_hdr": _hdr_stats(outputs[RenderBufferKind.RGB_HDR][0].cpu().numpy()),
        "depth": _depth_stats(outputs[RenderBufferKind.DISTANCE_TO_IMAGE_PLANE][0].cpu().numpy()),
    }
    if ldr is not None:
        stats["ldr_color"] = _ldr_stats(ldr)
    return stats


def _save(prefix: str, outputs: dict[str, torch.Tensor], ldr: np.ndarray | None) -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(outputs[RenderBufferKind.RGB][0, ..., :3].cpu().numpy().astype(np.uint8)).save(f"{prefix}-rgb.png")
    hdr = outputs[RenderBufferKind.RGB_HDR][0, ..., :3].cpu().numpy().astype(np.float32)
    np.save(f"{prefix}-rgb_hdr.npy", hdr)
    tone = np.nan_to_num(hdr, nan=0.0, posinf=0.0, neginf=0.0).clip(min=0.0)
    Image.fromarray((255.0 * (tone / (1.0 + tone)) ** (1.0 / 2.2)).clip(0, 255).astype(np.uint8)).save(
        f"{prefix}-rgb_hdr.png"
    )
    depth = outputs[RenderBufferKind.DISTANCE_TO_IMAGE_PLANE][0, ..., 0].cpu().numpy().astype(np.float32)
    np.save(f"{prefix}-depth.npy", depth)
    if ldr is not None:
        Image.fromarray(np.ascontiguousarray(ldr[..., :3]).astype(np.uint8)).save(f"{prefix}-ldr_color.png")


def _state_dump(runtime) -> dict:
    import carb

    settings = carb.settings.get_settings()
    stage = runtime.sim.stage
    camera_path = runtime.camera.cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
    camera = stage.GetPrimAtPath(camera_path)
    render_product = stage.GetPrimAtPath(runtime.camera._render_data.render_product.path)
    return {
        "settings": {root: settings.get(root) for root in _SETTING_ROOTS},
        "camera_applied_schemas": list(camera.GetAppliedSchemas()) if camera else None,
        "camera_attributes": {
            a.GetName(): str(a.Get())
            for a in (camera.GetAttributes() if camera else [])
            if a.GetName().startswith(("exposure", "omni:rtx", "ppisp"))
        },
        "render_product_attributes": {
            a.GetName(): str(a.Get())
            for a in (render_product.GetAttributes() if render_product else [])
            if a.GetName().startswith("omni:rtx")
        },
    }


@pytest.mark.parametrize("iteration", range(_ITERATIONS))
def test_valiant_teal_diagnostics(iteration: int, request: pytest.FixtureRequest) -> None:
    import omni.replicator.core as rep

    configure_seed(42, torch_deterministic=True)
    spec = make_rendering_scene_spec("gaussian_splats_valiant_1env", "physx")
    spec.cfg.camera.renderer_cfg = IsaacRtxRendererCfg()
    with build_rendering_scene(
        spec.cfg,
        "physx",
        data_types=_AOVS,
        physics_cfg=spec.physics_cfg,
        dt=spec.dt,
        on_scene_created=spec.on_scene_created,
    ) as runtime:
        if spec.after_reset is not None:
            spec.after_reset(runtime)
        outputs, _ = runtime.camera_outputs()
        waits = 0
        while waits < 10 and not all(output.count_nonzero() for output in outputs.values()):
            runtime.render_camera()
            outputs, _ = runtime.camera_outputs()
            waits += 1
        first_outputs = outputs
        record = {"iteration": iteration, "waits": waits, "first": _snapshot(outputs, None)}

        ldr_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        ldr_annotator.attach([runtime.camera._render_data.render_product.path])
        rendered = 0
        for target in (1, 5, 30):
            while rendered < target:
                runtime.render_camera()
                rendered += 1
            outputs, _ = runtime.camera_outputs()
            record[f"after_{target}"] = _snapshot(outputs, np.asarray(ldr_annotator.get_data()))
        ldr = np.asarray(ldr_annotator.get_data())

        teal = record["first"]["rgb"]["teal"]
        clean_saved = sum(1 for r in _SUMMARY if r.get("saved") and not r["first"]["rgb"]["teal"])
        if teal or clean_saved < _CLEAN_REFERENCES:
            prefix = str(_OUT / f"iter{iteration:03d}-{'teal' if teal else 'clean'}")
            _save(f"{prefix}-first", first_outputs, None)
            _save(f"{prefix}-after30", outputs, ldr)
            record["state"] = _state_dump(runtime)
            record["saved"] = True
            (_OUT / f"iter{iteration:03d}.json").write_text(json.dumps(record, indent=1, default=str))
        ldr_annotator.detach([runtime.camera._render_data.render_product.path])

    _SUMMARY.append(record)
    request.node.user_properties.extend(
        [
            ("teal_first", str(teal)),
            ("teal_after_1", str(record["after_1"]["rgb"]["teal"])),
            ("teal_after_30", str(record["after_30"]["rgb"]["teal"])),
            ("ldr_color_teal_after_1", str(record["after_1"]["ldr_color"]["teal"])),
            ("first_rgb_mean", str(record["first"]["rgb"]["mean"])),
            ("first_hdr_greenish", str(record["first"]["rgb_hdr"]["greenish"])),
            ("first_depth_valid", str(record["first"]["depth"]["valid_frac"])),
            ("waits", str(waits)),
        ]
    )
    _OUT.mkdir(parents=True, exist_ok=True)
    (_OUT / "summary.json").write_text(json.dumps(_SUMMARY, indent=1, default=str))
