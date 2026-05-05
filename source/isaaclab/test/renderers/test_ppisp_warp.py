# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import torch

from pxr import Sdf, Usd, UsdGeom, UsdShade

from isaaclab.renderers.ppisp import normalize_ppisp_cfg, parse_render_product
from isaaclab.renderers.ppisp_warp import apply_ppisp_to_rgba
from isaaclab.sensors.camera.camera import Camera
from isaaclab.sensors.camera.camera_data import CameraData


def test_ppisp_warp_exposure_increases_ldr_output():
    hdr_color = torch.full((1, 4, 4, 3), 0.25, dtype=torch.float32)
    baseline = torch.zeros((1, 4, 4, 4), dtype=torch.uint8)
    exposed = torch.zeros_like(baseline)

    apply_ppisp_to_rgba(hdr_color, baseline, normalize_ppisp_cfg({"inputs": {"exposureOffset": 0.0}}))
    apply_ppisp_to_rgba(hdr_color, exposed, normalize_ppisp_cfg({"inputs": {"exposureOffset": 1.0}}))

    assert torch.all(baseline[..., 3] == 255)
    assert torch.all(exposed[..., 3] == 255)
    assert exposed[..., :3].float().mean() > baseline[..., :3].float().mean()


def test_camera_wrapper_applies_parsed_ppisp_from_render_product():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Camera.Define(stage, "/World/Camera")
    stage.DefinePrim("/Render", "Scope")
    render_product = stage.DefinePrim("/Render/RenderProduct", "RenderProduct")
    render_product.CreateRelationship("camera").SetTargets([Sdf.Path("/World/Camera")])
    shader = UsdShade.Shader.Define(stage, "/Render/RenderProduct/PPISP")
    shader.CreateInput("exposureOffset", Sdf.ValueTypeNames.Float).Set(1.0)

    ppisp_cfg = parse_render_product(stage, "/Render/RenderProduct").ppisp
    assert ppisp_cfg is not None
    hdr_color = torch.full((1, 4, 4, 3), 0.25, dtype=torch.float32)
    baseline = torch.zeros((1, 4, 4, 4), dtype=torch.uint8)
    rgba = torch.zeros((1, 4, 4, 4), dtype=torch.uint8)
    apply_ppisp_to_rgba(hdr_color, baseline, normalize_ppisp_cfg({"inputs": {"exposureOffset": 0.0}}))

    camera = SimpleNamespace(
        cfg=SimpleNamespace(ppisp=ppisp_cfg),
        _renderer_output_data={"rgb_hdr": hdr_color},
        _data=CameraData(output={"rgba": rgba, "rgb": rgba[..., :3]}),
    )

    Camera._apply_ppisp_if_needed(camera)

    assert torch.all(camera._data.output["rgba"][..., 3] == 255)
    assert torch.all(camera._data.output["rgb"] == camera._data.output["rgba"][..., :3])
    assert camera._data.output["rgb"].float().mean() > baseline[..., :3].float().mean()
