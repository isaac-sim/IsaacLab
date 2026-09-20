# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-less tests for the renderer-agnostic OpenCV lens-distortion camera model.

Renderer-independent authoring and calibration are covered without a running simulation app:

* Authoring: :func:`isaaclab.sim.spawners.sensors.spawn_camera` with a ``distortion`` cfg authors the
  ``omni:lensdistortion:*`` USD API on the camera prim (which the RTX/OVRTX renderer round-trips
  through the USD export it loads).
* Readback: :meth:`isaaclab.sensors.camera.Camera._initialize_intrinsics` reconstructs
  ``camera.data.intrinsic_matrices`` from the authored ``fx/fy/cx/cy`` (which may be non-square or
  off-center) instead of assuming ``fx == fy`` and a centered principal point.

The renderer actually rendering *through* the distortion is a backend-specific concern covered in
``test_camera_opencv_distortion_ovrtx.py``.
"""

from __future__ import annotations

import importlib.util
import logging
from types import SimpleNamespace

import numpy as np
import pytest

_REQUIRED_MODULES = ("isaaclab", "pxr", "warp")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}",
    ),
]

if not _MISSING_MODULES:
    import torch
    import warp as wp

    from pxr import Gf, Sdf, Usd, UsdGeom

    import isaaclab.sim as sim_utils
    from isaaclab.sensors.camera.camera import Camera
    from isaaclab.sensors.camera.camera_data import CameraData
    from isaaclab.sim.spawners.sensors.sensors import spawn_camera
    from isaaclab.sim.spawners.sensors.sensors_cfg import (
        FisheyeCameraCfg,
        OpenCvFisheyeDistortionCfg,
        OpenCvPinholeDistortionCfg,
        PinholeCameraCfg,
    )
    from isaaclab.utils.warp import ProxyArray


# Real SO-101 wrist-camera calibration: exercises fx != fy and an off-center principal point, which
# stock Isaac Lab camera cfgs cannot express.
_PINHOLE_CALIB = dict(
    fx=339.26592887,
    fy=338.82010626,
    cx=323.55809091,
    cy=250.27360914,
    image_size=(640, 480),
    k1=0.07702322,
    k2=-0.13605453,
    k3=0.05163219,
    p1=-0.00024938,
    p2=-0.00175006,
)


"""
Authoring: spawn_camera with a distortion cfg.
"""


def _spawn_camera_on_new_stage(cfg, prim_path="/World/envs/env_0/Camera"):
    """Create a fresh in-memory stage and spawn a camera prim carrying ``cfg`` on it."""
    stage = sim_utils.create_new_stage()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/envs")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    # spawn_camera is @clone-decorated; call the undecorated function to spawn a single prim.
    prim = spawn_camera.__wrapped__(prim_path, cfg)
    return stage, prim


def test_pinhole_distortion_authors_opencv_api():
    """Spawning a pinhole camera with a distortion cfg authors the OpenCV pinhole USD API."""
    cfg = PinholeCameraCfg(distortion=OpenCvPinholeDistortionCfg(apply_lens_distortion=True, **_PINHOLE_CALIB))
    _stage, prim = _spawn_camera_on_new_stage(cfg)

    prefix = "omni:lensdistortion:opencvPinhole"
    assert prim.GetAttribute("omni:lensdistortion:model").Get() == "opencvPinhole"
    assert "OmniLensDistortionOpenCvPinholeAPI" in prim.GetMetadata("apiSchemas").GetAppliedItems()
    # intrinsics
    assert prim.GetAttribute(f"{prefix}:fx").Get() == pytest.approx(_PINHOLE_CALIB["fx"], abs=1e-3)
    assert prim.GetAttribute(f"{prefix}:fy").Get() == pytest.approx(_PINHOLE_CALIB["fy"], abs=1e-3)
    assert prim.GetAttribute(f"{prefix}:cx").Get() == pytest.approx(_PINHOLE_CALIB["cx"], abs=1e-3)
    assert prim.GetAttribute(f"{prefix}:cy").Get() == pytest.approx(_PINHOLE_CALIB["cy"], abs=1e-3)
    assert tuple(prim.GetAttribute(f"{prefix}:imageSize").Get()) == (640, 480)
    # distortion coefficients (applied)
    assert prim.GetAttribute(f"{prefix}:k1").Get() == pytest.approx(_PINHOLE_CALIB["k1"], abs=1e-6)
    assert prim.GetAttribute(f"{prefix}:p2").Get() == pytest.approx(_PINHOLE_CALIB["p2"], abs=1e-6)
    # unused higher-order coefficients default to zero
    assert prim.GetAttribute(f"{prefix}:k6").Get() == pytest.approx(0.0)
    assert prim.GetAttribute(f"{prefix}:s4").Get() == pytest.approx(0.0)


def test_apply_lens_distortion_false_zeros_coefficients():
    """With ``apply_lens_distortion=False`` the intrinsics are kept but coefficients are muted to zero."""
    cfg = PinholeCameraCfg(distortion=OpenCvPinholeDistortionCfg(apply_lens_distortion=False, **_PINHOLE_CALIB))
    _stage, prim = _spawn_camera_on_new_stage(cfg)

    prefix = "omni:lensdistortion:opencvPinhole"
    # intrinsics are still authored
    assert prim.GetAttribute(f"{prefix}:fx").Get() == pytest.approx(_PINHOLE_CALIB["fx"], abs=1e-3)
    # coefficients are zeroed despite non-zero cfg values
    assert prim.GetAttribute(f"{prefix}:k1").Get() == pytest.approx(0.0)
    assert prim.GetAttribute(f"{prefix}:k2").Get() == pytest.approx(0.0)
    assert prim.GetAttribute(f"{prefix}:p1").Get() == pytest.approx(0.0)


def test_fisheye_distortion_authors_opencv_fisheye_api():
    """A distortion cfg with the fisheye model authors the OpenCV fisheye USD API."""
    cfg = FisheyeCameraCfg(
        distortion=OpenCvFisheyeDistortionCfg(
            fx=300.0, fy=300.0, cx=320.0, cy=240.0, image_size=(640, 480), k1=0.1, k2=0.02, k3=0.0, k4=0.0
        )
    )
    _stage, prim = _spawn_camera_on_new_stage(cfg)

    prefix = "omni:lensdistortion:opencvFisheye"
    assert prim.GetAttribute("omni:lensdistortion:model").Get() == "opencvFisheye"
    assert "OmniLensDistortionOpenCvFisheyeAPI" in prim.GetMetadata("apiSchemas").GetAppliedItems()
    assert prim.GetAttribute(f"{prefix}:fx").Get() == pytest.approx(300.0)
    assert prim.GetAttribute(f"{prefix}:k1").Get() == pytest.approx(0.1)
    assert prim.GetAttribute(f"{prefix}:k4").Get() == pytest.approx(0.0)


def test_distortion_attributes_survive_usd_export():
    """The authored OpenCV distortion API survives serialization, which the RTX/OVRTX path consumes."""
    cfg = PinholeCameraCfg(distortion=OpenCvPinholeDistortionCfg(apply_lens_distortion=True, **_PINHOLE_CALIB))
    stage, _prim = _spawn_camera_on_new_stage(cfg)

    exported = stage.ExportToString()
    assert "OmniLensDistortionOpenCvPinholeAPI" in exported
    assert "omni:lensdistortion:opencvPinhole:fx" in exported
    assert "omni:lensdistortion:model" in exported


def test_camera_without_distortion_authors_no_opencv_api():
    """A camera cfg without a distortion model does not author any ``omni:lensdistortion`` attributes."""
    cfg = PinholeCameraCfg()
    _stage, prim = _spawn_camera_on_new_stage(cfg)

    assert not prim.GetAttribute("omni:lensdistortion:model").IsValid()
    applied = prim.GetMetadata("apiSchemas")
    applied_items = list(applied.GetAppliedItems()) if applied else []
    assert not any("LensDistortion" in item for item in applied_items)


"""
Readback: Camera._initialize_intrinsics with an authored distortion model.
"""


def _camera_prim_with_pinhole_distortion(fx, fy, cx, cy, width=640, height=480):
    """Build an in-memory ``UsdGeom.Camera`` carrying an OpenCV pinhole distortion model."""
    stage = Usd.Stage.CreateInMemory()
    cam = UsdGeom.Camera.Define(stage, "/Camera")
    prim = cam.GetPrim()
    prim.CreateAttribute("omni:lensdistortion:model", Sdf.ValueTypeNames.Token).Set("opencvPinhole")
    prefix = "omni:lensdistortion:opencvPinhole"
    for name, value in (("fx", fx), ("fy", fy), ("cx", cx), ("cy", cy)):
        prim.CreateAttribute(f"{prefix}:{name}", Sdf.ValueTypeNames.Float).Set(float(value))
    prim.CreateAttribute(f"{prefix}:imageSize", Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(width, height))
    return stage, cam


def _camera_for_prims(prims, width=640, height=480, device="cpu"):
    """Use real camera buffers and USD prims without creating a renderer."""
    fake = Camera.__new__(Camera)
    fake.stage = prims[0].GetPrim().GetStage()
    fake._sensor_prims = prims
    fake._device = device
    fake.cfg = SimpleNamespace(height=height, width=width)
    fake._view = SimpleNamespace(count=len(prims), close=lambda: None)
    fake._data = CameraData()
    fake._data.create_buffers(len(prims), device)
    fake._frame = ProxyArray(wp.zeros(len(prims), dtype=wp.int64, device=device))
    fake._ALL_INDICES = wp.array(np.arange(len(prims)), dtype=wp.int32, device=device)
    fake._ALL_ENV_MASK = wp.ones(len(prims), dtype=wp.bool, device=device)
    # attributes touched by ``__del__``/``_clear_callbacks`` when the fake object is garbage collected
    fake._initialize_handle = None
    fake._invalidate_initialize_handle = None
    fake._prim_deletion_handle = None
    fake._debug_vis_handle = None
    fake._render_data = SimpleNamespace(parameters=None)
    fake._renderer = SimpleNamespace(
        update_camera_intrinsics=lambda data, _matrices, parameters: setattr(data, "parameters", wp.clone(parameters)),
        cleanup=lambda _data: None,
    )
    fake._initialize_intrinsics()
    return fake


def _read_back_intrinsics(cam, width, height):
    """Read authored USD calibration into the camera's real Warp matrix buffer."""
    camera = _camera_for_prims([cam], width, height)
    return camera._data.intrinsic_matrices.warp.numpy()[0]


def test_readback_uses_authored_fx_fy_cx_cy():
    """The reconstructed intrinsic matrix reflects the authored, non-square, off-center calibration."""
    fx, fy, cx, cy = 339.26592887, 338.82010626, 323.55809091, 250.27360914
    width, height = 640, 480
    _stage, cam = _camera_prim_with_pinhole_distortion(fx, fy, cx, cy, width, height)

    k = _read_back_intrinsics(cam, width, height)

    assert k[0, 0] == pytest.approx(fx, abs=1e-2)
    assert k[1, 1] == pytest.approx(fy, abs=1e-2)
    assert k[0, 2] == pytest.approx(cx, abs=1e-2)
    assert k[1, 2] == pytest.approx(cy, abs=1e-2)
    assert k[2, 2] == pytest.approx(1.0)


def test_readback_preserves_non_square_and_off_center():
    """The readback must not collapse to the stock ``fx == fy`` / centered principal point."""
    fx, fy, cx, cy = 339.26592887, 338.82010626, 323.55809091, 250.27360914
    width, height = 640, 480
    _stage, cam = _camera_prim_with_pinhole_distortion(fx, fy, cx, cy, width, height)

    k = _read_back_intrinsics(cam, width, height)

    # non-square pixels are preserved (the old readback forced fx == fy)
    assert k[0, 0] != k[1, 1]
    # principal point is off-center (the old readback forced width/2, height/2)
    assert k[0, 2] != pytest.approx(width / 2)
    assert k[1, 2] != pytest.approx(height / 2)


def _camera_prim_with_model_token_only():
    """Build a camera prim that declares the distortion model token but authors no fx/fy/cx/cy."""
    stage = Usd.Stage.CreateInMemory()
    cam = UsdGeom.Camera.Define(stage, "/Camera")
    cam.GetPrim().CreateAttribute("omni:lensdistortion:model", Sdf.ValueTypeNames.Token).Set("opencvPinhole")
    return stage, cam


def test_readback_missing_intrinsics_falls_back_to_focal_length():
    """A model token without fx/fy/cx/cy falls back to the focal-length projection instead of raising."""
    width, height = 640, 480
    _stage, cam = _camera_prim_with_model_token_only()

    k = _read_back_intrinsics(cam, width, height)

    # focal-length/aperture projection: square pixels, centered principal point
    assert k[0, 0] == pytest.approx(k[1, 1])
    assert k[0, 2] == pytest.approx(width / 2)
    assert k[1, 2] == pytest.approx(height / 2)
    assert k[2, 2] == pytest.approx(1.0)


def test_readback_distinct_image_size_mismatches_each_warn():
    """Distinct authored image sizes across a camera's prims each warn, not only the first."""
    width, height = 320, 240
    _stage_a, cam_a = _camera_prim_with_pinhole_distortion(300.0, 300.0, 160.0, 120.0, 640, 480)
    _stage_b, cam_b = _camera_prim_with_pinhole_distortion(300.0, 300.0, 160.0, 120.0, 1280, 720)

    fake = _camera_for_prims([cam_a, cam_b], width, height)

    messages: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda record: messages.append(record.getMessage())
    cam_logger = logging.getLogger("isaaclab.sensors.camera.camera")
    cam_logger.addHandler(handler)
    try:
        Camera._initialize_intrinsics(fake)
    finally:
        cam_logger.removeHandler(handler)

    mismatch_warnings = [message for message in messages if "imageSize" in message]
    assert len(mismatch_warnings) == 2
    assert any("(640, 480)" in message for message in mismatch_warnings)
    assert any("(1280, 720)" in message for message in mismatch_warnings)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_set_intrinsic_matrices_skips_only_distortion_cameras_in_batch(device):
    """In a mixed batch only the distortion camera is skipped (with a warning); a plain camera is updated.

    The authored ``omni:lensdistortion:*`` fx/fy/cx/cy are the readback's source of truth, so the
    focal-length/aperture write would be discarded for a distortion camera. Skipping the whole call would
    also drop ordinary selected cameras; only the distortion entries must be left untouched.
    """
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA is unavailable")
    width, height = 640, 480
    _stage_d, distortion_cam = _camera_prim_with_pinhole_distortion(339.0, 338.0, 323.0, 250.0, width, height)
    plain_stage = Usd.Stage.CreateInMemory()
    plain_cam = UsdGeom.Camera.Define(plain_stage, "/PlainCamera")

    fake = _camera_for_prims([distortion_cam, plain_cam], width, height, device)

    messages: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda record: messages.append(record.getMessage())
    cam_logger = logging.getLogger("isaaclab.sensors.camera.camera")
    cam_logger.addHandler(handler)
    # row 0 targets the distortion camera (skipped); row 1 recalibrates the plain camera to fx = fy = 500
    requested = torch.tensor(
        [
            [[999.0, 0.0, 111.0], [0.0, 999.0, 222.0], [0.0, 0.0, 1.0]],
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
        ],
        dtype=torch.float32,
        device=device,
    )
    try:
        Camera.set_intrinsic_matrices(fake, requested, env_ids=[0, 1])
    finally:
        cam_logger.removeHandler(handler)

    k = fake._data.intrinsic_matrices.warp.numpy()
    # the distortion camera keeps its authored calibration (skipped, request ignored)
    assert k[0, 0, 0] == pytest.approx(339.0, abs=1e-2)
    # the plain camera reflects the requested focal length (updated, not over-skipped)
    assert k[1, 0, 0] == pytest.approx(500.0, abs=1e-2)
    assert any("skipped" in message.lower() for message in messages)


@pytest.fixture(params=["cpu", "cuda:0"])
def intrinsic_camera(request):
    """Three independent camera prims, with buffers on the selected device."""
    if request.param.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA is unavailable")
    stage = Usd.Stage.CreateInMemory()
    prims = [UsdGeom.Camera.Define(stage, f"/Camera_{i}") for i in range(3)]
    camera = _camera_for_prims(prims, device=request.param)
    return stage, camera


@pytest.mark.parametrize("env_ids", [None, [2, 0]])
@pytest.mark.parametrize("batch_delta", [-1, 1])
def test_intrinsic_batch_mismatch_is_atomic(intrinsic_camera, env_ids, batch_delta):
    """Both mismatch directions fail before authoring USD or changing the camera's matrices."""
    stage, camera = intrinsic_camera
    count = 3 if env_ids is None else len(env_ids)
    matrices = torch.eye(3, device=camera.device).repeat(count + batch_delta, 1, 1)
    usd_before = stage.ExportToString()
    data_before = camera._data.intrinsic_matrices.warp.numpy().copy()

    with pytest.raises(ValueError, match="number of intrinsic matrices"):
        camera.set_intrinsic_matrices(matrices, env_ids=env_ids)

    assert stage.ExportToString() == usd_before
    np.testing.assert_array_equal(camera._data.intrinsic_matrices.warp.numpy(), data_before)


@pytest.mark.parametrize(
    "input_kind", ["torch", "torch_strided", "torch_double", "host_torch", "warp", "warp_matrix", "warp_matrix_double"]
)
@pytest.mark.parametrize("focal_length", [None, 24.0])
def test_intrinsic_batch_matches_runtime_projection(intrinsic_camera, input_kind, focal_length, caplog):
    """Float32/64 and strided batches preserve projection semantics and selected-camera order."""
    stage, camera = intrinsic_camera
    usd_before = stage.ExportToString()
    requested = torch.tensor(
        [[[200.125, 0, 140], [0, 300.375, 110], [0, 0, 1]], [[611.25, 0, 350], [0, 589.5, 260], [0, 0, 1]]],
        dtype=torch.float32,
        device=camera.device,
    )
    matrices = requested
    if input_kind == "torch_strided":
        matrices = requested.transpose(1, 2).contiguous().transpose(1, 2)
        assert not matrices.is_contiguous()
    elif input_kind == "torch_double":
        matrices = requested.double()
    elif input_kind == "host_torch":
        matrices = requested.cpu()
    elif input_kind == "warp":
        matrices = wp.from_torch(requested)
    elif input_kind == "warp_matrix":
        matrices = wp.from_torch(requested, dtype=wp.mat33f)
    elif input_kind == "warp_matrix_double":
        matrices = wp.from_torch(requested.double(), dtype=wp.mat33d)
    untouched = camera._data.intrinsic_matrices.warp.numpy()[1].copy()

    camera.set_intrinsic_matrices(matrices, focal_length=focal_length, env_ids=[2, 0])

    assert sum("non square pixels" in message for message in caplog.messages) == 1
    assert sum("aperture offsets" in message for message in caplog.messages) == 1
    actual = camera._data.intrinsic_matrices.warp.numpy()
    np.testing.assert_array_equal(actual[1], untouched)
    for row, env_id in enumerate([2, 0]):
        mean_focal = float((requested[row, 0, 0] + requested[row, 1, 1]).item()) / 2
        pixel_size = 1 / 640 if focal_length is None else focal_length / mean_focal
        parameters = camera._render_data.parameters.numpy()[:, env_id]
        np.testing.assert_allclose(parameters, [pixel_size * mean_focal, pixel_size * 640, pixel_size * 480, 0, 0])
        effective_focal = 640 * float(parameters[0]) / float(parameters[1])
        expected = [[effective_focal, 0, 320], [0, effective_focal, 240], [0, 0, 1]]
        np.testing.assert_allclose(actual[env_id], expected, rtol=1e-7)
    assert stage.ExportToString() == usd_before


@pytest.mark.parametrize("selection", ["all", "slice", "torch", "warp", "repeated", "negative", "empty"])
def test_intrinsic_camera_selections(intrinsic_camera, selection):
    """Selection forms preserve untouched cameras and repeated IDs retain the last update."""
    stage, camera = intrinsic_camera
    ids = [0, 1, 2] if selection == "all" else [2, 0]
    env_ids = ids
    if selection == "all":
        env_ids = None
    elif selection == "slice":
        env_ids = slice(None, None, -2)
    elif selection == "torch":
        env_ids = torch.tensor(ids, device=camera.device)
    elif selection == "warp":
        env_ids = wp.array(ids, dtype=wp.int32, device=camera.device)
    elif selection == "repeated":
        ids = env_ids = [2, 0, 2]
    elif selection == "negative":
        env_ids = [-1, 0]
    elif selection == "empty":
        ids = env_ids = []
    matrices = torch.eye(3, device=camera.device).repeat(len(ids), 1, 1)
    for row in range(len(ids)):
        matrices[row, 0, 0] = matrices[row, 1, 1] = 200 + row * 100
    matrices[:, 0, 2] = 320
    matrices[:, 1, 2] = 240
    before = camera._data.intrinsic_matrices.warp.numpy().copy()
    usd_before = stage.ExportToString()

    camera.set_intrinsic_matrices(matrices, env_ids=env_ids)

    expected = before.copy()
    for row, env_id in enumerate(ids):
        expected[env_id] = [[200 + row * 100, 0, 320], [0, 200 + row * 100, 240], [0, 0, 1]]
    np.testing.assert_allclose(camera._data.intrinsic_matrices.warp.numpy(), expected)
    assert stage.ExportToString() == usd_before


def test_intrinsic_setter_has_no_usd_or_batch_readback(intrinsic_camera, monkeypatch):
    """Gate runtime ownership: only scalar validation status may cross to the host, and USD is unavailable."""
    _stage, camera = intrinsic_camera
    matrices = wp.from_torch(torch.eye(3, device=camera.device).repeat(3, 1, 1))
    original_numpy = wp.array.numpy

    def checked_numpy(array):
        assert array.ptr == camera._intrinsic_status.ptr, "Only scalar validation status may reach the host"
        return original_numpy(array)

    def reject_readback(*args, **kwargs):
        pytest.fail("The runtime setter tried to import USD calibration")

    monkeypatch.setattr(wp.array, "numpy", checked_numpy)
    monkeypatch.setattr(camera, "_initialize_intrinsics", reject_readback)
    # Runtime calibration must not even resolve a USD prim or layer.
    camera.stage = None
    camera._sensor_prims = None
    indices = wp.array([2, 0, 1], dtype=wp.int32, device=camera.device)
    camera.set_intrinsic_matrices(matrices, env_ids=indices)
    wp.synchronize_device(camera.device)


@pytest.mark.parametrize("bad_shape", [(3, 2, 2), (3, 9), (3, 3, 3, 1)])
def test_invalid_intrinsic_shape_does_not_modify_usd(intrinsic_camera, bad_shape):
    stage, camera = intrinsic_camera
    before = stage.ExportToString()
    with pytest.raises(ValueError, match="shape"):
        camera.set_intrinsic_matrices(torch.zeros(bad_shape, device=camera.device))
    assert stage.ExportToString() == before


def test_invalid_intrinsic_selection_does_not_partially_modify_usd(intrinsic_camera):
    stage, camera = intrinsic_camera
    before = stage.ExportToString()
    with pytest.raises(IndexError, match="out of range"):
        camera.set_intrinsic_matrices(torch.eye(3, device=camera.device).repeat(2, 1, 1), env_ids=[0, 3])
    assert stage.ExportToString() == before


@pytest.mark.parametrize("as_warp", [False, True])
def test_single_intrinsic_matrix(intrinsic_camera, as_warp):
    _stage, camera = intrinsic_camera
    matrix = torch.tensor([[240.0, 0, 320], [0, 240.0, 240], [0, 0, 1]], device=camera.device)
    camera.set_intrinsic_matrices(wp.from_torch(matrix) if as_warp else matrix, env_ids=[1])
    np.testing.assert_allclose(camera._data.intrinsic_matrices.warp.numpy()[1], matrix.cpu().numpy())


@pytest.mark.parametrize("edit_layer", ["root", "sublayer", "session"])
def test_runtime_intrinsics_are_independent_of_usd_edit_target(intrinsic_camera, edit_layer):
    """Runtime calibration owns its values independently of USD composition and never authors layers."""
    stage, camera = intrinsic_camera
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        camera._sensor_prims[1].GetFocalLengthAttr().Set(25.0)
        camera._sensor_prims[1].GetHorizontalApertureAttr().Set(32.0)
    target = stage.GetRootLayer()
    if edit_layer == "sublayer":
        target = Sdf.Layer.CreateAnonymous()
        stage.GetRootLayer().subLayerPaths.append(target.identifier)
    elif edit_layer == "session":
        target = stage.GetSessionLayer()
    matrix = torch.tensor([[[100.0, 0, 320], [0, 100.0, 240], [0, 0, 1]]], device=camera.device)

    before = stage.ExportToString()
    with Usd.EditContext(stage, target):
        camera.set_intrinsic_matrices(matrix, env_ids=[1])
    assert stage.ExportToString() == before

    expected_focal = 100
    expected = [[expected_focal, 0, 320], [0, expected_focal, 240], [0, 0, 1]]
    np.testing.assert_allclose(camera._data.intrinsic_matrices.warp.numpy()[1], expected)
