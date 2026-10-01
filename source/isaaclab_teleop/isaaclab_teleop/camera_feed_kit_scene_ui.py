# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit Scene UI presentation for one XR camera feed."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from contextlib import ExitStack, suppress
from typing import Any

import torch

import omni.gpu_foundation_factory as gf
import omni.ui as ui
from omni.kit.scene_view.xr import XRSceneView
from omni.kit.scene_view.xr_utils import SpatialSource, UiContainer, UpdatePolicy, WidgetComponent
from omni.kit.xr.core import XRCore, XRCoreEventType, XRPoseValidityFlags
from pxr import Gf, Sdf, Usd

from isaaclab.app.settings_manager import get_settings_manager
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.array import convert_to_torch
from isaaclab.utils.renderers import ISAAC_RTX_SHOW_ALL_PARTITIONS_BY_DEFAULT_SETTING

logger = logging.getLogger(__name__)

_XR_CAMERA_PATH = "/_xr/stage/xrCamera"
_SCENE_UI_ROOT_PATH = "/ui"
_XR_CAMERA_PIP_PARTITION = "isaaclab_teleop_xr_camera_pip"


def _subscribe_to_kit_frame_updates(callback: Callable[[Any], None], observer_name: str) -> _KitFrameSubscription:
    """Subscribe to Kit post-update events through one resettable wrapper."""
    import carb.eventdispatcher
    import omni.kit.app

    observer = carb.eventdispatcher.get_eventdispatcher().observe_event(
        order=omni.kit.app.POST_UPDATE_ORDER_PYTHON_EXEC,
        event_name=omni.kit.app.GLOBAL_EVENT_POST_UPDATE,
        on_event=callback,
        observer_name=observer_name,
    )
    return _KitFrameSubscription(observer)


def _replicator_output_to_torch(output: Any) -> torch.Tensor:
    """Wrap Replicator's GPU output without copying when it exposes DLPack."""
    if hasattr(output, "__dlpack__"):
        return torch.utils.dlpack.from_dlpack(output)
    return convert_to_torch(output)


def _set_render_product_schema_attribute(
    render_product: Any,
    api_schema: str,
    attribute_name: str,
    value: bool | str,
) -> None:
    """Apply one RTX API schema and author its validated RenderProduct attribute."""
    if not render_product.ApplyAPI(api_schema):
        raise RuntimeError(f"Failed to apply RTX API schema {api_schema!r} to {render_product.GetPath()!s}.")
    attribute = render_product.GetAttribute(attribute_name)
    if not attribute.IsValid():
        raise RuntimeError(
            f"RTX API schema {api_schema!r} does not provide attribute {attribute_name!r} "
            f"on {render_product.GetPath()!s}."
        )
    if not attribute.Set(value):
        raise RuntimeError(f"Failed to set RTX attribute {attribute_name!r} on {render_product.GetPath()!s}.")


def _apply_feed_render_product_settings(render_product_path: Any, cfg: Any | None) -> None:
    """Author optional PiP settings while binding the selected render product."""
    if cfg is None:
        return
    ray_reconstruction = getattr(cfg, "enable_dlss_ray_reconstruction", None)
    dlss_exec_mode = getattr(cfg, "dlss_exec_mode", None)
    if ray_reconstruction is None and dlss_exec_mode is None:
        return
    stage = get_current_stage()
    if stage is None:
        raise RuntimeError("The USD stage is unavailable while configuring an XR camera feed.")
    render_product = stage.GetPrimAtPath(render_product_path)
    if not render_product.IsValid():
        raise RuntimeError(f"Render product {render_product_path!s} was not materialized on the USD stage.")
    if render_product.GetTypeName() != "RenderProduct":
        raise RuntimeError(f"Prim {render_product_path!s} is not a RenderProduct.")
    # Keep transient feed-local opinions stronger than legacy RTX synchronization
    # without persisting presentation policy to the environment's USD layers.
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        if ray_reconstruction is not None:
            _set_render_product_schema_attribute(
                render_product,
                "OmniRtxDebugSettingsAPI_1",
                "omni:rtx:newDenoiser:enabled",
                ray_reconstruction,
            )
        if dlss_exec_mode is not None:
            _set_render_product_schema_attribute(
                render_product,
                "OmniRtxSettingsRtAPI_1",
                "omni:rtx:post:dlss:execMode",
                dlss_exec_mode,
            )


def _try_apply_feed_render_product_settings(camera_name: str, render_product_path: Any, cfg: Any | None) -> None:
    """Apply optional feed settings without disabling PiP when tuning is unavailable."""
    try:
        _apply_feed_render_product_settings(render_product_path, cfg)
    except Exception as exc:
        ray_reconstruction = getattr(cfg, "enable_dlss_ray_reconstruction", None)
        dlss_exec_mode = getattr(cfg, "dlss_exec_mode", None)
        logger.warning(
            "XR camera feed %r could not apply render-product settings to %r "
            "(ray_reconstruction=%r, dlss_exec_mode=%r; %s: %s).",
            camera_name,
            render_product_path,
            ray_reconstruction,
            dlss_exec_mode,
            type(exc).__name__,
            exc,
        )


def _meters_per_unit(coordinate_system: Any, name: str) -> float:
    """Return a validated coordinate-system scale."""
    value = float(coordinate_system.meters_per_unit)
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError(f"{name} meters_per_unit must be finite and positive, got {value}.")
    return value


class KitSceneUiViewerStartAnchor:
    """Share one frozen viewer-start pose across Kit Scene UI panels."""

    def __init__(self, xr_core: Any | None = None):
        self._xr_core = xr_core or XRCore.get_singleton()
        self._registrations: dict[int, tuple[Any, tuple[float, float], float, Callable[[bool], None]]] = {}
        self._upright_pose = None
        self._post_sync_subscription = None
        self._display_disabled_subscription = None

    @property
    def captured(self) -> bool:
        """Whether a valid viewer-start pose has been frozen."""
        return self._upright_pose is not None

    @property
    def xr_core(self) -> Any:
        """XR core used to resolve this anchor."""
        return self._xr_core

    def register(
        self,
        source: Any,
        offset_m: tuple[float, float],
        distance_m: float,
        readiness_callback: Callable[[bool], None],
    ) -> int:
        """Register a panel source and return its lifecycle token."""
        token = id(source)
        self._registrations[token] = (source, offset_m, distance_m, readiness_callback)
        try:
            self._ensure_subscriptions()
            if self._upright_pose is not None:
                self._apply_registration(self._registrations[token])
                readiness_callback(True)
            else:
                readiness_callback(False)
        except Exception:
            self.unregister(token)
            raise
        return token

    def unregister(self, token: int) -> None:
        """Release one panel and subscriptions after the final panel closes."""
        self._registrations.pop(token, None)
        if self._registrations:
            return
        self._post_sync_subscription = None
        self._display_disabled_subscription = None
        self._upright_pose = None

    def _ensure_subscriptions(self) -> None:
        if self._post_sync_subscription is not None:
            return
        message_bus = self._xr_core.get_message_bus()
        self._post_sync_subscription = message_bus.create_subscription_to_pop_by_type(
            XRCoreEventType.post_sync_update,
            self._on_post_sync_update,
            name="Isaac Lab XR camera PiP viewer-start capture",
        )
        self._display_disabled_subscription = message_bus.create_subscription_to_pop_by_type(
            XRCoreEventType.xr_display_disabled,
            self._on_display_disabled,
            name="Isaac Lab XR camera PiP viewer-start reset",
        )

    def _on_post_sync_update(self, _event: Any) -> None:
        self._try_capture()

    def _on_display_disabled(self, _event: Any) -> None:
        """Hide panels and capture a new starting pose after reconnect."""
        self._upright_pose = None
        for _, _, _, callback in tuple(self._registrations.values()):
            callback(False)

    def _try_capture(self) -> bool:
        if self._upright_pose is not None or not self._registrations:
            return self._upright_pose is not None
        pose_desc = self._get_valid_pose_desc()
        if pose_desc is None:
            return False

        coordinate_system = self._xr_core.get_coordinate_system()
        self._upright_pose = self._xr_core.reorient_transform_matrix_up_right(
            pose_desc.pose_matrix,
            coordinate_system.up_axis == "y",
        )
        for registration in tuple(self._registrations.values()):
            self._apply_registration(registration)
            registration[3](True)
        logger.info(
            "XR camera PiP viewer-start anchor captured at %s (validity_flags=%s).",
            tuple(float(value) for value in self._upright_pose.ExtractTranslation()),
            pose_desc.validity_flags,
        )
        return True

    def _get_valid_pose_desc(self) -> Any | None:
        if not self._xr_core.is_xr_display_enabled():
            return None
        input_device = self._xr_core.get_input_device("displayDevice")
        if input_device is None:
            input_device = self._xr_core.get_input_device("/user/head")
        if input_device is None:
            return None
        pose_desc = input_device.get_virtual_world_pose_desc("")
        if pose_desc is None:
            return None
        required_flags = XRPoseValidityFlags.POSITION_VALID | XRPoseValidityFlags.ORIENTATION_VALID
        if pose_desc.validity_flags & required_flags != required_flags:
            return None
        return pose_desc

    def _apply_registration(
        self,
        registration: tuple[Any, tuple[float, float], float, Callable[[bool], None]],
    ) -> None:
        source, offset_m, distance_m, _ = registration
        coordinate_system = self._xr_core.get_coordinate_system()
        meters_per_unit = _meters_per_unit(coordinate_system, "XR coordinate-system")
        # XR poses use local +X right, +Y up, and -Z forward. Apply the
        # panel offset in that local frame before the frozen pose maps it into
        # the stage's Y-up or Z-up world.
        offset = Gf.Vec3d(offset_m[0], offset_m[1], -distance_m) / meters_per_unit
        panel_world = Gf.Matrix4d().SetTranslate(offset) * self._upright_pose
        source.source = SpatialSource.new_transform_matrix_source(panel_world).source


class KitSceneUiHeadLockedAnchor(KitSceneUiViewerStartAnchor):
    """Drive panels from the current viewer pose on every XR frame."""

    def _try_capture(self) -> bool:
        if not self._registrations:
            return False
        pose_desc = self._get_valid_pose_desc()
        if pose_desc is None:
            if self._upright_pose is not None:
                self._on_display_disabled(None)
            return False

        was_ready = self._upright_pose is not None
        # Preserve the complete headset pose. Unlike viewer-start placement,
        # head-locked panels must follow pitch and roll as well as position/yaw.
        self._upright_pose = pose_desc.pose_matrix
        for registration in tuple(self._registrations.values()):
            self._apply_registration(registration)
            if not was_ready:
                registration[3](True)
        if not was_ready:
            logger.info(
                "XR camera PiP head-locked anchor is following the display pose (validity_flags=%s).",
                pose_desc.validity_flags,
            )
        return True


def _world_panel_matrix(
    descriptor: Any,
    meters_per_unit: float,
) -> Gf.Matrix4d:
    """Build one fixed panel transform from the explicit world layout pose."""
    if not math.isfinite(meters_per_unit) or meters_per_unit <= 0.0:
        raise RuntimeError(f"Stage meters_per_unit must be finite and positive, got {meters_per_unit}.")
    if descriptor.world_position_m is None:
        raise ValueError("world_position_m is required for world XR camera-feed placement.")
    x, y, z, w = descriptor.world_orientation_xyzw
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm <= 1.0e-8:
        raise ValueError("world_orientation_xyzw must be a finite, non-zero quaternion.")
    rotation = Gf.Quatd(w / norm, Gf.Vec3d(x / norm, y / norm, z / norm))
    anchor = Gf.Matrix4d(1.0).SetRotate(rotation)
    anchor.SetTranslateOnly(Gf.Vec3d(*(value / meters_per_unit for value in descriptor.world_position_m)))
    local_offset = Gf.Matrix4d().SetTranslate(
        Gf.Vec3d(
            descriptor.offset_m[0] / meters_per_unit,
            descriptor.offset_m[1] / meters_per_unit,
            0.0,
        )
    )
    return local_offset * anchor


class _CameraImageWidget(ui.Widget):
    """Omni UI image surface backed by a Kit byte-image provider."""

    def __init__(self, provider: ui.ByteImageProvider, label: str | None = None, **kwargs):
        super().__init__(**kwargs)
        with ui.ZStack():
            ui.Rectangle(
                style={
                    "Rectangle": {
                        "background_color": 0xFF101214,
                        "border_color": 0xFF73787E,
                        "border_width": 1,
                        "border_radius": 2,
                    }
                }
            )
            with ui.VStack(spacing=0):
                if label:
                    ui.Label(
                        label,
                        height=22,
                        alignment=ui.Alignment.CENTER,
                        style={"Label": {"color": 0xFFE8EAED, "font_size": 14}},
                    )
                ui.ImageWithProvider(
                    provider,
                    fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT,
                )


class KitSceneUiCameraFeedPanel:
    """Persistent Kit Scene UI panel for one RGBA camera."""

    def __init__(
        self,
        descriptor: Any,
        image_width: int,
        image_height: int,
        viewer_start_anchor: KitSceneUiViewerStartAnchor | None = None,
        head_locked_anchor: KitSceneUiHeadLockedAnchor | None = None,
        presenter: _KitSceneUiCameraFeedPresenter | None = None,
    ):
        """Create the image provider and attach its panel to the XR scene."""
        self._closed = False
        self._provider = None
        self._component = None
        self._container = None
        self._pose_anchor = {"viewer_start": viewer_start_anchor, "head_locked": head_locked_anchor}.get(
            descriptor.placement
        )
        self._pose_registration = None
        self._presenter = presenter
        self._pose_ready = descriptor.placement == "world"
        self._partition_ready = presenter is None

        if descriptor.placement not in {"viewer_start", "head_locked", "world"}:
            raise ValueError(f"Unknown XR camera-feed placement {descriptor.placement!r}.")
        if descriptor.placement != "world" and self._pose_anchor is None:
            raise ValueError(f"{descriptor.placement} placement requires a shared pose anchor.")
        xr_core = self._pose_anchor.xr_core if self._pose_anchor is not None else XRCore.get_singleton()
        coordinate_system = (
            xr_core.get_stage_coordinate_system()
            if descriptor.placement == "world"
            else xr_core.get_coordinate_system()
        )
        coordinate_system_name = "Stage" if descriptor.placement == "world" else "XR coordinate-system"
        meters_per_unit = _meters_per_unit(coordinate_system, coordinate_system_name)
        image_height_m = descriptor.width_m * image_height / image_width
        label_height_m = 0.04 if descriptor.label else 0.0
        panel_width_units = descriptor.width_m / meters_per_unit
        panel_height_units = (image_height_m + label_height_m) / meters_per_unit
        resolution_scale = max(image_width / descriptor.width_m, image_height / image_height_m)
        if self._pose_anchor is not None:
            anchor_source = SpatialSource.new_transform_matrix_source(Gf.Matrix4d(1.0))
            space_stack = [anchor_source]
        else:
            anchor_source = None
            space_stack = [
                SpatialSource.new_transform_matrix_source(
                    _world_panel_matrix(descriptor, meters_per_unit),
                )
            ]
        try:
            self._provider = ui.ByteImageProvider()
            self._component = WidgetComponent(
                _CameraImageWidget,
                width=panel_width_units,
                height=panel_height_units,
                resolution_scale=resolution_scale,
                unit_to_pixel_scale=meters_per_unit,
                update_policy=UpdatePolicy.ALWAYS,
                widget_args=[self._provider, descriptor.label],
            )
            self._container = UiContainer(
                XRSceneView,
                self._component,
                space_stack=space_stack,
            )
            self._update_visibility()
            if self._pose_anchor is not None:
                self._pose_registration = self._pose_anchor.register(
                    anchor_source,
                    descriptor.offset_m,
                    descriptor.distance_m,
                    self._on_pose_readiness_changed,
                )
            if self._presenter is not None:
                self._presenter._partition_panels.add(self)
                self._presenter._refresh_scene_partition()
        except Exception:
            with suppress(Exception):
                self.close()
            raise

    def _on_pose_readiness_changed(self, ready: bool) -> None:
        self._pose_ready = ready
        self._update_visibility()

    def _update_visibility(self) -> None:
        if self._container is None:
            return
        if self._pose_ready and self._partition_ready:
            self._container.show()
        else:
            self._container.hide()

    def upload(self, image: torch.Tensor) -> None:
        """Upload a contiguous RGBA tensor through Kit's matching provider path."""
        if self._closed:
            return
        size = [int(image.shape[1]), int(image.shape[0])]
        if image.device.type == "cuda":
            self._provider.set_bytes_data_from_gpu(
                int(image.data_ptr()),
                size,
                gf.TextureFormat.RGBA8_UNORM,
            )
        else:
            self._provider.set_bytes_data(
                image.numpy().reshape(-1).data,
                size,
                gf.TextureFormat.RGBA8_UNORM,
            )

    def close(self) -> None:
        """Release the XR scene hierarchy and provider references."""
        if self._closed:
            return
        self._closed = True
        if self._pose_anchor is not None and self._pose_registration is not None:
            self._pose_anchor.unregister(self._pose_registration)
        self._pose_registration = None
        self._pose_anchor = None
        container = self._container
        self._component = None
        self._container = None
        self._provider = None
        try:
            if container is not None:
                try:
                    container.hide()
                finally:
                    container.root.clear()
        finally:
            if self._presenter is not None:
                self._presenter._partition_panels.discard(self)
                if not self._presenter._partition_panels:
                    self._presenter._clear_scene_partition()
                self._presenter = None


class _ReplicatorCameraFeedSource:
    """Feed-owned CUDA view of an existing RTX camera render product."""

    def __init__(self, camera_name: str, annotator: Any, render_product_path: Any):
        self._camera_name = camera_name
        self._annotator = annotator
        self._render_product_path = render_product_path
        self._read_error_reported = False
        self._ready_reported = False

    @classmethod
    def try_create(cls, camera_name: str, camera: Any, cfg: Any | None = None) -> _ReplicatorCameraFeedSource | None:
        """Attach to a camera's existing RTX render product when one is available."""
        # Keep renderer-private discovery contained in this optional presentation adapter.
        # Backends without this RTX render-product shape use the Camera buffer fallback.
        render_data = getattr(camera, "_render_data", None)
        render_product = getattr(render_data, "render_product", None)
        render_product_path = getattr(render_product, "path", None)
        if not render_product_path:
            return None

        annotator = None
        try:
            import omni.replicator.core as rep

            annotator = rep.AnnotatorRegistry.get_annotator(
                "rgb",
                # Do not impose a CUDA ordinal. With the zero-copy pointer path,
                # Replicator exposes the render product's actual output device.
                device="cuda",
                do_array_copy=False,
            )
            annotator.attach([render_product_path])
        except Exception as exc:
            if annotator is not None:
                with suppress(Exception):
                    annotator.detach([render_product_path])
            # Camera-buffer fallback still displays pixels from this render product.
            # A failed attach may have synchronized legacy settings, so restore the
            # feed-local policy just as on the successful CUDA-source path below.
            _try_apply_feed_render_product_settings(camera_name, render_product_path, cfg)
            logger.warning(
                "XR camera feed %r could not attach a CUDA annotator to render product %r "
                "(%s: %s). Falling back to the Camera RGBA buffer.",
                camera_name,
                render_product_path,
                type(exc).__name__,
                exc,
            )
            return None
        _try_apply_feed_render_product_settings(camera_name, render_product_path, cfg)
        return cls(camera_name, annotator, render_product_path)

    def get_image(self, expected_shape: tuple[int, ...]) -> torch.Tensor | None:
        """Return a current zero-copy CUDA frame when one is ready and valid."""
        if self._annotator is None:
            return None
        try:
            output = self._annotator.get_data()
            if isinstance(output, dict):
                output = output.get("data")
            if output is None:
                return None
            image = _replicator_output_to_torch(output)
            if image.numel() == 0:
                return None
        except Exception as exc:
            if not self._read_error_reported:
                logger.warning(
                    "XR camera feed %r could not read its CUDA annotator (%s: %s). "
                    "Falling back to the Camera RGBA buffer.",
                    self._camera_name,
                    type(exc).__name__,
                    exc,
                )
                self._read_error_reported = True
            return None

        shape = tuple(image.shape)
        contiguous = image.is_contiguous()
        if shape != expected_shape or image.dtype != torch.uint8 or image.device.type != "cuda" or not contiguous:
            if not self._read_error_reported:
                logger.warning(
                    "XR camera feed %r received an incompatible CUDA annotator frame "
                    "(shape=%s, expected_shape=%s, dtype=%s, device=%s, contiguous=%s). "
                    "Falling back to the Camera RGBA buffer.",
                    self._camera_name,
                    shape,
                    expected_shape,
                    image.dtype,
                    image.device,
                    contiguous,
                )
                self._read_error_reported = True
            return None
        if not self._ready_reported:
            logger.info(
                "XR camera feed %r is using its feed-owned zero-copy CUDA annotator.",
                self._camera_name,
            )
            self._ready_reported = True
        self._read_error_reported = False
        return image

    def close(self) -> None:
        """Detach the feed annotator without destroying the camera-owned render product."""
        annotator = self._annotator
        self._annotator = None
        if annotator is not None:
            annotator.detach([self._render_product_path])


class _KitSceneUiCameraFeedPresenter:
    """Private adapter from camera buffers to Kit SceneUI panels."""

    # All presenters share Kit's /ui root. Track the panels themselves,
    # and keep temporary USD edits in one cleanup stack for their combined lifetime.
    _partition_panels: set[KitSceneUiCameraFeedPanel] = set()
    _partition_stage = None
    _partition_cleanup = ExitStack()
    _partition_authored: set[Sdf.Path] = set()
    _partition_ui_root = None

    def __init__(self):
        self._viewer_start_anchor = None
        self._head_locked_anchor = None
        self._cpu_upload_warnings: set[str] = set()
        self._scene_partition_update_error_reported = False

    @staticmethod
    def _validate_image(camera_name: str, image: torch.Tensor) -> None:
        if image.device.type not in {"cpu", "cuda"}:
            raise ValueError(f"Camera {camera_name!r} RGBA image must reside on CPU or CUDA, got {image.device}.")
        if not image.is_contiguous():
            raise ValueError(f"Camera {camera_name!r} RGBA image must be contiguous.")

    @staticmethod
    def create_image_source(
        camera_name: str,
        camera: Any,
        cfg: Any | None = None,
    ) -> _ReplicatorCameraFeedSource | None:
        """Create an opportunistic CUDA source from the camera's existing render product."""
        return _ReplicatorCameraFeedSource.try_create(camera_name, camera, cfg)

    def prepare_upload_image(
        self,
        camera_name: str,
        image: torch.Tensor,
        previous_source: torch.Tensor | None = None,
        previous_upload: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return direct storage or stage CPU pixels on the last render-product GPU."""
        self._validate_image(camera_name, image)
        if image.device.type == "cuda":
            return image
        if (
            previous_source is not None
            and previous_upload is not None
            and previous_upload is not previous_source
            and tuple(previous_upload.shape) == tuple(image.shape)
            and previous_upload.dtype == image.dtype
            and previous_upload.device.type == "cuda"
        ):
            return previous_upload
        if previous_source is None or previous_source.device.type != "cuda":
            if camera_name not in self._cpu_upload_warnings:
                logger.warning(
                    "XR camera feed %r is using the CPU ByteImageProvider upload path. "
                    "The per-frame host transfer may reduce PiP performance.",
                    camera_name,
                )
                self._cpu_upload_warnings.add(camera_name)
            return image
        logger.warning(
            "XR camera feed %r is using a CPU buffer and requires a CPU-to-GPU staging copy each frame.",
            camera_name,
        )
        return torch.empty_like(image, device=previous_source.device, memory_format=torch.contiguous_format)

    def create_panel(self, descriptor: Any, width: int, height: int) -> KitSceneUiCameraFeedPanel:
        viewer_start_anchor = None
        head_locked_anchor = None
        if descriptor.placement == "viewer_start":
            if self._viewer_start_anchor is None:
                self._viewer_start_anchor = KitSceneUiViewerStartAnchor()
            viewer_start_anchor = self._viewer_start_anchor
        elif descriptor.placement == "head_locked":
            if self._head_locked_anchor is None:
                self._head_locked_anchor = KitSceneUiHeadLockedAnchor()
            head_locked_anchor = self._head_locked_anchor
        return KitSceneUiCameraFeedPanel(
            descriptor=descriptor,
            image_width=width,
            image_height=height,
            viewer_start_anchor=viewer_start_anchor,
            head_locked_anchor=head_locked_anchor,
            presenter=self if descriptor.use_scene_partition else None,
        )

    @staticmethod
    def validate_camera_partition(camera_name: str, camera: Any) -> None:
        """Require an unpartitioned Isaac RTX source camera before displaying its image."""
        render_data = getattr(camera, "_render_data", None)
        if getattr(render_data, "render_product", None) is None:
            raise ValueError(f"Isolated XR camera feed {camera_name!r} requires an Isaac RTX camera.")
        stage = get_current_stage()
        for path in render_data.spec.camera_prim_paths:
            prim = stage.GetPrimAtPath(path)
            partition = prim.GetAttribute("omni:scenePartition") if prim else None
            if partition and partition.Get() not in (None, ""):
                raise ValueError(
                    f"Isolated XR camera feed {camera_name!r} requires an unpartitioned camera; "
                    "prepare the camera-feed session before constructing the environment."
                )

    @classmethod
    def _author_scene_partition(cls, prim: Any, name: str, value: str) -> None:
        """Override one session-layer token, restoring only opinions still owned by the presenter."""
        attribute = prim.GetAttribute(name)
        current = attribute.Get() if attribute else None
        if current not in (None, "", value):
            raise RuntimeError(f"XR camera PiP cannot replace existing partition {current!r} on {prim.GetPath()}.")
        layer = prim.GetStage().GetSessionLayer()
        path = prim.GetPath().AppendProperty(name)
        if path not in cls._partition_authored:
            spec = layer.GetAttributeAtPath(path)
            had_property = spec is not None
            previous = spec.default if spec is not None else None
            had_prim = layer.GetPrimAtPath(prim.GetPath()) is not None

            def restore():
                spec = layer.GetAttributeAtPath(path)
                if spec is None or spec.default != value:
                    return
                if not had_property:
                    owner = spec.owner
                    owner.RemoveProperty(spec)
                    if not had_prim:
                        layer.ScheduleRemoveIfInert(owner)
                elif previous is None:
                    spec.ClearDefaultValue()
                else:
                    spec.default = previous

            cls._partition_cleanup.callback(restore)
            cls._partition_authored.add(path)
        if current != value:
            with Usd.EditContext(prim.GetStage(), layer):
                if not attribute:
                    attribute = prim.CreateAttribute(name, Sdf.ValueTypeNames.Token)
                if not attribute.Set(value):
                    raise RuntimeError(f"Failed to author XR camera PiP partition on {prim.GetPath()}.")

    @classmethod
    def _refresh_scene_partition(cls) -> None:
        """Keep the environment shared, with overlapping SceneUI visible only to the XR partition."""
        if not cls._partition_panels:
            return
        stage = get_current_stage()
        if stage is not cls._partition_stage:
            cls._clear_scene_partition()
            cls._partition_stage = stage
        if stage is None:
            return
        try:
            if get_settings_manager().get(ISAAC_RTX_SHOW_ALL_PARTITIONS_BY_DEFAULT_SETTING) is not False:
                raise RuntimeError("Isolated XR PiP requires showAllPartitionsByDefault=False.")
            env_root = stage.GetPrimAtPath("/World/envs/env_0")
            env_partition = env_root.GetAttribute("primvars:omni:scenePartition") if env_root else None
            if env_partition and env_partition.Get() not in (None, ""):
                raise RuntimeError(
                    "Isolated XR PiP requires an unpartitioned environment before camera creation; "
                    "late partition changes can hide instanced geometry."
                )
            camera = stage.GetPrimAtPath(_XR_CAMERA_PATH)
            if camera:
                cls._author_scene_partition(camera, "omni:scenePartition", _XR_CAMERA_PIP_PARTITION)

            layer = stage.GetSessionLayer()
            ui_root = stage.GetPrimAtPath(_SCENE_UI_ROOT_PATH)
            if not ui_root:
                # SceneUI creates /ui on its first draw. Seed its partition before showing a panel.
                with Usd.EditContext(stage, layer):
                    ui_root = stage.OverridePrim(_SCENE_UI_ROOT_PATH)
                cls._partition_cleanup.callback(layer.ScheduleRemoveIfInert, layer.GetPrimAtPath(_SCENE_UI_ROOT_PATH))
            name = "primvars:omni:scenePartition"
            cls._author_scene_partition(ui_root, name, _XR_CAMERA_PIP_PARTITION)
            path = ui_root.GetPath().AppendProperty(name)
            populated_root = ui_root if ui_root.GetChildren() else None
            refresh_inheritance = populated_root is not None and populated_root != cls._partition_ui_root
            if refresh_inheritance:
                # Fabric skips inheritance on childless roots. Notify it once descendants appear.
                spec = layer.GetAttributeAtPath(path)
                if spec is None:
                    with Usd.EditContext(stage, layer):
                        ui_root.GetAttribute(name).Set(_XR_CAMERA_PIP_PARTITION)
                else:
                    with Sdf.ChangeBlock():
                        spec.ClearDefaultValue()
                        spec.default = _XR_CAMERA_PIP_PARTITION
            cls._partition_ui_root = populated_root
            ready = bool(camera)
            for panel in tuple(cls._partition_panels):
                if panel._partition_ready != ready:
                    panel._partition_ready = ready
                    panel._update_visibility()
        except Exception:
            cls._clear_scene_partition()
            raise

    @classmethod
    def _clear_scene_partition(cls) -> None:
        for panel in tuple(cls._partition_panels):
            panel._partition_ready = False
            panel._update_visibility()
        try:
            cls._partition_cleanup.close()
        finally:
            cls._partition_authored.clear()
            cls._partition_stage = None
            cls._partition_ui_root = None

    @staticmethod
    def stage_upload_image(image: torch.Tensor, upload_image: torch.Tensor) -> None:
        if upload_image is not image:
            upload_image.copy_(image, non_blocking=False)

    def subscribe_to_frame_updates(self, callback: Callable[[Any], None]) -> _KitFrameSubscription:
        def on_frame(event: Any) -> None:
            try:
                self._refresh_scene_partition()
                self._scene_partition_update_error_reported = False
            except Exception as exc:
                if not self._scene_partition_update_error_reported:
                    logger.warning(
                        "XR camera PiP could not update its SceneUI scene partition (%s: %s).",
                        type(exc).__name__,
                        exc,
                    )
                    self._scene_partition_update_error_reported = True
            finally:
                callback(event)

        return _subscribe_to_kit_frame_updates(
            on_frame,
            "Isaac Lab XR camera PiP frame update",
        )


class _KitFrameSubscription:
    def __init__(self, observer: Any):
        self._observer = observer

    def close(self) -> None:
        if self._observer is not None:
            self._observer.reset()
            self._observer = None
