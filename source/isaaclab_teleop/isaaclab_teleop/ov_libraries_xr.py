# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab host adapter for IsaacTeleop-owned OVXR presentation."""

from __future__ import annotations

import contextlib
import logging
import math
import re
import threading
from collections.abc import Hashable, Iterable
from dataclasses import dataclass, replace
from typing import Any, Literal

from isaaccapture.ovxr_session_manager import OvxrApplicationInterface, StageOrdinalLedger
from isaaclab_ov.ovxr_session import OvxrSessionBootstrap, validate_ovxr_session_preparation
from isaaclab_ov.renderers import OVRTXRendererCfg

logger = logging.getLogger(__name__)

_DEFAULT_OVXR_CAMERA_PATHS = (
    "/OVXRGeneratedCameras/OVXR_LeftEyeCamera",
    "/OVXRGeneratedCameras/OVXR_RightEyeCamera",
)
_DEFAULT_OVXR_RENDER_PRODUCT_PATHS = (
    "/Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_0_OVXR_Left",
    "/Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_0_OVXR_Right",
)
_DEFAULT_OVXR_RENDER_VAR_PATHS = (
    "/Render/Vars/LdrColor",
    "/Render/Vars/XRDepth",
)
_DEFAULT_OVXR_RENDER_SETTINGS_PATH = "/Render/OmniverseGlobalRenderSettings"
_ENVIRONMENT_ROOT_PATH = "/World/envs/env_0"


def _apply_hidden_visual_prims(env: Any, patterns: Iterable[str]) -> tuple[str, ...]:
    """Author persistent application-stage visibility before materialization.

    This affects every view of the application scene, not just the XR eyes.
    Use scene partitions when source visibility must remain unchanged.
    """
    patterns = tuple(patterns)
    if not patterns:
        return ()

    from pxr import UsdGeom

    unwrapped = getattr(env, "unwrapped", env)
    scene = getattr(unwrapped, "scene", None)
    stage = getattr(scene, "stage", None)
    if stage is None:
        raise RuntimeError("Configured hidden visual prims require an environment Stage")

    compiled = [(pattern, re.compile(pattern)) for pattern in patterns]
    matches: dict[str, list[Any]] = {pattern: [] for pattern in patterns}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Imageable):
            continue
        path = str(prim.GetPath())
        for pattern, expression in compiled:
            if expression.fullmatch(path):
                matches[pattern].append(prim)

    missing = [pattern for pattern, prims in matches.items() if not prims]
    if missing:
        raise RuntimeError(f"Hidden visual prim patterns matched no imageable prims: {missing}")

    hidden_paths: list[str] = []
    hidden_path_set: set[str] = set()
    for pattern in patterns:
        for prim in matches[pattern]:
            path = str(prim.GetPath())
            if path in hidden_path_set:
                continue
            UsdGeom.Imageable(prim).MakeInvisible()
            hidden_paths.append(path)
            hidden_path_set.add(path)
    logger.info("Hid %d prim(s) from all camera views (%s)", len(hidden_paths), list(patterns))
    return tuple(hidden_paths)


def _declare_ovxr_runtime_prims(env: Any, cfg: OvLibrariesXrConfig) -> tuple[tuple[str, str], ...]:
    """Declare type-only OVXR prims for the application Stage's lifetime."""
    if not cfg.create_render_products:
        return ()

    unwrapped = getattr(env, "unwrapped", env)
    scene = getattr(unwrapped, "scene", None)
    stage = getattr(scene, "stage", None)
    if stage is None:
        raise RuntimeError("OVXR runtime prim declaration requires an environment Stage")

    render_products = cfg.render_products or _DEFAULT_OVXR_RENDER_PRODUCT_PATHS
    declarations = (
        *((path, "Camera") for path in cfg.camera_paths),
        *((path, "RenderProduct") for path in render_products),
        *((path, "RenderVar") for path in _DEFAULT_OVXR_RENDER_VAR_PATHS),
        (_DEFAULT_OVXR_RENDER_SETTINGS_PATH, "RenderSettings"),
    )
    expected_types: dict[str, str] = {}
    for path, type_name in declarations:
        if not path.startswith("/") or path == "/":
            raise ValueError(f"OVXR runtime prim path must be absolute and non-root: {path!r}")
        previous_type = expected_types.setdefault(path, type_name)
        if previous_type != type_name:
            raise ValueError(f"OVXR runtime prim path {path!r} has conflicting requested types")

    for path, type_name in expected_types.items():
        existing = stage.GetPrimAtPath(path)
        existing_type = str(existing.GetTypeName()) if existing and existing.IsValid() else ""
        if existing_type and existing_type != type_name:
            raise RuntimeError(
                f"OVXR runtime prim {path!r} already has incompatible type {existing_type!r}; expected {type_name!r}"
            )
        stage.DefinePrim(path, type_name)
    return tuple(expected_types.items())


def _import_ovxr():
    import ovxr

    return ovxr


class _StageOrdinalLedger(StageOrdinalLedger):
    """Reserve explicit OVStage ordinals for the application publication lane."""

    def __init__(self, stage: Any) -> None:
        self._stage = stage
        self._state_lock = threading.Lock()
        self._active_ordinal: int | None = None
        self._closed = False
        self._floor = self._read_floor()

    def _read_floor(self) -> int:
        with self._stage.get_attribute_write_floor() as query:
            query.wait()
            floor = int(query.fetch())
        if floor < 0 or floor > (1 << 64) - 1:
            raise RuntimeError(f"OVStage returned an invalid write floor: {floor}")
        return floor

    def _require_active(self, ordinal: int) -> None:
        if self._active_ordinal != ordinal:
            raise RuntimeError(
                f"OVStage ordinal {ordinal} is not the active application reservation ({self._active_ordinal=})"
            )

    def reserve(self) -> int:
        """Reserve exactly ``floor + 1`` while the application publication guard is held."""
        with self._state_lock:
            if self._closed:
                raise RuntimeError("OVStage ordinal ledger is closed")
            if self._active_ordinal is not None:
                raise RuntimeError(f"OVStage ordinal {self._active_ordinal} is still unresolved")
            if self._floor == (1 << 64) - 1:
                raise RuntimeError("OVStage ordinal range is exhausted")
            self._active_ordinal = self._floor + 1
            return self._active_ordinal

    def publish(self, ordinal: int) -> None:
        """Seal one application-written reservation and mark it resolved."""
        with self._state_lock:
            self._require_active(ordinal)
            self._stage.advance_write_floor(ordinal=ordinal).wait()
            self._floor = ordinal
            self._active_ordinal = None

    def resolve_external(self, ordinal: int, *, expected_published: bool | None) -> bool:
        """Resolve an external write and verify its documented publication result."""
        with self._state_lock:
            self._require_active(ordinal)
            if expected_published is True:
                published = True
            else:
                floor = self._read_floor()
                if floor < self._floor or floor > ordinal:
                    raise RuntimeError(
                        f"OVStage write floor {floor} is outside the active ledger range [{self._floor}, {ordinal}]"
                    )
                published = floor == ordinal
            # Record native side effects even when the response contradicts the
            # expected result. A reported error cannot make a sealed ordinal reusable.
            if published:
                self._floor = ordinal
            if expected_published is not None and published != expected_published:
                self._active_ordinal = None
                state = "sealed" if published else "unsealed"
                expected = "sealed" if expected_published else "unsealed"
                raise RuntimeError(f"OVStage ordinal {ordinal} is {state} after an external write; expected {expected}")
            self._active_ordinal = None
            return published

    def abandon(self, ordinal: int) -> None:
        """Reconcile a failed writer before releasing its reservation."""
        with self._state_lock:
            self._require_active(ordinal)
            # A publication wait can fail after the native Stage has sealed it.
            # Query only on the failure/abandon path; an unreadable or unrelated
            # floor leaves the reservation unresolved so it cannot be reused.
            floor = self._read_floor()
            if floor < self._floor or floor > ordinal:
                raise RuntimeError(
                    f"OVStage write floor {floor} is outside the active ledger range [{self._floor}, {ordinal}]"
                )
            self._floor = floor
            self._active_ordinal = None

    def destroy(self) -> None:
        """Close the ledger after every application and external write is resolved."""
        with self._state_lock:
            if self._active_ordinal is not None:
                raise RuntimeError(f"Cannot close with unresolved OVStage ordinal {self._active_ordinal}")
            self._closed = True


@dataclass
class OvLibrariesXrConfig:
    """Isaac Lab configuration for an IsaacTeleop-owned OVXR session."""

    application_name: str = "isaaclab-kitless-teleop"
    session_mode: Literal["ar", "vr"] = "ar"
    renderer_settings: str = "--/app/hydra/aperture/conform=stretch"
    cloudxr_env_file: str | None = None
    auto_launch_cloudxr: bool = True
    render_products: tuple[str, ...] = ()
    camera_paths: tuple[str, ...] = _DEFAULT_OVXR_CAMERA_PATHS
    create_render_products: bool = True
    meters_per_unit: float = 1.0
    depth_near_z: float = 0.0
    depth_far_z: float = 0.0
    tracking_anchor_path: str = ""
    host_wait_render_output: bool = True
    extra_openxr_extensions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.session_mode not in ("ar", "vr"):
            raise ValueError("session_mode must be 'ar' or 'vr'")
        if self.meters_per_unit <= 0:
            raise ValueError("meters_per_unit must be positive")
        if (self.depth_near_z, self.depth_far_z) != (0.0, 0.0) and (
            not math.isfinite(self.depth_near_z)
            or not math.isfinite(self.depth_far_z)
            or self.depth_near_z <= 0.0
            or self.depth_far_z <= self.depth_near_z
        ):
            raise ValueError("depth_near_z and depth_far_z must both be zero or finite with 0 < near < far")
        if len(self.camera_paths) != 2 or any(not path for path in self.camera_paths):
            raise ValueError("Kitless stereo requires exactly two non-empty OVXR camera paths")


class _IsaacLabOvxrApplicationInterface(OvxrApplicationInterface):
    """Supply only the renderer/Stage/coordinator and simulation publication."""

    def __init__(
        self,
        bootstrap: OvxrSessionBootstrap,
        env: Any,
        cfg: OvLibrariesXrConfig,
        hidden_visual_prims: Iterable[str],
        external_anchor_state=None,
    ) -> None:
        self._bootstrap = bootstrap
        self._env = env
        self._cfg = cfg
        self._hidden_visual_prims = tuple(hidden_visual_prims)
        self._publication_lock = threading.Lock()
        self._scene_publisher = None
        self._lease = None
        self._ordinal_ledger = None
        self._last_published_ordinal: int | None = None
        self._publication_closed = False
        self._external_anchor = external_anchor_state
        self._anchor_snapshot = None
        self._anchor_resources = contextlib.ExitStack()
        self._anchor_query = None
        self._anchor_stage = None
        self._anchor_retirement_failed = False

    def offer_render_owner_transfer(self) -> object:
        self._scene_publisher = self._bootstrap.capture_scene_publisher()
        return self._bootstrap.offer_render_owner_transfer()

    def attach_on_render_thread(self, capability: object):
        from isaaccapture.ovxr_session_manager.ovxr_session import OvxrSessionBinding

        self._bootstrap.claim_render_owner_transfer(capability)
        _apply_hidden_visual_prims(self._env, self._hidden_visual_prims)
        _declare_ovxr_runtime_prims(self._env, self._cfg)
        self._lease = self._bootstrap.bind_environment(self._env)
        with self._lease.ovstage_owner_lane():
            self._ordinal_ledger = _StageOrdinalLedger(self._lease.ovstage)
            if self._external_anchor is not None:
                import ovstage

                self._anchor_stage = self._lease.ovstage
                try:
                    paths = self._anchor_resources.enter_context(ovstage.PathDictionary(self._anchor_stage))
                    path_list = paths.create_path_list_from_strings(self._external_anchor.external_anchor_paths)
                    self._anchor_resources.callback(paths.destroy_path_list, path_list)
                    self._anchor_query = self._anchor_resources.enter_context(
                        self._anchor_stage.query_from_path_list(path_list)
                    )
                    self._anchor_query.wait()
                except BaseException as setup_error:
                    try:
                        self._anchor_resources.close()
                    except BaseException as cleanup_error:
                        self._anchor_retirement_failed = True
                        setup_error.add_note(f"Anchor query cleanup also failed: {cleanup_error!r}")
                    raise
        return OvxrSessionBinding(
            renderer=self._lease.ovrtx_renderer,
            stage=self._lease.ovstage,
            ordinal_ledger=self._ordinal_ledger,
            prepare_frame_scene=self.prepare_frame_scene if self._external_anchor is not None else None,
        )

    def publication_guard(self):
        return self._publication_lock

    def owner_lane(self):
        if self._lease is None:
            raise RuntimeError("OVRTX/OVStage lease is not attached")
        return self._lease.ovstage_owner_lane()

    def publish_complete_scene(self, publication_id: Hashable) -> int:
        if self._publication_closed or self._scene_publisher is None or self._ordinal_ledger is None:
            raise RuntimeError("IsaacLab scene publication is not available")
        anchor_epoch = None
        if self._external_anchor is not None:
            # A failed seed can follow the publisher recording its identity.
            # Check the terminal policy before dedup may return an older seal.
            anchor_epoch = self._external_anchor.external_epoch
        if not self._scene_publisher.publication_required(publication_id):
            if self._last_published_ordinal is None:
                raise RuntimeError("OVStage publication deduplicated before a sealed ordinal exists")
            return self._last_published_ordinal
        snapshot = None
        seed = False
        if self._external_anchor is not None:
            epoch, seed = anchor_epoch
            reference = self._bootstrap.get_world_transform(self._external_anchor.external_reference_path)
            unwrapped = getattr(self._env, "unwrapped", self._env)
            snapshot = (epoch, reference, unwrapped.sim.get_rendering_dt())
        ordinal = self._ordinal_ledger.reserve()
        try:
            self._scene_publisher.write_complete_scene(ordinal, publication_id)
        except BaseException as write_error:
            try:
                self._ordinal_ledger.abandon(ordinal)
            except BaseException as abandon_error:
                write_error.add_note(f"Abandon after failed scene write also failed: {abandon_error!r}")
            raise
        try:
            if seed:
                epoch, reference, dt = snapshot

                def commit(matrices):
                    self._write_anchor_matrices(matrices, ordinal)
                    self._ordinal_ledger.publish(ordinal)

                if not self._external_anchor.commit_external_pose(reference, dt, epoch, commit):
                    raise RuntimeError("XR anchor reset changed during complete scene publication")
            else:
                self._ordinal_ledger.publish(ordinal)
        except BaseException as publish_error:
            try:
                self._ordinal_ledger.abandon(ordinal)
            except BaseException as abandon_error:
                publish_error.add_note(f"Abandon after failed scene publication also failed: {abandon_error!r}")
            raise
        self._last_published_ordinal = ordinal
        self._anchor_snapshot = snapshot
        return ordinal

    def _write_anchor_matrices(self, matrices, ordinal: int) -> None:
        import numpy as np
        import ovstage
        from isaaclab_ov.stage import xform_tensor_from_numpy

        # SDP uses column vectors; the Stage's omni:xform stores USD row matrices.
        rows = np.ascontiguousarray(matrices.transpose(0, 2, 1), dtype=np.float64)
        tensor = xform_tensor_from_numpy(rows)
        self._anchor_stage.write_attribute(
            self._anchor_query,
            "omni:xform",
            ordinal=ordinal,
            tensors=tensor,
            semantic=ovstage.AttributeSemantic.MATRIX,
            is_array=False,
        ).wait()
        self._anchor_stage.write_attribute(
            self._anchor_query,
            "omni:resetXformStack",
            ordinal=ordinal,
            tensors=np.ones(len(rows), dtype=np.bool_),
            is_array=False,
        ).wait()

    def prepare_frame_scene(self) -> None:
        """Advance only from the latest committed A snapshot, once per begun XR frame."""
        if self._anchor_snapshot is None:
            return
        epoch, reference, dt = self._anchor_snapshot

        def commit(matrices):
            ordinal = self._ordinal_ledger.reserve()
            try:
                self._write_anchor_matrices(matrices, ordinal)
                # OVXR reads sealed anchor state at bind; bind reserves the next ordinal.
                self._ordinal_ledger.publish(ordinal)
            except BaseException as publication_error:
                try:
                    self._ordinal_ledger.abandon(ordinal)
                except BaseException as abandon_error:
                    publication_error.add_note(
                        f"Abandon after failed anchor publication also failed: {abandon_error!r}"
                    )
                raise

        self._external_anchor.commit_external_pose(reference, dt, epoch, commit)

    def set_external_anchor_state(self, state) -> None:
        """Replace a replay device's policy while preserving native query ownership."""
        if self._external_anchor is not None:
            # A new replay manager must not revive a failed borrowed native pair.
            _ = self._external_anchor.external_epoch
        if (state is None) != (self._external_anchor is None) or (
            state is not None and state.external_anchor_paths != self._external_anchor.external_anchor_paths
        ):
            raise ValueError("A reused XR environment must preserve its anchor paths")
        self._external_anchor = state
        self._anchor_snapshot = None

    def validate_first_published_scene(self, ordinal: int) -> None:
        if self._lease is None:
            raise RuntimeError("OVRTX/OVStage lease is not attached")
        with self._lease.ovstage_owner_lane():
            self._lease.validate_external_scene_partitions(ordinal)

    def get_target_frame_T_world(self, prim_path: str):
        return self._bootstrap.get_target_frame_T_world(prim_path)

    def close_publication(self) -> None:
        if self._publication_closed:
            return
        self._publication_closed = True
        if self._scene_publisher is not None:
            self._scene_publisher.close()

    def close_on_render_thread(self) -> None:
        first_error: BaseException | None = None
        try:
            self._anchor_resources.close()
        except BaseException:
            self._anchor_retirement_failed = True
            raise
        if self._anchor_retirement_failed:
            raise RuntimeError("Anchor query retirement is unproved; retaining its borrowed Stage")
        if self._ordinal_ledger is not None:
            try:
                self._ordinal_ledger.destroy()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
            finally:
                self._ordinal_ledger = None
        try:
            self._bootstrap.close()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
        if first_error is not None:
            raise first_error

    def close_environment_on_render_thread(self) -> None:
        if self._anchor_retirement_failed:
            raise RuntimeError("Anchor query retirement is unproved; retaining its environment")
        self._env.close()

    def abort(self) -> None:
        self._bootstrap.abort()


class OvLibrariesXrSession:
    """Thin Isaac Lab adapter around IsaacTeleop's OVXR session."""

    def __init__(self, bootstrap: OvxrSessionBootstrap, cfg: OvLibrariesXrConfig):
        self._bootstrap = bootstrap
        self._cfg = cfg
        self._owner = None
        self._host = None
        self._closed = False

    @classmethod
    def prepare(
        cls,
        sim_cfg: Any,
        cfg: OvLibrariesXrConfig,
        renderer_cfg: OVRTXRendererCfg | None = None,
    ) -> OvLibrariesXrSession:
        """Snapshot session selection, create simulation and reserve its OVRTX lease.

        Renderer settings are process-global on the stock renderer interface;
        use one configured session per process. No supported readback exists
        here to restore a previous renderer's settings on close.
        """
        validate_ovxr_session_preparation()
        # Stock settings are applied before Renderer construction. Preserve explicit caller choices.
        _import_ovxr().apply_renderer_settings(cfg.renderer_settings.strip())
        cfg = replace(cfg)
        bootstrap = OvxrSessionBootstrap(sim_cfg, renderer_cfg)
        try:
            return cls(bootstrap, cfg)
        except BaseException as preparation_error:
            try:
                bootstrap.abort()
            except BaseException as abort_error:
                # Cleanup follows the failure; retain its original cause.
                preparation_error.add_note(f"Cleanup after failed XR preparation also failed: {abort_error!r}")
            raise

    def start_render_worker(
        self,
        env: Any,
        *,
        tracking_anchor_path: str | None = None,
        required_openxr_extensions: Iterable[str] = (),
        hidden_visual_prims: Iterable[str] = (),
        max_blank_frames: int = 1,
        timeout_s: float = 30.0,
        external_anchor_state=None,
    ) -> bool:
        """Give IsaacTeleop the application pair and report whether XR started."""
        self._require_open()
        if self._owner is not None:
            raise RuntimeError("OVXR async render worker is already started")
        from isaaccapture.ovxr_session_manager.ovxr_session import OvxrSession, OvxrSessionConfig

        from .session_lifecycle import cloudxr_eula_accepted

        anchor_path = self._cfg.tracking_anchor_path if tracking_anchor_path is None else tracking_anchor_path
        host = _IsaacLabOvxrApplicationInterface(
            self._bootstrap, env, self._cfg, hidden_visual_prims, external_anchor_state
        )
        self._host = host
        self._owner = OvxrSession(
            host,
            OvxrSessionConfig(
                application_name=self._cfg.application_name,
                session_mode=self._cfg.session_mode,
                cloudxr_env_file=self._cfg.cloudxr_env_file,
                auto_launch_cloudxr=self._cfg.auto_launch_cloudxr,
                accept_eula=cloudxr_eula_accepted(),
                render_products=self._cfg.render_products,
                camera_paths=self._cfg.camera_paths,
                create_render_products=self._cfg.create_render_products,
                meters_per_unit=self._cfg.meters_per_unit,
                depth_near_z=self._cfg.depth_near_z,
                depth_far_z=self._cfg.depth_far_z,
                tracking_anchor_path=anchor_path,
                floor_aligned_anchor=False,
                host_wait_render_output=self._cfg.host_wait_render_output,
                extra_openxr_extensions=self._cfg.extra_openxr_extensions,
            ),
            required_openxr_extensions=required_openxr_extensions,
            max_blank_frames=max_blank_frames,
        )
        return self._owner.start(timeout_s=timeout_s)

    def external_oxr_handles(self):
        """Return IsaacTeleop's typed external handles once its owner is ready."""
        self._require_open()
        return None if self._owner is None else self._owner.external_oxr_handles()

    def request_render(self, physics_step_id: Hashable, *, force: bool = False) -> int | None:
        """Offer one complete application generation for IsaacTeleop to render.

        Return the admitted ordinal, or ``None`` when render demand is already
        satisfied. Retry the latest dirty generation until admitted, including
        while paused. ``force`` preserves reset/lifecycle discontinuities.
        """
        self._require_started()
        return self._owner.request_render(physics_step_id, force=force)

    def get_target_frame_T_world(self, prim_path: str):
        """Read a target transform through the thin simulation host adapter."""
        self._require_started()
        return self._owner.get_target_frame_T_world(prim_path)

    def close(self) -> None:
        """Close IsaacTeleop's owner before releasing the application environment."""
        if self._closed:
            return
        if self._owner is None:
            self._bootstrap.abort()
        else:
            self._owner.close()
        self._closed = True

    @property
    def async_worker_owns_environment(self) -> bool:
        """Whether IsaacTeleop accepted environment teardown ownership."""
        return self._owner is not None and self._owner.owns_environment

    @property
    def shutdown_requested(self) -> bool:
        """Whether the OVXR session requested application shutdown."""
        return self._owner is not None and self._owner.shutdown_requested

    def request_shutdown(self) -> None:
        """Request orderly OVXR shutdown."""
        if self._owner is not None:
            self._owner.request_shutdown()

    def set_external_anchor_state(self, state) -> None:
        """Adopt the current replay device's anchor policy between runs."""
        self._require_started()
        with self._host.publication_guard():
            self._host.set_external_anchor_state(state)

    def wait_for_shutdown(self, timeout_s: float) -> bool:
        """Wait for application cancellation without delaying orderly teardown."""
        self._require_started()
        return self._owner.wait_for_shutdown(timeout_s)

    @property
    def render_worker_bootstrap_ready(self) -> bool:
        """Whether IsaacTeleop claimed the renderer before peer-gated startup."""
        return self._owner is not None and self._owner.render_worker_bootstrap_ready

    def abort(self) -> None:
        """Tear down a failed pre-environment bootstrap."""
        self.close()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("OVXR session is closed")

    def _require_started(self) -> None:
        self._require_open()
        if self._owner is None:
            raise RuntimeError("OVXR session is not started")

    def __enter__(self) -> OvLibrariesXrSession:
        self._require_open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
