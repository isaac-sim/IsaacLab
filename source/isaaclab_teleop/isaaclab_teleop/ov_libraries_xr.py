# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab host adapter for Isaac Capture-owned OVXR presentation."""

from __future__ import annotations

import contextlib
import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Hashable, Iterable
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
from isaaccapture.ovxr_session_manager import OvxrApplicationInterface, StageOrdinalLedger

from isaaclab.physics import PhysicsEvent

from .ovxr_cfg import OvLibrariesXrCameraFeedCfg, OvLibrariesXrCfg, OvLibrariesXrDesktopMirrorCfg
from .ovxr_session import OvxrSessionBootstrap, validate_ovxr_session_preparation

if TYPE_CHECKING:
    from isaaccapture.ovxr_session_manager.ovxr_session import OvxrSessionBinding
    from isaacteleop.oxr import OpenXRSessionHandles

    from isaaclab.sim import SimulationCfg, SimulationContext

    from .ovxr_renderer import OvxrRendererCfg

__all__ = ["OvLibrariesXrCameraFeedCfg", "OvLibrariesXrCfg", "OvLibrariesXrDesktopMirrorCfg", "OvLibrariesXrSession"]

logger = logging.getLogger(__name__)

_STARTUP_STATUS_INTERVAL_S = 30.0

_STARTUP_STATUS_POLL_S = 1.0
"""Seconds between checks for the end of attachment, which bounds the delay of the ready line."""


class OvLibrariesXrSession:
    """Isaac Lab adapter around Isaac Capture's OVXR session.

    The host owns the environment. Stop the XR input borrowers, close this session, and close the
    environment and desktop windows on the host thread once :attr:`retirement_complete` holds.
    Worker termination alone is insufficient.

    :meth:`prepare` also registers the session's retirement as the first
    :attr:`~isaaclab.physics.PhysicsEvent.STOP` listener of the kitless simulation, which dispatches STOP
    when its context is cleared, for example by closing the environment. Isaac Capture's render worker
    therefore drains and the render lease is released before cameras, renderers, the native OVRTX engine
    and the OVStage close. The listener closes a session that is still open only if it handed out no
    OpenXR handles, since only the host can confirm that their input borrowers stopped. A STOP while the
    simulation runs, from a hard reset or a physics re-warm, raises instead and leaves the session open:
    the raise ends that dispatch before anything is released.

    When the listener cannot prove retirement during a shutdown, it keeps the leased renderer and OVStage
    alive for the rest of the process and raises, so the simulation's teardown destroys neither under a
    live render owner. The pair cannot be reused; recreate the session in a new process.
    """

    @runtime_checkable
    class ExternalAnchorState(Protocol):
        """Dynamic XR-anchor policy sealed together with complete scene publications.

        ``isaaclab_teleop.xr_anchor_manager.ExternalXrAnchor`` implements this protocol. It
        drives the anchor of an OVXR session with ``XrCfg.anchor_prim_path`` set,
        which is when :attr:`~isaaclab_teleop.IsaacTeleopDevice.external_anchor_state` returns it.
        """

        @property
        def external_anchor_paths(self) -> tuple[str, ...]:
            """Anchor Xform paths written together at each committed pose."""
            ...

        @property
        def external_reference_path(self) -> str:
            """Absolute path of the physics prim whose world transform drives the anchor."""
            ...

        @property
        def external_epoch(self) -> tuple[int, bool]:
            """Reset identity and whether that epoch still needs its seeding publication.

            Raises:
                RuntimeError: If a previous anchor publication failed.
            """
            ...

        def commit_external_pose(
            self,
            reference: np.ndarray,
            rendering_dt: float,
            epoch: int,
            commit: Callable[[np.ndarray], None],
        ) -> bool:
            """Advance the policy from ``reference`` and seal its matrices through ``commit``.

            Args:
                reference: World transform of :attr:`external_reference_path` [4, 4].
                rendering_dt: Elapsed XR-frame time since the last committed pose [s].
                epoch: Reset identity read from :attr:`external_epoch`.
                commit: Callable that writes and seals the anchor matrices [N, 4, 4].

            Returns:
                ``False`` when ``epoch`` is stale and nothing was committed.
            """
            ...

    def __init__(self, bootstrap: OvxrSessionBootstrap, cfg: OvLibrariesXrCfg) -> None:
        self._bootstrap = bootstrap
        self._cfg = cfg
        self._owner = None
        self._desktop_mirror = None
        self._host = None
        self._closed = False
        self._handles_lent = False
        self._close_started = False
        self._stop_handled = False
        self._physics_rate = _CounterRate()

    def __enter__(self) -> OvLibrariesXrSession:
        self._require_open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @property
    def retirement_complete(self) -> bool:
        """Whether all XR borrowers retired, including after a failed startup.

        This remains readable after close succeeds or raises. A stopped or joined
        worker alone does not authorize destruction of the borrowed environment.
        """
        return self._bootstrap.closed if self._owner is None else self._owner.retirement_complete

    @property
    def shutdown_requested(self) -> bool:
        """Whether the OVXR session requested application shutdown."""
        return self._owner is not None and self._owner.shutdown_requested

    def start_render_worker(
        self,
        env: Any,
        *,
        tracking_anchor_path: str | None = None,
        required_openxr_extensions: Iterable[str] = (),
        excluded_visual_meshes: Iterable[str] = (),
        max_blank_frames: int = 1,
        timeout_s: float = 30.0,
        external_anchor_state: OvLibrariesXrSession.ExternalAnchorState | None = None,
    ) -> bool:
        """Give Isaac Capture the application pair and report whether XR started.

        Args:
            env: Constructed environment sharing the prepared simulation context.
            tracking_anchor_path: Override the configured USD tracking-anchor path.
            required_openxr_extensions: Extensions required by the input pipeline.
            excluded_visual_meshes: Mesh patterns hidden from the XR render export.
            max_blank_frames: Nonnegative integer bootstrap-frame count; zero skips bootstrap frames.
            timeout_s: Maximum worker startup wait [s].
            external_anchor_state: Dynamic anchor policy seeded from committed scene publications.

        Raises:
            RuntimeError: If the session is closed or already started, or ``env`` does not reuse the
                prepared simulation context, has no completed stage or not exactly one environment,
                selects a different simulation device or physics time step, or already has a prim of
                another type at a configured OVXR prim path.
            TypeError: If ``external_anchor_state`` does not implement :class:`ExternalAnchorState`.
            ValueError: If ``max_blank_frames`` is not a nonnegative integer, selected camera feeds
                use eager scene sensor updates, or a configured OVXR prim path is not absolute or is
                requested with two types.
        """
        self._require_open()
        if self._owner is not None:
            raise RuntimeError("OVXR async render worker is already started")
        if isinstance(max_blank_frames, bool) or not isinstance(max_blank_frames, int) or max_blank_frames < 0:
            raise ValueError("max_blank_frames must be a non-negative integer")
        _require_external_anchor_state(external_anchor_state)
        # Reject an inconsistent environment or configuration before the worker claims the single-use
        # transfer and declares the eye cameras.
        self._bootstrap.validate_environment(env)
        if self._cfg.camera_feeds and not env.unwrapped.scene.cfg.lazy_sensor_update:
            raise ValueError(
                "OVXR camera feeds require scene.lazy_sensor_update=True; eager sensor updates"
                " would capture cameras whose renderer is owned by the XR worker"
            )
        _resolve_ovxr_runtime_prims(env, self._cfg)
        from isaaccapture.ovxr_session_manager.ovxr_session import (
            OvxrSession,
            OvxrSessionConfig,
        )

        from .session_lifecycle import cloudxr_eula_accepted

        anchor_path = self._cfg.tracking_anchor_path if tracking_anchor_path is None else tracking_anchor_path
        host = _IsaacLabOvxrApplicationInterface(
            self._bootstrap,
            env,
            self._cfg,
            excluded_visual_meshes,
            external_anchor_state,
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
        try:
            with _StartupStatusReporter(self._owner):
                return self._owner.start(timeout_s=timeout_s)
        finally:
            self._desktop_mirror = host.desktop_mirror

    def external_oxr_handles(self) -> OpenXRSessionHandles | None:
        """Return Isaac Capture's typed external handles once its owner is ready."""
        self._require_open()
        handles = None if self._owner is None else self._owner.external_oxr_handles()
        if handles is not None:
            self._handles_lent = True
        return handles

    def request_render(self, physics_step_id: Hashable, *, force: bool = False) -> int | None:
        """Offer one complete application generation for Isaac Capture to render.

        Return the admitted ordinal, or ``None`` when render demand is already
        satisfied. Retry the latest dirty generation until admitted, including
        while paused. ``force`` preserves reset/lifecycle discontinuities.
        """
        self._require_started()
        return self._owner.request_render(physics_step_id, force=force)

    def report_host_timing(self, physics_step_count: int, physics_dt_s: float | None = None) -> None:
        """Publish physics counter and optional simulation timestep [s] to the mirror."""
        self._require_started()
        if self._desktop_mirror is None:
            return
        self._desktop_mirror.record_host_frame()
        physics_hz = self._physics_rate.sample(physics_step_count)
        if physics_dt_s is not None and math.isfinite(physics_dt_s) and physics_dt_s > 0.0:
            target_hz = 1.0 / physics_dt_s
            physics_text = f"Physics {physics_hz:5.1f}/{target_hz:.0f} Hz ({physics_hz / target_hz:.2f}× RT)"
        else:
            physics_text = f"Physics {physics_hz:5.1f} Hz"
        mirror_status = self._desktop_mirror.status
        if mirror_status is not None:
            self._desktop_mirror.set_status_text(
                f"Main {mirror_status.main_thread_fps:5.1f} FPS   "
                f"{physics_text}   Render {mirror_status.render_thread_fps:5.1f} FPS"
            )

    def get_target_frame_T_world(self, prim_path: str) -> np.ndarray:
        """Read world-to-target transform [4, 4], float32, with translation in scene units."""
        self._require_started()
        return self._owner.get_target_frame_T_world(prim_path)

    def close(self) -> None:
        """Retire XR borrowers before the host releases the environment.

        Failed retirement leaves the session open, and closing the environment then keeps the
        leased renderer and OVStage alive. Stop all input borrowers before calling, including after
        worker failure.

        A render worker that is still attaching the scene cannot be cancelled and uses the environment,
        so this call waits for the attachment to end, however long it takes. When the retirement
        completed despite a worker failure, only the first call raises that failure.
        """
        if self._closed:
            return
        self._close_started = True
        try:
            if self._owner is None:
                self._bootstrap.close()
            else:
                self._close_owner()
        except BaseException:
            # Isaac Capture raises a latched worker error from every close. A completed retirement is
            # final, so report that error once.
            self._closed = self.retirement_complete
            raise
        if not self.retirement_complete:
            raise RuntimeError("OVXR retirement is unproved; retaining its borrowed renderer and Stage")
        self._closed = True

    def _close_owner(self) -> None:
        """Close Isaac Capture's owner, outlasting its bounded join while the render worker attaches.

        Isaac Capture stops joining its worker after 30 s. Returning then would let the host close the
        environment and simulation that an attaching worker still reads.
        """
        waiting = False
        while True:
            try:
                self._owner.close()
                return
            except TimeoutError:
                if self._owner.render_worker_bootstrap_ready or not _render_worker_alive(self._owner):
                    raise
            if not waiting:
                waiting = True
                logger.warning(
                    "The OVXR render worker is still attaching the scene, which cannot be interrupted;"
                    " waiting for it to finish before the simulation closes."
                )

    def request_shutdown(self) -> None:
        """Request orderly OVXR shutdown."""
        if self._owner is not None:
            self._owner.request_shutdown()

    def set_external_anchor_state(self, state: OvLibrariesXrSession.ExternalAnchorState | None) -> None:
        """Adopt the current replay device's anchor policy between runs."""
        self._require_started()
        _require_external_anchor_state(state)
        with self._host.publication_guard():
            self._host.set_external_anchor_state(state)

    def wait_for_shutdown(self, timeout_s: float) -> bool:
        """Wait up to timeout_s [s]; return whether application cancellation was requested."""
        self._require_started()
        return self._owner.wait_for_shutdown(timeout_s)

    @classmethod
    def prepare(
        cls,
        sim_cfg: SimulationCfg,
        cfg: OvLibrariesXrCfg,
        renderer_cfg: OvxrRendererCfg | None = None,
    ) -> OvLibrariesXrSession:
        """Snapshot session selection, create simulation and reserve its OVRTX lease.

        Renderer settings are process-global on the stock renderer interface;
        use one configured session per process. No supported readback exists
        here to restore a previous renderer's settings on close.

        Args:
            sim_cfg: Simulation configuration used to construct the context immediately
                before the environment. The environment must reuse that exact context.
            cfg: XR session configuration, snapshotted before native preparation.
            renderer_cfg: OVXR renderer registered in that context. Camera panels must use the same
                configuration. ``None`` selects the default :class:`~isaaclab_teleop.OvxrRendererCfg`.

        Returns:
            A prepared session whose lease is reserved but has not transferred to a worker. The
            simulation retires it on STOP, before releasing its cameras and renderers.

        Raises:
            RuntimeError: If a simulation context already exists or extension mode is active.
            ValueError: If the XR configuration is invalid.
        """
        cfg = replace(cfg, camera_feeds=tuple(cfg.camera_feeds))
        validate_ovxr_session_preparation()
        import ovxr

        bootstrap = OvxrSessionBootstrap(sim_cfg, renderer_cfg)
        try:
            for camera in cfg.camera_feeds:
                bootstrap.configure_auxiliary_camera(camera.camera_cfg, max_fps=camera.max_fps)
            # Process-global settings change only after preparation succeeds, and they precede
            # the native renderer, which the session creates next.
            ovxr.apply_renderer_settings(cfg.renderer_settings.strip())
            bootstrap.create_native_pair()
            session = cls(bootstrap, cfg)
            sim = bootstrap.simulation_context
            # Retire XR before every Isaac Lab STOP listener and the native backends close. The closure
            # keeps the session alive until the simulation closes, even if its caller drops it.
            sim.backend_manager.register_callback(
                lambda _payload: session._retire_on_simulation_stop(sim),
                PhysicsEvent.STOP,
                order=-100,
                name="ovxr_session_retirement",
            )
        except BaseException as preparation_error:
            try:
                bootstrap.abort()
            except BaseException as abort_error:
                # Cleanup follows the failure; retain its original cause.
                preparation_error.add_note(f"Cleanup after failed XR preparation also failed: {abort_error!r}")
            raise
        return session

    def _retire_on_simulation_stop(self, sim: SimulationContext) -> None:
        """Close the session before the simulation releases its cameras, renderers and native backends.

        Retires once, and not at all after a retirement that already completed. If retirement fails or
        is refused while the lease is still held, the bootstrap keeps the leased pair alive (see the class
        description) before the error propagates.

        Raises:
            RuntimeError: If the simulation is still running, the session handed out OpenXR handles
                but was never closed, or its retirement is unproved.
        """
        if self._closed or self._stop_handled or self.retirement_complete:
            return
        if not sim.is_stopped():
            # This raise ends the reset's STOP dispatch before any listener releases the renderer, so
            # the pair stays usable and a later shutdown can still retire the session.
            raise RuntimeError(
                "The simulation stopped physics while running (a hard reset or physics re-warm), which"
                " releases the OVXR renderer; close the OVXR session first"
            )
        self._stop_handled = True
        try:
            if self._handles_lent and not self._close_started:
                raise RuntimeError(
                    "The simulation is closing while XR input may still borrow the OVXR session; stop its input"
                    " and close the session first"
                )
            self.close()
        except BaseException:
            if not self._bootstrap.closed:
                self._bootstrap._retain_native_pair()
            raise

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("OVXR session is closed")

    def _require_started(self) -> None:
        self._require_open()
        if self._owner is None:
            raise RuntimeError("OVXR session is not started")


def _render_worker_alive(owner: Any) -> bool:
    """Whether the render worker of Isaac Capture's session owner is still running."""
    worker_id = owner.render_worker_native_thread_id
    return worker_id is not None and any(thread.native_id == worker_id for thread in threading.enumerate())


class _StartupStatusReporter:
    """Log render-worker start-up progress from a side thread while the application thread waits for it.

    The waiting thread is blocked inside Isaac Capture's start, so it cannot log. The reporter logs
    nothing once the wait ends without the worker becoming ready, whether it failed or the application
    shut down.
    """

    def __init__(self, owner: Any) -> None:
        self._owner = owner
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="ovxr-startup-status", daemon=True)
        self._started_s = 0.0
        self._ready_logged = False

    def __enter__(self) -> _StartupStatusReporter:
        self._started_s = time.monotonic()
        self._thread.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self._stop.set()
        self._thread.join()
        # The wait can end between two polls.
        self._log_ready_once()

    def _log_ready_once(self) -> None:
        if self._ready_logged or not self._owner.render_worker_bootstrap_ready:
            return
        self._ready_logged = True
        logger.info(
            "OVXR renderer ready after %.0f s. An XR client can connect once the CloudXR WSS proxy is listening.",
            time.monotonic() - self._started_s,
        )

    def _run(self) -> None:
        next_report_s = self._started_s + _STARTUP_STATUS_INTERVAL_S
        while not self._stop.wait(_STARTUP_STATUS_POLL_S):
            now_s = time.monotonic()
            if self._owner.render_worker_bootstrap_ready:
                self._log_ready_once()
                return
            if now_s >= next_report_s:
                logger.info("Still preparing the OVXR renderer (%.0f s).", now_s - self._started_s)
                next_report_s += _STARTUP_STATUS_INTERVAL_S


class _CounterRate:
    """Measure a cumulative application counter over a recent wall-clock window."""

    def __init__(self, window_s: float = 1.0) -> None:
        self._window_ns = int(window_s * 1_000_000_000)
        self._samples: deque[tuple[int, int]] = deque()

    def sample(self, count: int, now_ns: int | None = None) -> float:
        now_ns = time.perf_counter_ns() if now_ns is None else now_ns
        if self._samples and count < self._samples[-1][1]:
            self._samples.clear()
        self._samples.append((now_ns, count))
        while len(self._samples) > 2 and self._samples[1][0] <= now_ns - self._window_ns:
            self._samples.popleft()
        if len(self._samples) < 2:
            return 0.0
        elapsed_s = (self._samples[-1][0] - self._samples[0][0]) / 1_000_000_000
        count_delta = self._samples[-1][1] - self._samples[0][1]
        return 0.0 if elapsed_s <= 0.0 else max(0, count_delta) / elapsed_s


def _resolve_ovxr_runtime_prims(env: Any, cfg: OvLibrariesXrCfg) -> dict[str, str]:
    """Return the type of every type-only OVXR prim to declare, by path, without authoring anything.

    Only the eye cameras are declared: the view partitions are written on them before the first
    frame. OVXR authors its render products, render vars and render settings itself.

    Raises:
        RuntimeError: If the environment has no Stage, or a path already has a prim of another type.
        ValueError: If a path is not absolute or is requested with two types.
    """
    if not cfg.create_render_products:
        return {}

    unwrapped = getattr(env, "unwrapped", env)
    stage = unwrapped.scene.stage
    if stage is None:
        raise RuntimeError("OVXR runtime prim declaration requires an environment Stage")

    declarations = tuple((path, "Camera") for path in cfg.camera_paths)
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
    return expected_types


def _declare_ovxr_runtime_prims(env: Any, cfg: OvLibrariesXrCfg) -> None:
    """Declare type-only OVXR prims for the application Stage's lifetime."""
    declarations = _resolve_ovxr_runtime_prims(env, cfg)
    if declarations:
        stage = getattr(env, "unwrapped", env).scene.stage
        for path, type_name in declarations.items():
            stage.DefinePrim(path, type_name)


class _StageOrdinalLedger(StageOrdinalLedger):
    """Reserve explicit OVStage ordinals for the application publication lane."""

    def __init__(self, stage: Any) -> None:
        import ovstage

        self._stage = stage
        self._state_lock = threading.Lock()
        self._active_ordinal: int | None = None
        self._closed = False
        # Ordinals are signed: a new Stage is sealed through the minimum, and the maximum is never written.
        self._ordinal_min = ovstage.ORDINAL_MIN
        self._ordinal_max = ovstage.ORDINAL_MAX
        self._floor = self._read_floor()

    def reserve(self) -> int:
        """Reserve exactly ``floor + 1`` while the application publication guard is held."""
        with self._state_lock:
            if self._closed:
                raise RuntimeError("OVStage ordinal ledger is closed")
            if self._active_ordinal is not None:
                raise RuntimeError(f"OVStage ordinal {self._active_ordinal} is still unresolved")
            if self._floor >= self._ordinal_max:
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

    def _read_floor(self) -> int:
        with self._stage.get_attribute_write_floor() as query:
            query.wait()
            floor = int(query.fetch())
        if not self._ordinal_min <= floor <= self._ordinal_max:
            raise RuntimeError(f"OVStage returned an invalid write floor: {floor}")
        return floor

    def _require_active(self, ordinal: int) -> None:
        if self._active_ordinal != ordinal:
            raise RuntimeError(
                f"OVStage ordinal {ordinal} is not the active application reservation ({self._active_ordinal=})"
            )


class _IsaacLabOvxrApplicationInterface(OvxrApplicationInterface):
    """Supply only the renderer/Stage/coordinator and simulation publication."""

    def __init__(
        self,
        bootstrap: OvxrSessionBootstrap,
        env: Any,
        cfg: OvLibrariesXrCfg,
        excluded_visual_meshes: Iterable[str] = (),
        external_anchor_state: OvLibrariesXrSession.ExternalAnchorState | None = None,
    ) -> None:
        self._bootstrap = bootstrap
        self._env = env
        self._cfg = cfg
        self._excluded_visual_meshes = tuple(excluded_visual_meshes)
        self._publication_lock = threading.Lock()
        self._scene_publisher = None
        self._lease = None
        self._ordinal_ledger = None
        self._last_published_ordinal: int | None = None
        self._publication_closed = False
        self._camera_feeds = cfg.camera_feeds
        self.desktop_mirror = None
        self._external_anchor = external_anchor_state
        self._anchor_snapshot: tuple[int, np.ndarray] | None = None
        self._last_anchor_frame_time: float | None = None
        self._anchor_resources = contextlib.ExitStack()
        self._anchor_query = None
        self._anchor_reset_flags = None
        self._anchor_stage = None
        self._anchor_retirement_failed = False

    def offer_render_owner_transfer(self) -> object:
        self._scene_publisher = self._bootstrap.capture_scene_publisher()
        return self._bootstrap.offer_render_owner_transfer()

    def attach_on_render_thread(self, capability: object) -> OvxrSessionBinding:
        from isaaccapture.ovxr_session_manager.ovxr_session import OvxrSessionBinding

        self._bootstrap.claim_render_owner_transfer(capability)
        unwrapped = self._env.unwrapped
        if self._camera_feeds:
            # Fail before native Stage binding and visual-exclusion export.
            sensors = unwrapped.scene.sensors
            for camera in self._camera_feeds:
                if sensors.get(camera.name) is None:
                    raise RuntimeError(f"Configured camera feed {camera.name!r} is not present in the environment")
        _declare_ovxr_runtime_prims(self._env, self._cfg)
        from .xr_visual_exclusions import exclude_visual_meshes

        stage = unwrapped.scene.stage
        environment_paths = unwrapped.scene.env_prim_paths
        if len(environment_paths) != 1:
            raise RuntimeError("OVXR visual exclusions require exactly one environment root")
        environment_root_path = environment_paths[0]
        with exclude_visual_meshes(stage, self._excluded_visual_meshes, environment_root_path=environment_root_path):
            self._lease = self._bootstrap.bind_environment(self._env)
        auxiliary_paths = ()
        if self._camera_feeds:
            auxiliary_paths = self._lease.auxiliary_render_product_paths
        with self._lease.ovstage_owner_lane():
            self._ordinal_ledger = _StageOrdinalLedger(self._lease.ovstage)
            self._bootstrap.configure_view_partitions(self._cfg.camera_paths, self._ordinal_ledger)
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
                    self._anchor_reset_flags = np.ones(len(self._external_anchor.external_anchor_paths), dtype=np.bool_)
                except BaseException as setup_error:
                    try:
                        self._anchor_resources.close()
                    except BaseException as cleanup_error:
                        self._anchor_retirement_failed = True
                        setup_error.add_note(f"Anchor query cleanup also failed: {cleanup_error!r}")
                    raise
        from isaaccapture.ovxr_session_manager.camera_panels import CameraPanelConfig, CameraPanelConsumer

        panels = tuple(
            CameraPanelConfig(
                render_product_path=path,
                label=camera.name,
                max_image_bytes=int(camera.camera_cfg.width) * int(camera.camera_cfg.height) * 4,
                panel_width_m=camera.panel_width_m,
                position=camera.position,
                orientation_xyzw=camera.orientation_xyzw,
                space=camera.space,
            )
            for path, camera in zip(auxiliary_paths, self._camera_feeds, strict=True)
        )
        consumers = [CameraPanelConsumer(panels)] if panels else []
        if self._cfg.desktop_mirror is not None:
            # Headless sessions never import the optional mirror modules.
            from isaaccapture.ovxr_session_manager.desktop_mirror import DesktopMirrorConfig
            from isaaccapture.ovxr_session_manager.owned_mirror import MirrorProduct, OwnedDesktopMirror

            self.desktop_mirror = OwnedDesktopMirror(
                DesktopMirrorConfig(
                    max_fps=self._cfg.desktop_mirror.max_fps,
                    initial_status_text="Waiting for headset",
                ),
                camera_products=tuple(MirrorProduct(panel.render_product_path, panel.label) for panel in panels),
            )
            consumers.append(self.desktop_mirror)
        return OvxrSessionBinding(
            renderer=self._lease.ovrtx_renderer,
            stage=self._lease.ovstage,
            ordinal_ledger=self._ordinal_ledger,
            frame_consumers=tuple(consumers),
            prepare_frame_scene=self.prepare_frame_scene if self._external_anchor is not None else None,
        )

    def publication_guard(self) -> contextlib.AbstractContextManager[None]:
        return self._publication_lock

    def owner_lane(self) -> contextlib.AbstractContextManager[None]:
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
        seed_pending = anchor_epoch is not None and (anchor_epoch[1] or self._anchor_snapshot is None)
        snapshot = None

        def prepare() -> None:
            nonlocal snapshot
            if self._camera_feeds:
                sensors = self._env.unwrapped.scene.sensors
                for camera in self._camera_feeds:
                    # OVXR presentation bypasses Camera's image-update path. Refresh only poses
                    # after simulation/reset kinematics, before publishing the matching scene.
                    # Camera has no public pose-only refresh; its pose update renders and counts no frame.
                    sensors[camera.name]._update_poses(env_ids=None)
            if anchor_epoch is not None:
                epoch, _ = anchor_epoch
                reference = self._bootstrap.get_world_transform(self._external_anchor.external_reference_path)
                snapshot = (epoch, reference)

        def seal(ordinal: int) -> None:
            if not seed_pending:
                self._ordinal_ledger.publish(ordinal)
                return
            epoch, reference = snapshot

            def commit(matrices: np.ndarray) -> None:
                self._write_anchor_matrices(matrices, ordinal)
                self._ordinal_ledger.publish(ordinal)

            # Reset seeding must be sealed with this scene before input resumes.
            # The first pose initializes the filter without an elapsed interval.
            if not self._external_anchor.commit_external_pose(reference, 0.0, epoch, commit):
                raise RuntimeError("XR anchor reset changed during complete scene publication")
            self._last_anchor_frame_time = None

        try:
            ordinal = self._scene_publisher.publish(
                publication_id, self._ordinal_ledger, prepare=prepare, seal=seal, force=seed_pending
            )
        except BaseException:
            # A failed write or seal closed the publisher and poisoned the pair; a rejected sample did not.
            if self._scene_publisher.closed:
                self.close_publication()
            raise
        if ordinal is None:
            if self._last_published_ordinal is None:
                raise RuntimeError("OVStage publication deduplicated before a sealed ordinal exists")
            return self._last_published_ordinal
        self._last_published_ordinal = ordinal
        self._anchor_snapshot = snapshot
        return ordinal

    def _write_anchor_matrices(self, matrices: np.ndarray, ordinal: int) -> None:
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
        # Fixed for this query's lifetime; OVStage retains the value across later ordinals.
        if self._anchor_reset_flags is not None:
            self._anchor_stage.write_attribute(
                self._anchor_query,
                "omni:resetXformStack",
                ordinal=ordinal,
                tensors=self._anchor_reset_flags,
                is_array=False,
            ).wait()
            self._anchor_reset_flags = None

    def prepare_frame_scene(self) -> None:
        """Advance the committed reference with elapsed XR-frame time [s].

        The first begun frame after reset establishes the monotonic clock with
        zero elapsed time. Scene publication never advances this frame clock.
        """
        if self._publication_closed:
            raise RuntimeError("IsaacLab scene publication is not available")
        if self._anchor_snapshot is None:
            return
        epoch, reference = self._anchor_snapshot
        now = time.monotonic()
        dt = 0.0 if self._last_anchor_frame_time is None else now - self._last_anchor_frame_time

        def commit(matrices: np.ndarray) -> None:
            ordinal = self._ordinal_ledger.reserve()
            try:
                self._write_anchor_matrices(matrices, ordinal)
                # OVXR reads sealed anchor state at bind; bind reserves the next ordinal.
                self._ordinal_ledger.publish(ordinal)
            except BaseException as publication_error:
                self._scene_publisher.invalidate()
                self.close_publication()
                try:
                    self._ordinal_ledger.abandon(ordinal)
                except BaseException as abandon_error:
                    publication_error.add_note(
                        f"Abandon after failed anchor publication also failed: {abandon_error!r}"
                    )
                raise

        if self._external_anchor.commit_external_pose(reference, dt, epoch, commit):
            self._last_anchor_frame_time = now

    def set_external_anchor_state(self, state: OvLibrariesXrSession.ExternalAnchorState | None) -> None:
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
        self._last_anchor_frame_time = None

    def validate_first_published_scene(self, ordinal: int) -> None:
        if self._lease is None:
            raise RuntimeError("OVRTX/OVStage lease is not attached")
        with self._lease.ovstage_owner_lane():
            self._lease.validate_live()

    def get_target_frame_T_world(self, prim_path: str) -> np.ndarray:
        return self._bootstrap.get_target_frame_T_world(prim_path)

    def close_publication(self) -> None:
        if self._publication_closed:
            return
        self._publication_closed = True
        if self._scene_publisher is not None:
            self._scene_publisher.close()

    def close_on_render_thread(self) -> None:
        if self._ordinal_ledger is not None:
            # Unresolved writes may still use the anchor queries and Stage.
            # Retain their resources; retirement requires proven completion.
            self._ordinal_ledger.destroy()
            self._ordinal_ledger = None
        try:
            self._anchor_resources.close()
        except BaseException:
            self._anchor_retirement_failed = True
            raise
        if self._anchor_retirement_failed:
            raise RuntimeError("Anchor query retirement is unproved; retaining its borrowed Stage")
        self._bootstrap.close()

    def close_environment_on_render_thread(self) -> None:
        """Verify XR retirement; the host closes the environment after the session closes.

        Capture calls this on its worker. A failure leaves the session's retirement unproved.
        """
        if self._ordinal_ledger is not None:
            raise RuntimeError("OVStage write retirement is unproved; retaining its renderer and Stage")
        if self._anchor_retirement_failed:
            raise RuntimeError("Anchor query retirement is unproved; retaining its renderer and Stage")
        if not self._bootstrap.closed:
            raise RuntimeError("Render lease retirement is unproved; retaining its renderer and Stage")

    def abort(self) -> None:
        # The caller owns an environment even when worker startup failed.
        # Release the lease before it closes that environment and its context.
        self._bootstrap.close()


def _require_external_anchor_state(state: object) -> None:
    """Reject anchor policies that lack the external-publication protocol on the caller's thread."""
    if state is None:
        return
    if not isinstance(state, OvLibrariesXrSession.ExternalAnchorState):
        raise TypeError(
            f"external_anchor_state must implement OvLibrariesXrSession.ExternalAnchorState; got {type(state).__name__}"
        )
    reference_path = state.external_reference_path
    if not isinstance(reference_path, str) or not reference_path.startswith("/"):
        raise ValueError(f"external_reference_path must be an absolute prim path; got {reference_path!r}")
