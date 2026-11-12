# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVXR session lifecycle shared by the live and replay OVXR entrypoints."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab_ov.renderers import OVRTXRendererCfg

    from ..ov_libraries_xr import OvLibrariesXrSession
    from ..ovxr_cfg import OvLibrariesXrCameraCfg


class OvxrPresentation:
    """Own one OVXR session from preparation through render-worker startup, publication and close.

    The input device that borrows the session's OpenXR handles is assigned to :attr:`input_device`
    once it exists. :meth:`start` reads its anchors, and :meth:`close` releases the session only
    after that device drained.
    """

    def __init__(self, session: OvLibrariesXrSession, *, excluded_visual_meshes: Iterable[str] = ()) -> None:
        self.session = session
        """Prepared OVXR session."""
        self._excluded_visual_meshes = excluded_visual_meshes
        self.input_device: Any = None
        """Input device that borrows the session, or ``None`` before it is created."""
        self.started = False
        """Whether the render worker started."""
        self._reset_generation = 0
        self._pending: tuple[int, int] | None = None
        self._force = False

    @classmethod
    def prepare(
        cls,
        env_cfg: Any,
        args: argparse.Namespace,
        *,
        application_name: str,
        auxiliary_cameras: tuple[OvLibrariesXrCameraCfg, ...] = (),
        renderer_cfg: OVRTXRendererCfg | None = None,
    ) -> OvxrPresentation:
        """Prepare the OVXR session before the environment is constructed.

        Args:
            env_cfg: Resolved environment configuration. Its ``isaac_teleop.xr_cfg`` selects the visual
                meshes excluded from the XR view.
            args: Parsed namespace with ``ovxr_session_mode``, ``cloudxr_env``, ``auto_launch_cloudxr``
                and the shared OVXR presentation options (see
                :func:`~isaaclab_teleop.entrypoints._ovxr_launch.add_ovxr_presentation_args`).
            application_name: OpenXR application name.
            auxiliary_cameras: Resolved PiP camera panels. The session renders with their shared OVRTX
                configuration.
            renderer_cfg: OVRTX configuration without camera panels. Defaults to
                :class:`~isaaclab_ov.renderers.OVRTXRendererCfg`.

        Returns:
            The prepared presentation.
        """
        from isaaclab_ov.renderers import OVRTXRendererCfg

        from .. import OvLibrariesXrCfg, OvLibrariesXrSession

        if auxiliary_cameras:
            # Camera selection assigned one OVRTX configuration to every panel; the session shares it.
            renderer_cfg = auxiliary_cameras[0].camera_cfg.renderer_cfg
        elif renderer_cfg is None:
            renderer_cfg = OVRTXRendererCfg()
        session = OvLibrariesXrSession.prepare(
            env_cfg.sim,
            OvLibrariesXrCfg(
                application_name=application_name,
                session_mode=args.ovxr_session_mode,
                cloudxr_env_file=args.cloudxr_env,
                auto_launch_cloudxr=args.auto_launch_cloudxr,
                desktop_mirror_enabled=args.ovxr_desktop_mirror,
                desktop_mirror_max_fps=args.ovxr_desktop_mirror_fps,
                auxiliary_cameras=auxiliary_cameras,
            ),
            renderer_cfg=renderer_cfg,
        )
        return cls(session, excluded_visual_meshes=env_cfg.isaac_teleop.xr_cfg.excluded_visual_meshes)

    def start(self, env: Any, *, timeout_s: float, required_openxr_extensions: Iterable[str] = ()) -> bool:
        """Start the render worker for :attr:`input_device`, or adopt its anchor once started.

        Args:
            env: Constructed environment sharing the prepared simulation context.
            timeout_s: Maximum worker startup wait [s].
            required_openxr_extensions: Extensions required by the input pipeline.

        Returns:
            Whether the render worker runs. ``False`` means the session shut down during startup.
        """
        device = self.input_device
        if self.started:
            self.session.set_external_anchor_state(device.external_anchor_state)
            return True
        self.started = self.session.start_render_worker(
            env,
            tracking_anchor_path=device.xr_render_anchor_path,
            required_openxr_extensions=required_openxr_extensions,
            excluded_visual_meshes=self._excluded_visual_meshes,
            external_anchor_state=device.external_anchor_state,
            timeout_s=timeout_s,
        )
        return self.started

    @property
    def reset_pending(self) -> bool:
        """Whether a reset publication still waits for admission."""
        return self._force

    def mark_changed(self, env: Any, *, reset: bool = False) -> None:
        """Record that the scene changed, without requesting a render.

        Args:
            env: Environment whose physics step identifies the publication.
            reset: Whether the change is a reset. Its publication is forced and retained until admitted.
        """
        if reset:
            self._reset_generation += 1
            self._force = True
        self._pending = (self._reset_generation, env.sim.get_physics_step_count())

    def publish(self, env: Any, *, changed: bool = False, reset: bool = False, report_timing: bool = True) -> bool:
        """Offer the latest unadmitted scene to the render worker.

        Args:
            env: Environment whose physics step identifies the publication.
            changed: Whether the scene changed since the last call.
            reset: Whether the change is a reset (see :meth:`mark_changed`).
            report_timing: Whether this call completes a host frame, whose physics rate the
                desktop mirror reports.

        Returns:
            Whether no publication remains pending.
        """
        if changed or reset:
            self.mark_changed(env, reset=reset)
        if self._pending is not None and self.session.request_render(self._pending, force=self._force) is not None:
            self._pending = None
            self._force = False
        if report_timing:
            self.session.report_host_timing(env.sim.get_physics_step_count(), env.sim.get_physics_dt())
        return self._pending is None

    def close(self) -> None:
        """Close the session once :attr:`input_device` drained; otherwise only request shutdown.

        Raises:
            RuntimeError: If the input device failed to drain. The session and the environment stay
                borrowed, so the caller must not release them.
        """
        if getattr(self.input_device, "session_cleanup_failed", False) is True:
            error = RuntimeError("OVXR input did not drain; retaining the borrowed XR session and environment")
            try:
                self.session.request_shutdown()
            except Exception as shutdown_error:
                error.add_note(f"Requesting XR shutdown also failed: {shutdown_error!r}")
            raise error
        self.session.close()


def update_idle_presentation(env: Any, presentation: OvxrPresentation | None, *, report_timing: bool = True) -> None:
    """Service a paused scene through its presentation owner without stepping physics.

    OVXR owns the leased renderer. Independent visualizers can be serviced here
    when supported, without calling the simulation's competing render path.
    """
    if presentation is None:
        env.sim.render()
    else:
        presentation.publish(env, report_timing=report_timing)


def admit_reset_publication(env: Any, presentation: OvxrPresentation | None) -> bool:
    """Allow input only after the current reset epoch has been admitted.

    A false return yields one host iteration, keeping shutdown and replay timeout
    checks active while a producer slot is unavailable. This is an admission fence,
    not a wait for the displayed frame.
    """
    if presentation is None or not presentation.reset_pending:
        return True
    if presentation.publish(env, report_timing=False):
        return True
    presentation.session.wait_for_shutdown(env.step_dt)
    return False
