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


class OvxrPresentation:
    """Own one OVXR session from preparation through render-worker startup, publication and close.

    The input device that borrows the session's OpenXR handles is assigned to :attr:`input_device`
    once it exists. :meth:`start` reads its anchors, and :meth:`close` releases the session only
    after that device drained.
    """

    def __init__(self, session: OvLibrariesXrSession) -> None:
        self.session = session
        """Prepared OVXR session."""
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
        renderer_cfg: OVRTXRendererCfg | None = None,
    ) -> OvxrPresentation:
        """Prepare the OVXR session before the environment is constructed.

        Args:
            env_cfg: Resolved environment configuration.
            args: Parsed namespace with ``ovxr_session_mode``, ``cloudxr_env`` and ``auto_launch_cloudxr``.
            application_name: OpenXR application name.
            renderer_cfg: OVRTX configuration. Defaults to :class:`~isaaclab_ov.renderers.OVRTXRendererCfg`.

        Returns:
            The prepared presentation.
        """
        from isaaclab_ov.renderers import OVRTXRendererCfg

        from .. import OvLibrariesXrCfg, OvLibrariesXrSession

        session = OvLibrariesXrSession.prepare(
            env_cfg.sim,
            OvLibrariesXrCfg(
                application_name=application_name,
                session_mode=args.ovxr_session_mode,
                cloudxr_env_file=args.cloudxr_env,
                auto_launch_cloudxr=args.auto_launch_cloudxr,
            ),
            renderer_cfg=OVRTXRendererCfg() if renderer_cfg is None else renderer_cfg,
        )
        return cls(session)

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

    def publish(self, env: Any, *, changed: bool = False, reset: bool = False) -> bool:
        """Offer the latest unadmitted scene to the render worker.

        Args:
            env: Environment whose physics step identifies the publication.
            changed: Whether the scene changed since the last call.
            reset: Whether the change is a reset (see :meth:`mark_changed`).

        Returns:
            Whether no publication remains pending.
        """
        if changed or reset:
            self.mark_changed(env, reset=reset)
        if self._pending is not None and self.session.request_render(self._pending, force=self._force) is not None:
            self._pending = None
            self._force = False
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
