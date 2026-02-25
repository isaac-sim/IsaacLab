# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IsaacTeleop-based teleoperation device for Isaac Lab."""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch

from .command_handler import CommandHandler
from .isaac_teleop_cfg import IsaacTeleopCfg
from .session_lifecycle import TeleopSessionLifecycle
from .xr_anchor_manager import XrAnchorManager

logger = logging.getLogger(__name__)


class IsaacTeleopDevice:
    """A IsaacTeleop-based teleoperation device for Isaac Lab.

    This device provides an interface between IsaacTeleop's retargeting pipeline
    and Isaac Lab environments.  It composes three focused collaborators:

    * :class:`XrAnchorManager` -- XR anchor prim setup, synchronization,
      and coordinate-frame transform computation.
    * :class:`TeleopSessionLifecycle` -- pipeline building, OpenXR handle
      acquisition, session creation/destruction, and action-tensor extraction.
    * :class:`CommandHandler` -- callback registration and XR message-bus
      command dispatch.

    Together they manage:

    1. XR anchor configuration and synchronization
    2. IsaacTeleop session lifecycle
    3. Action tensor generation from the retargeting pipeline

    The device uses IsaacTeleop's TensorReorderer to flatten pipeline outputs
    into a single action tensor matching the environment's action space.

    Teleop commands:
        The device supports callbacks for START, STOP, and RESET commands
        that can be triggered via XR controller buttons or the message bus.

    Example:
        .. code-block:: python

            cfg = IsaacTeleopCfg(
                pipeline_builder=my_pipeline_builder,
                sim_device="cuda:0",
            )

            with IsaacTeleopDevice(cfg) as device:
                while running:
                    action = device.advance()
                    env.step(action.repeat(num_envs, 1))
    """

    def __init__(self, cfg: IsaacTeleopCfg):
        """Initialize the IsaacTeleop device.

        Args:
            cfg: Configuration object for IsaacTeleop settings.
        """
        self._cfg = cfg

        # Compose the three collaborators
        self._anchor_manager = XrAnchorManager(cfg.xr_cfg)
        self._session_lifecycle = TeleopSessionLifecycle(cfg)
        self._command_handler = CommandHandler(
            xr_core=self._anchor_manager.xr_core,
            on_reset=self._anchor_manager.reset,
        )

        # Controller button polling state (edge detection for right 'A')
        self._prev_right_a_pressed = False

    def __del__(self):
        """Clean up resources when the object is destroyed."""
        if hasattr(self, "_command_handler"):
            self._command_handler.cleanup()
        if hasattr(self, "_anchor_manager"):
            self._anchor_manager.cleanup()

    def __str__(self) -> str:
        """Returns a string containing information about the IsaacTeleop device."""
        xr_cfg = self._cfg.xr_cfg
        msg = f"IsaacTeleop Device: {self.__class__.__name__}\n"
        msg += f"\tAnchor Position: {xr_cfg.anchor_pos}\n"
        msg += f"\tAnchor Rotation: {xr_cfg.anchor_rot}\n"
        if xr_cfg.anchor_prim_path is not None:
            msg += f"\tAnchor Prim Path: {xr_cfg.anchor_prim_path} (Dynamic Anchoring)\n"
        else:
            msg += "\tAnchor Mode: Static (Root Level)\n"
        msg += f"\tSim Device: {self._cfg.sim_device}\n"
        msg += f"\tApp Name: {self._cfg.app_name}\n"

        msg += "\t----------------------------------------------\n"
        msg += "\tAvailable Commands:\n"
        callbacks = self._command_handler.callbacks
        start_avail = "START" in callbacks
        stop_avail = "STOP" in callbacks
        reset_avail = "RESET" in callbacks
        msg += f"\t\tStart Teleoperation: {'registered' if start_avail else 'not registered'}\n"
        msg += f"\t\tStop Teleoperation: {'registered' if stop_avail else 'not registered'}\n"
        msg += f"\t\tReset Environment: {'registered' if reset_avail else 'not registered'}\n"

        return msg

    def __enter__(self) -> IsaacTeleopDevice:
        """Enter the context manager and prepare the IsaacTeleop session.

        Builds the retargeting pipeline and attempts to acquire OpenXR handles
        from Kit's XR bridge extension.  If the handles are not yet available
        (e.g. the user has not clicked "Start AR"), session creation is deferred
        and will be retried automatically on each :meth:`advance` call.

        Returns:
            Self for context manager protocol.
        """
        self._session_lifecycle.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and clean up the IsaacTeleop session."""
        self._anchor_manager.cleanup()
        self._session_lifecycle.stop(exc_type, exc_val, exc_tb)
        return False

    def reset(self) -> None:
        """Reset the device state.

        Resets the XR anchor synchronizer if present.
        """
        self._anchor_manager.reset()

    def add_callback(self, key: str, func: Callable) -> None:
        """Add a callback function for teleop commands.

        Args:
            key: The command type to bind to. Valid values are "START", "STOP", "RESET", and "R".
            func: The function to call when the command is received. Should take no arguments.
        """
        self._command_handler.add_callback(key, func)

    def advance(self) -> torch.Tensor | None:
        """Process current device state and return control commands.

        If the IsaacTeleop session has not been started yet (because the OpenXR
        handles were not available at ``__enter__`` time), this method will
        attempt to start it on each call.  Once the user clicks "Start AR" and
        the handles become available, the session is created transparently.

        Returns:
            A flattened action :class:`torch.Tensor` ready for the Isaac Lab
            environment, or ``None`` if the session has not started yet
            (e.g. still waiting for the user to start AR).

        Raises:
            RuntimeError: If called outside of a context manager.
        """
        # Step the session (handles lazy start and action extraction)
        action = self._session_lifecycle.step(
            anchor_world_matrix_fn=self._anchor_manager.get_world_matrix,
        )

        if action is not None:
            # Poll controller buttons (e.g. toggle anchor rotation on right 'A' press)
            self._poll_buttons()

        return action

    # ------------------------------------------------------------------
    # Controller button polling (glue between session and anchor manager)
    # ------------------------------------------------------------------

    def _poll_buttons(self) -> None:
        """Poll controller buttons and trigger actions on rising edges.

        Called once per :meth:`advance` frame, after ``session.step()`` has
        already called ``deviceio_session.update()`` so the controller data
        is fresh.

        Currently handles:
            * Right controller primary button ("A") -- toggles anchor rotation.
        """
        controller_tracker = self._session_lifecycle.controller_tracker
        deviceio_session = self._session_lifecycle.deviceio_session

        if controller_tracker is None or deviceio_session is None:
            return

        controller_data = controller_tracker.get_controller_data(deviceio_session)
        right = controller_data.right_controller

        if right is not None and right.is_active:
            current = right.inputs.primary_click
            if current and not self._prev_right_a_pressed:
                self._anchor_manager.toggle_anchor_rotation()
            self._prev_right_a_pressed = current
        else:
            self._prev_right_a_pressed = False


def _enable_teleop_bridge() -> None:
    """Enable the XR teleop bridge extension and configure carb settings.

    Must be called after the Omniverse AppLauncher has started.
    """
    import carb.settings
    import omni.kit.app

    carb.settings.get_settings().set("/persistent/xr/openxr/disableInputBindings", True)
    carb.settings.get_settings().set('/xr/openxr/components/"isaacsim.kit.xr.teleop.bridge"/enabled', True)
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    ext_manager.set_extension_enabled_immediate("isaacsim.kit.xr.teleop.bridge", True)


def create_isaac_teleop_device(
    cfg: IsaacTeleopCfg,
    sim_device: str | None = None,
    callbacks: dict[str, Callable] | None = None,
) -> IsaacTeleopDevice:
    """Create an :class:`IsaacTeleopDevice` with required Omniverse extension setup.

    This helper centralises the boilerplate that every script must execute
    before constructing an :class:`IsaacTeleopDevice`:

    1. Disable default OpenXR input bindings (prevents conflicts).
    2. Enable the ``isaacsim.kit.xr.teleop.bridge`` extension.
    3. Optionally override :attr:`IsaacTeleopCfg.sim_device` so action tensors
       land on the same device the caller uses for the simulation.

    Note:
        When *sim_device* is provided, ``cfg.sim_device`` is mutated in place
        before the device is constructed.

    Args:
        cfg: IsaacTeleop configuration.
        sim_device: If provided, overrides ``cfg.sim_device`` so action tensors
            are placed on the requested torch device (e.g. ``"cuda:0"``).
        callbacks: Optional mapping of command keys (e.g. ``"START"``, ``"STOP"``,
            ``"RESET"``) to callables registered on the device.

    Returns:
        A fully configured :class:`IsaacTeleopDevice` ready for use in a
        ``with`` block.
    """
    _enable_teleop_bridge()

    if sim_device is not None:
        cfg.sim_device = sim_device

    logger.info("Using IsaacTeleop stack for teleoperation")
    device = IsaacTeleopDevice(cfg)

    if callbacks is not None:
        for key, func in callbacks.items():
            device.add_callback(key, func)

    return device
