# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Haptic feedback (controller vibration) for IsaacTeleop teleoperation.

This module adds an *output* path to the otherwise input-only teleop stack:
a sim-side contact force (e.g. a robot gripper pressing on an object) is
turned into an XR motion-controller vibration on the corresponding hand.

The design mirrors the existing input pipeline in reverse and keeps the four
concerns separated:

* **Sensing** lives in the environment -- a :class:`~isaaclab.sensors.ContactSensor`
  per hand, referenced by name from :class:`HapticFeedbackCfg`.
* **Output port** is the narrow :class:`HapticFeedbackReceiver` protocol.  Only
  devices that can vibrate implement it; input-only devices (keyboard, gamepad)
  are unaffected, so the base ``DeviceBase`` contract stays input-only.
* **Mapping + transport** lives inside :class:`~isaaclab_teleop.IsaacTeleopDevice`,
  which builds an ``isaacteleop`` ``HapticSink`` fed by a
  ``TactileVectorToControllerPulse`` -- Isaac Lab supplies the force, ``isaacteleop``
  owns the force->amplitude curve and the OpenXR ``xrApplyHapticFeedback`` write.
* **Orchestration** is :class:`HapticFeedbackDriver`, used identically by every
  teleop script: it reads the contact sensors after ``env.step`` and forwards a
  per-hand force scalar to the device.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Well-known endpoint identifiers for the two motion controllers. These match
# ``isaacteleop``'s ``Endpoint`` strings for hand-mounted haptic devices.
ENDPOINT_LEFT: str = "left"
ENDPOINT_RIGHT: str = "right"


@runtime_checkable
class HapticFeedbackReceiver(Protocol):
    """Protocol for a teleop device that can render haptic feedback.

    This is deliberately a single-method protocol rather than an addition to
    ``DeviceBase``: input-only devices (keyboard, space-mouse, gamepad) cannot
    honor a feedback call, so forcing the method onto every device would break
    Liskov substitutability.  Scripts guard with ``isinstance(device,
    HapticFeedbackReceiver)`` and only drive feedback when the active device
    actually supports it.
    """

    def send_haptic(self, endpoint: str, force: float) -> None:
        """Render a haptic pulse on one controller from a contact force.

        Args:
            endpoint: Which controller to vibrate -- :data:`ENDPOINT_LEFT` or
                :data:`ENDPOINT_RIGHT`.
            force: Contact-force magnitude [N] on that hand.  ``0`` stops any
                active pulse.  The force-to-amplitude mapping (gain, deadband,
                saturation) is applied downstream by the device.
        """
        ...


@configclass
class HapticFeedbackCfg:
    """Configuration for gripper-force controller haptics on a teleop env.

    Attach this to an environment config as a sibling of ``isaac_teleop`` (e.g.
    ``self.haptic_feedback = HapticFeedbackCfg(...)`` in ``__post_init__``).  The
    teleop scripts discover it via ``hasattr(env_cfg, "haptic_feedback")`` and
    build a :class:`HapticFeedbackDriver` from it.

    The force-to-amplitude curve fields (:attr:`gain`, :attr:`deadband`,
    :attr:`saturation`) parameterize ``isaacteleop``'s
    ``TactileVectorToControllerPulse`` and are consumed by the device when it
    builds the haptic sink.  The sensor-name fields are consumed by the driver
    when it reads forces from the scene.
    """

    left_sensor_name: str = "left_hand_contact"
    """Scene entity name of the :class:`~isaaclab.sensors.ContactSensor` whose net
    force drives the left controller."""

    right_sensor_name: str = "right_hand_contact"
    """Scene entity name of the :class:`~isaaclab.sensors.ContactSensor` whose net
    force drives the right controller."""

    gain: float = 0.05
    """Force-to-amplitude gain applied after the deadband, ``1/N``.

    The default maps roughly a 20 N grip to full-scale amplitude
    (``amplitude = clamp(gain * (force - deadband), 0, saturation)``)."""

    deadband: float = 0.5
    """Contact force [N] below which no vibration is produced (rejects sensor noise)."""

    saturation: float = 1.0
    """Upper clamp on the normalized amplitude in ``[0, 1]``."""

    frequency_hz: float = 0.0
    """Vibration frequency [Hz]. ``0`` selects the XR runtime's default frequency."""

    duration_s: float = 0.0
    """Pulse duration [s]. ``0`` selects the shortest pulse the runtime supports;
    the pulse is refreshed every frame while force persists."""


class HapticFeedbackDriver:
    """Reads per-hand contact forces from the scene and pushes them to the device.

    One driver is shared by every teleop script.  Construct it with
    :func:`create_haptic_feedback_driver` (which returns ``None`` when the env
    has no haptic config or the device cannot vibrate), then call
    :meth:`update` once per step *after* ``env.step`` so the sensors hold fresh
    post-physics contact data.
    """

    def __init__(self, env: ManagerBasedRLEnv, device: HapticFeedbackReceiver, cfg: HapticFeedbackCfg):
        """Initialize the driver.

        Args:
            env: The (unwrapped) environment whose scene owns the contact sensors.
            device: The teleop device implementing :class:`HapticFeedbackReceiver`.
            cfg: Haptic feedback configuration naming the sensors.
        """
        self._device = device
        self._endpoint_sensors = {
            ENDPOINT_LEFT: env.scene[cfg.left_sensor_name],
            ENDPOINT_RIGHT: env.scene[cfg.right_sensor_name],
        }

    def update(self) -> None:
        """Read each hand's net contact force and forward it to the device.

        The force is reduced to a single scalar per hand: the sum of the
        per-body contact-force magnitudes for the first environment.  This
        favors a stable "am I pressing on something" signal over directional
        detail, which is all a single rumble motor can render.

        Both hands are reduced on-device and moved to the host in a single
        transfer to avoid a per-hand CUDA sync each frame. A hand whose sensor
        data is unavailable fails safe to ``0.0`` so the device does not keep
        rendering a stale force.
        """
        # net_forces_w: [num_envs, num_bodies, 3]; reduce env 0 to a scalar [N].
        # Use the explicit ``.torch`` view to avoid the ProxyArray deprecation path.
        reduced = {
            endpoint: torch.linalg.vector_norm(net.torch[0], dim=-1).sum()
            for endpoint, sensor in self._endpoint_sensors.items()
            if (net := sensor.data.net_forces_w) is not None
        }
        forces = dict(zip(reduced, torch.stack(list(reduced.values())).tolist())) if reduced else {}
        for endpoint in self._endpoint_sensors:
            self._device.send_haptic(endpoint, forces.get(endpoint, 0.0))

    def stop(self) -> None:
        """Zero every endpoint so any active pulse stops.

        Call this on every frame the robot is not being stepped (teleop paused or
        session not started). Because the device re-emits the cached force each
        frame, forgetting to zero it would leave the controller vibrating with a
        stale grip force until the next reset.
        """
        for endpoint in self._endpoint_sensors:
            self._device.send_haptic(endpoint, 0.0)


def create_haptic_feedback_driver(
    env: ManagerBasedRLEnv, device: object, env_cfg: object
) -> HapticFeedbackDriver | None:
    """Build a :class:`HapticFeedbackDriver` when the env and device support haptics.

    Follows the same capability-discovery idiom the teleop scripts use for
    ``isaac_teleop``: presence of a ``haptic_feedback`` attribute on the env
    config plus a device that satisfies :class:`HapticFeedbackReceiver`.

    Args:
        env: The (unwrapped) environment.
        device: The active teleop device (any type).
        env_cfg: The environment configuration; checked for a ``haptic_feedback``
            :class:`HapticFeedbackCfg` attribute.

    Returns:
        A ready driver, or ``None`` if haptics are not configured or the device
        cannot render them.
    """
    cfg = getattr(env_cfg, "haptic_feedback", None)
    if cfg is None:
        return None
    if not isinstance(device, HapticFeedbackReceiver):
        return None
    return HapticFeedbackDriver(env, device, cfg)
