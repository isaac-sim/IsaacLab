# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Haptic feedback (device output) for IsaacTeleop teleoperation.

This module adds an *output* path to the otherwise input-only teleop stack: a
sim-side signal (contact force on a gripper, per-finger grip force on an object,
...) is rendered on the operator's device (controller rumble, haptic glove, ...).

The design keeps three concerns pluggable and device-agnostic so a single seam
serves multiple devices:

* **Signal source** -- how a per-hand vector is read from the environment. Built
  by :meth:`HapticFeedbackCfg.make_signal_fn`; ready-made factories are
  :func:`contact_force_magnitude` (one scalar per hand) and
  :func:`per_finger_object_grip` (per-finger force against a target object).
* **Output port** -- the narrow :class:`HapticFeedbackReceiver` protocol. Its
  payload is a *vector* (length ``num_taxels``): length 1 for a rumble motor, one
  channel per finger for a glove. Kept off the input-only ``DeviceBase``.
* **Device backend** -- how the vector is rendered. Selected by the concrete
  :class:`HapticFeedbackCfg` subclass via :meth:`~HapticFeedbackCfg.build_sink`,
  which wires an ``isaacteleop`` retargeter + ``IHapticDevice`` behind a
  ``HapticSink``. Isaac Lab supplies the signal; ``isaacteleop`` owns the
  mapping curve and the device write.

:class:`HapticFeedbackDriver` ties them together and is used identically by every
teleop script.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import torch

from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Well-known endpoint identifiers for the two hands. These match ``isaacteleop``'s
# ``Endpoint`` strings for hand-mounted haptic devices (controllers and gloves).
ENDPOINT_LEFT: str = "left"
ENDPOINT_RIGHT: str = "right"

# A signal function reads the environment and returns the per-hand output vector
# (length ``num_taxels``) for each endpoint.
SignalFn = Callable[["ManagerBasedRLEnv"], "dict[str, np.ndarray]"]


@runtime_checkable
class HapticFeedbackReceiver(Protocol):
    """Protocol for a teleop device that can render haptic feedback.

    Deliberately a single-method protocol rather than an addition to
    ``DeviceBase``: input-only devices (keyboard, space-mouse, gamepad) cannot
    honor a feedback call, so forcing the method onto every device would break
    Liskov substitutability. Scripts guard with ``isinstance(device,
    HapticFeedbackReceiver)`` and only drive feedback when the active device
    actually supports it.
    """

    def send_haptic(self, endpoint: str, values: Sequence[float]) -> None:
        """Render one frame of haptic output on a device endpoint.

        Args:
            endpoint: Which hand to actuate -- :data:`ENDPOINT_LEFT` or
                :data:`ENDPOINT_RIGHT`.
            values: The per-hand output vector (length ``num_taxels``): a single
                scalar for a rumble motor, one value per finger for a glove. An
                all-zero vector stops any active feedback. The signal-to-device
                mapping (gain, deadband, saturation) is applied downstream.
        """
        ...


# ----------------------------------------------------------------------------
# Signal-source factories (environment -> per-hand vector)
# ----------------------------------------------------------------------------


def contact_force_magnitude(left_sensor_name: str, right_sensor_name: str) -> SignalFn:
    """Signal source: total contact-force magnitude per hand (a length-1 vector).

    Sums the per-body contact-force magnitudes of each hand's
    :class:`~isaaclab.sensors.ContactSensor` for the first environment. Suited to
    a single-actuator device such as a controller rumble motor.

    Args:
        left_sensor_name: Scene entity name of the left-hand contact sensor.
        right_sensor_name: Scene entity name of the right-hand contact sensor.

    Returns:
        A signal function mapping the environment to ``{endpoint: [force]}`` [N].
    """
    names = {ENDPOINT_LEFT: left_sensor_name, ENDPOINT_RIGHT: right_sensor_name}

    def signal_fn(env: ManagerBasedRLEnv) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for endpoint, name in names.items():
            forces = env.scene[name].data.net_forces_w
            if forces is None:
                out[endpoint] = np.zeros(1, dtype=np.float32)
            else:
                magnitude = torch.linalg.vector_norm(forces[0], dim=-1).sum().item()
                out[endpoint] = np.array([magnitude], dtype=np.float32)
        return out

    return signal_fn


def per_finger_object_grip(
    left_sensor_name: str,
    right_sensor_name: str,
    object_name: str,
    finger_order: Sequence[str],
) -> SignalFn:
    """Signal source: per-finger grip force against a specific object.

    Reads each hand's contact sensor ``force_matrix_w`` -- the contact force
    *filtered against the object* (see :attr:`ContactSensorCfg.filter_prim_paths_expr`)
    -- and orders the per-finger magnitudes into the device's channel order. Only
    force against ``object_name`` contributes, so incidental table or self
    contact does not trigger feedback. Suited to a haptic glove.

    The sensor's matched bodies are mapped to output channels by substring: for
    channel ``c``, the first sensor body whose (lower-cased) name contains
    ``finger_order[c]`` supplies that channel. The map is resolved once per
    sensor and cached.

    Args:
        left_sensor_name: Scene entity name of the left-hand contact sensor
            (must be configured with the object as a contact filter).
        right_sensor_name: Scene entity name of the right-hand contact sensor.
        object_name: Scene entity name of the grasped object (used only for docs;
            the filtering itself is configured on the sensors).
        finger_order: Per-channel finger substrings in device channel order
            (e.g. ``["thumb", "index", "middle", "ring", "pinky"]``).

    Returns:
        A signal function mapping the environment to ``{endpoint: [f0, ..., fN]}`` [N].
    """
    names = {ENDPOINT_LEFT: left_sensor_name, ENDPOINT_RIGHT: right_sensor_name}
    fingers = [f.lower() for f in finger_order]
    num_channels = len(fingers)
    channel_maps: dict[str, list[int | None]] = {}

    def _resolve_channel_map(sensor: Any) -> list[int | None]:
        body_names = [n.lower() for n in (sensor.body_names or [])]
        return [next((i for i, bn in enumerate(body_names) if finger in bn), None) for finger in fingers]

    def signal_fn(env: ManagerBasedRLEnv) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for endpoint, name in names.items():
            sensor = env.scene[name]
            vec = np.zeros(num_channels, dtype=np.float32)
            # force_matrix_w: [num_envs, num_bodies, num_filters, 3] (None before first contact).
            force_matrix = sensor.data.force_matrix_w
            if force_matrix is not None:
                if name not in channel_maps:
                    channel_maps[name] = _resolve_channel_map(sensor)
                # Per body: sum force magnitude over all filter shapes for env 0.
                per_body = torch.linalg.vector_norm(force_matrix[0], dim=-1).sum(dim=-1)
                for channel, body_index in enumerate(channel_maps[name]):
                    if body_index is not None:
                        vec[channel] = float(per_body[body_index])
            out[endpoint] = vec
        return out

    return signal_fn


# ----------------------------------------------------------------------------
# Configuration: device-agnostic base + backend-specific subclasses
# ----------------------------------------------------------------------------


@configclass
class HapticFeedbackCfg:
    """Base configuration for teleop haptic feedback.

    Device-agnostic. Attach a concrete subclass
    (:class:`ControllerHapticFeedbackCfg` or :class:`GloveHapticFeedbackCfg`) to an
    environment config as a sibling of ``isaac_teleop`` (e.g.
    ``self.haptic_feedback = ...`` in ``__post_init__``). The teleop scripts
    discover it via ``hasattr(env_cfg, "haptic_feedback")``.

    A subclass supplies the two pluggable pieces: :meth:`make_signal_fn` (how the
    per-hand vector is read from the environment) and :meth:`build_sink` (how that
    vector is rendered by an ``isaacteleop`` device).
    """

    left_sensor_name: str = "left_hand_contact"
    """Scene entity name of the left-hand :class:`~isaaclab.sensors.ContactSensor`."""

    right_sensor_name: str = "right_hand_contact"
    """Scene entity name of the right-hand :class:`~isaaclab.sensors.ContactSensor`."""

    num_taxels: int = 1
    """Length of the per-hand output vector: 1 for a single rumble motor, one
    channel per finger for a glove."""

    gain: float = 1.0
    """Signal-to-output gain applied after the deadband (see the subclass docs for units)."""

    deadband: float = 0.0
    """Signal magnitude below which no output is produced (rejects sensor noise)."""

    saturation: float = 1.0
    """Upper clamp on the normalized output amplitude in ``[0, 1]``."""

    def make_signal_fn(self) -> SignalFn:
        """Return the signal source (environment -> per-hand vector) for this config."""
        raise NotImplementedError

    def build_sink(self, force_inputs: dict[str, Any], tracker_provider: Callable[[], Any]) -> Any:
        """Build the ``isaacteleop`` ``HapticSink`` that renders the cached vectors.

        Args:
            force_inputs: Per-endpoint graph output nodes, each carrying a
                ``TactileVector(num_taxels)`` fed from the driver every step.
            tracker_provider: Callable returning a DeviceIO tracker to reuse for
                devices that need one (e.g. a controller). Cross-process devices
                (e.g. a glove) may ignore it.

        Returns:
            A connected ``HapticSink`` node for ``TeleopSessionConfig(sinks=[...])``.
        """
        raise NotImplementedError


@configclass
class ControllerHapticFeedbackCfg(HapticFeedbackCfg):
    """Haptic feedback rendered as XR motion-controller vibration.

    Renders a single scalar per hand (total gripper contact force) as controller
    rumble via ``ControllerHapticDevice`` + ``TactileVectorToControllerPulse``.
    """

    num_taxels: int = 1
    gain: float = 0.05
    """Force-to-amplitude gain [1/N] applied after the deadband. Default maps ~20 N to full scale."""
    deadband: float = 0.5
    """Contact force [N] below which no vibration is produced."""
    frequency_hz: float = 0.0
    """Vibration frequency [Hz]. ``0`` selects the XR runtime default."""
    duration_s: float = 0.0
    """Pulse duration [s]. ``0`` selects the shortest supported pulse; refreshed each frame."""

    def make_signal_fn(self) -> SignalFn:
        return contact_force_magnitude(self.left_sensor_name, self.right_sensor_name)

    def build_sink(self, force_inputs: dict[str, Any], tracker_provider: Callable[[], Any]) -> Any:
        from isaacteleop.haptic_devices.controller import ControllerHapticDevice
        from isaacteleop.retargeters.tactile_retargeters import TactileVectorToControllerPulse
        from isaacteleop.retargeting_engine.deviceio_source_nodes import HapticSink

        sink_inputs = {}
        for endpoint, force_output in force_inputs.items():
            pulse = TactileVectorToControllerPulse(
                f"_haptic_pulse_{endpoint}",
                num_taxels=self.num_taxels,
                gain=self.gain,
                deadband=self.deadband,
                saturation=self.saturation,
                frequency_hz=self.frequency_hz,
                duration_s=self.duration_s,
            ).connect({TactileVectorToControllerPulse.INPUT_TACTILE: force_output})
            sink_inputs[endpoint] = pulse.output(TactileVectorToControllerPulse.OUTPUT_PULSE)

        device = ControllerHapticDevice(tracker_provider())
        return HapticSink("_haptic_sink", device).connect(sink_inputs)


@configclass
class GloveHapticFeedbackCfg(HapticFeedbackCfg):
    """Haptic feedback rendered as per-finger power on a haptic glove.

    Renders per-finger grip force against the grasped object as finger vibration
    via a cross-process ``haptic_glove_device`` + ``TactileVectorToFingerPower``.
    """

    num_taxels: int = 5
    """Finger channels per hand (Thumb..Pinky by default)."""
    gain: float = 0.1
    """Force-to-power gain [1/N] applied after the deadband. Default maps ~10 N to full power."""
    deadband: float = 0.5
    """Contact force [N] below which no vibration is produced."""
    smoothing: float = 0.5
    """EMA new-sample weight in ``[0, 1]`` (1.0 = no smoothing) applied to each finger power."""
    collection_id: str = "manus_glove_haptic"
    """Push-tensor collection id pairing Isaac Teleop with the glove plugin process
    (the Manus plugin's default). Change it to target a different glove vendor."""
    object_name: str = "object"
    """Scene entity name of the grasped object the fingers are gripping."""
    finger_order: list[str] = ["thumb", "index", "middle", "ring", "pinky"]
    """Per-channel finger substrings, in glove channel order, matched against the
    contact sensor's body names to order the per-finger forces."""

    def make_signal_fn(self) -> SignalFn:
        return per_finger_object_grip(
            self.left_sensor_name, self.right_sensor_name, self.object_name, self.finger_order
        )

    def build_sink(self, force_inputs: dict[str, Any], tracker_provider: Callable[[], Any]) -> Any:
        from isaacteleop.haptic_devices.glove import haptic_glove_device
        from isaacteleop.retargeters.tactile_retargeters import TactileVectorToFingerPower
        from isaacteleop.retargeting_engine.deviceio_source_nodes import HapticSink

        sink_inputs = {}
        for endpoint, force_output in force_inputs.items():
            powers = TactileVectorToFingerPower(
                f"_haptic_powers_{endpoint}",
                num_taxels=self.num_taxels,
                num_fingers=self.num_taxels,
                gain=self.gain,
                deadband=self.deadband,
                saturation=self.saturation,
                smoothing=self.smoothing,
            ).connect({TactileVectorToFingerPower.INPUT_TACTILE: force_output})
            sink_inputs[endpoint] = powers.output(TactileVectorToFingerPower.OUTPUT_POWERS)

        # Cross-process device; it has no DeviceIO tracker, so tracker_provider is unused.
        device = haptic_glove_device(self.collection_id, num_fingers=self.num_taxels)
        return HapticSink("_haptic_sink", device).connect(sink_inputs)


# ----------------------------------------------------------------------------
# Driver + discovery
# ----------------------------------------------------------------------------


class HapticFeedbackDriver:
    """Reads the per-hand signal from the scene and pushes it to the device.

    One driver is shared by every teleop script. Construct it with
    :func:`create_haptic_feedback_driver` (which returns ``None`` when the env has
    no haptic config or the device cannot render it), then call :meth:`update`
    once per step *after* ``env.step`` so the sensors hold fresh post-physics data.
    """

    def __init__(self, env: ManagerBasedRLEnv, device: HapticFeedbackReceiver, cfg: HapticFeedbackCfg):
        """Initialize the driver.

        Args:
            env: The (unwrapped) environment whose scene owns the sensors.
            device: The teleop device implementing :class:`HapticFeedbackReceiver`.
            cfg: Haptic feedback configuration providing the signal source.
        """
        self._env = env
        self._device = device
        self._signal_fn = cfg.make_signal_fn()
        self._zero = np.zeros(cfg.num_taxels, dtype=np.float32)

    def update(self) -> None:
        """Read the per-hand signal and forward each endpoint's vector to the device."""
        for endpoint, values in self._signal_fn(self._env).items():
            self._device.send_haptic(endpoint, values)

    def stop(self) -> None:
        """Zero every endpoint so any active feedback stops.

        Call this on every frame the robot is not being stepped (teleop paused or
        session not started). Because the device re-emits the cached vector each
        frame, forgetting to zero it would leave the device rendering a stale grip
        force until the next reset.
        """
        for endpoint in (ENDPOINT_LEFT, ENDPOINT_RIGHT):
            self._device.send_haptic(endpoint, self._zero)


def create_haptic_feedback_driver(
    env: ManagerBasedRLEnv, device: object, env_cfg: object
) -> HapticFeedbackDriver | None:
    """Build a :class:`HapticFeedbackDriver` when the env and device support haptics.

    Follows the same capability-discovery idiom the teleop scripts use for
    ``isaac_teleop``: a ``haptic_feedback`` attribute on the env config plus a
    device that satisfies :class:`HapticFeedbackReceiver`.

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
