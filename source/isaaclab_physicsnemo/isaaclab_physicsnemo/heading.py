# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Heading-frame state and body-frame gravity for free-root learned dynamics.

These follow the input recipe of the PhysicsNeMo Robotics G1 application: the model sees the robot in
its current heading frame, so its predictions do not depend on world position or yaw, and gravity in the
root body frame restores the orientation with respect to gravity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import newton
import torch
from physicsnemo_robotics.dynamics import FeatureEncoder, FeatureSignature, JointStateCodec, StateCodec
from physicsnemo_robotics.newton import NewtonStepContext
from tensordict import TensorDictBase

from isaaclab.utils.math import normalize, quat_apply, quat_apply_inverse, quat_conjugate, quat_mul

_HEADING_FRAME = "free_root_heading_swing_twist_z"


class HeadingJointStateCodec(JointStateCodec):
    """Joint state of a free-root robot expressed in its current heading frame.

    The stored and live state stays Newton world state. Model inputs place each frame in its own heading
    frame: the root's planar position and its twist about the world z axis are removed, while height,
    tilt, and joint coordinates are kept. Transition targets express both endpoints in the heading frame
    of the current state, and predictions are mapped back to world state before they are written.

    The heading is the twist about z of a swing/twist decomposition, which stays well defined for tilted
    bodies, unlike an Euler yaw angle.
    """

    def __init__(self, model: newton.Model) -> None:
        """Build the codec for a Newton model whose articulation root is a free joint.

        Args:
            model: Finalized Newton model with z-up coordinates.

        Raises:
            ValueError: If the articulation root is not a free joint at the start of the coordinates, or the
                model is not z-up.
        """
        super().__init__(model, remove_root_translation=False)
        layout = self.layout
        if not layout.root_is_free or layout.up_axis_index != 2:
            raise ValueError("The heading codec requires a free articulation root in z-up coordinates.")
        if layout.free_root_translation_q_starts != (0,) or layout.quaternion_q_starts[:1] != (3,):
            raise ValueError("The heading codec requires root position at q[0:3] and orientation at q[3:7].")

    def artifact_config(self) -> dict[str, object]:
        """Return the portable constructor contract stored in an artifact."""
        return {"version": 1, "frame": _HEADING_FRAME}

    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        """Encode every frame in its own heading frame."""
        local, _, _ = self._to_heading(state, anchor=state)
        return super().encode_state(local)

    def state_to_delta(self, current: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        """Encode a transition with both endpoints in the heading frame of ``current``."""
        current_local, _, _ = self._to_heading(current, anchor=current)
        next_local, _, _ = self._to_heading(next_state, anchor=current)
        return super().state_to_delta(current_local, next_local)

    def delta_to_state(self, current: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Integrate a heading-frame prediction and return the next world state."""
        current_local, heading, origin = self._to_heading(current, anchor=current)
        next_local = super().delta_to_state(current_local, delta)
        return self._to_world(next_local, heading, origin)

    def _to_heading(
        self, state: torch.Tensor, *, anchor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Express ``state`` in the heading frame of ``anchor`` and return the frame."""
        dof_q = self.layout.dof_q
        heading = heading_quaternion(anchor[..., 3:7])
        origin = torch.cat((anchor[..., :2], torch.zeros_like(anchor[..., 2:3])), dim=-1)
        q, qd = state[..., :dof_q], state[..., dof_q:]
        local = torch.cat(
            (
                quat_apply_inverse(heading, q[..., :3] - origin),
                quat_mul(quat_conjugate(heading), q[..., 3:7]),
                q[..., 7:],
                quat_apply_inverse(heading, qd[..., :3]),
                quat_apply_inverse(heading, qd[..., 3:6]),
                qd[..., 6:],
            ),
            dim=-1,
        )
        return local, heading, origin

    def _to_world(self, local: torch.Tensor, heading: torch.Tensor, origin: torch.Tensor) -> torch.Tensor:
        """Map a heading-frame state back to Newton world state."""
        dof_q = self.layout.dof_q
        q, qd = local[..., :dof_q], local[..., dof_q:]
        world = torch.cat(
            (
                quat_apply(heading, q[..., :3]) + origin,
                quat_mul(heading, q[..., 3:7]),
                q[..., 7:],
                quat_apply(heading, qd[..., :3]),
                quat_apply(heading, qd[..., 3:6]),
                qd[..., 6:],
            ),
            dim=-1,
        )
        return self.canonicalize_state(world)


class RootGravityEncoder:
    """Unit gravity direction in the free-root body frame, computed from the joint state.

    The value is a function of the state, so it is derived from the stored state when training and from
    the live state when deployed, and needs no recorded field.
    """

    def __init__(self, codec: JointStateCodec, world_gravity: Sequence[float] = (0.0, 0.0, -1.0)) -> None:
        """Bind the encoder to a free-root joint codec.

        Args:
            codec: Joint codec whose root orientation is the quaternion at ``q[3:7]``.
            world_gravity: Gravity direction in world coordinates. It is normalized.

        Raises:
            ValueError: If the codec has no free root.
        """
        if not codec.layout.root_is_free or codec.layout.quaternion_q_starts[:1] != (3,):
            raise ValueError("Root gravity requires a free root with orientation at q[3:7].")
        self.codec = codec
        self.world_gravity = tuple(normalize(torch.tensor(world_gravity, dtype=torch.float64)).tolist())

    @property
    def signature(self) -> FeatureSignature:
        """Describe the gravity feature for artifact compatibility checks."""
        return FeatureSignature(
            "root_body_gravity",
            3,
            requires_contacts=False,
            dtype="float32",
            semantics={"world_gravity": list(self.world_gravity), "frame": "free_root_body"},
        )

    def artifact_config(self) -> dict[str, object]:
        """Return the portable constructor contract stored in an artifact."""
        return {"version": 1, "world_gravity": list(self.world_gravity)}

    def encode_window(self, window: TensorDictBase) -> torch.Tensor:
        """Encode gravity for every stored state of a transition window."""
        return self._encode(window.get("history/state"))

    def encode_step(self, context: NewtonStepContext) -> torch.Tensor:
        """Encode gravity for the live state that starts the transition."""
        return self._encode(self.codec.read(context.state_in))

    def _encode(self, state: torch.Tensor) -> torch.Tensor:
        gravity = state.new_tensor(self.world_gravity).expand(*state.shape[:-1], 3)
        return quat_apply_inverse(normalize(state[..., 3:7]), gravity)


def heading_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    """Return the twist about z of an ``(x, y, z, w)`` orientation, with a non-negative ``w``.

    Orientations whose twist is undefined (a half turn about a horizontal axis) map to the identity.
    """
    twist = torch.zeros_like(quaternion)
    twist[..., 2:] = quaternion[..., 2:]
    defined = twist.norm(dim=-1, keepdim=True) >= 1.0e-8
    twist = torch.where(defined, twist, torch.tensor([0.0, 0.0, 0.0, 1.0], device=twist.device, dtype=twist.dtype))
    twist = normalize(twist)
    return torch.where(twist[..., 3:] < 0.0, -twist, twist)


def artifact_codec_factory(descriptor: Mapping[str, Any], model: newton.Model) -> StateCodec:
    """Rebuild a custom state codec of this extension from an artifact descriptor.

    The learned-dynamics manager uses it when
    :attr:`~isaaclab_physicsnemo.physics.PhysicsNeMoSolverCfg.codec_factory` is not set.

    Raises:
        ValueError: If the descriptor names another class or an unsupported version.
    """
    _require_descriptor(descriptor, HeadingJointStateCodec, {"version": 1, "frame": _HEADING_FRAME})
    return HeadingJointStateCodec(model)


def artifact_feature_factory(descriptor: Mapping[str, Any], model: newton.Model, codec: object) -> FeatureEncoder:
    """Rebuild a custom feature encoder of this extension from an artifact descriptor.

    The learned-dynamics manager uses it when
    :attr:`~isaaclab_physicsnemo.physics.PhysicsNeMoSolverCfg.feature_factory` is not set.

    Raises:
        ValueError: If the descriptor names another class or an unsupported version.
        TypeError: If the rebuilt state codec is not a joint codec.
    """
    config = descriptor["config"]
    _require_descriptor(descriptor, RootGravityEncoder, {"version": 1, "world_gravity": config.get("world_gravity")})
    if not isinstance(codec, JointStateCodec):
        raise TypeError("Root gravity artifacts require a joint state codec.")
    return RootGravityEncoder(codec, config["world_gravity"])


def _require_descriptor(descriptor: Mapping[str, Any], expected: type, config: Mapping[str, Any]) -> None:
    name = f"{expected.__module__}.{expected.__qualname__}"
    if descriptor.get("class") != name or dict(descriptor.get("config", {})) != dict(config):
        raise ValueError(f"Unsupported custom artifact descriptor {descriptor.get('class')!r}; expected {name!r}.")
