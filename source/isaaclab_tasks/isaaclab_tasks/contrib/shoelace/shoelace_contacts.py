# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Signed finger-tail contact observations from Newton collision candidates."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager

from isaaclab.sensors import SensorBase

from . import shoelace_constants as physics

if TYPE_CHECKING:
    from newton import Model


@wp.kernel
def _aggregate_finger_tail_signed_distance(
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=wp.int32),
    contact_count: wp.array(dtype=wp.int32),
    contact_shape0: wp.array(dtype=wp.int32),
    contact_shape1: wp.array(dtype=wp.int32),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_margin0: wp.array(dtype=wp.float32),
    contact_margin1: wp.array(dtype=wp.float32),
    shape_slots: wp.array2d(dtype=wp.int32),
    signed_distance: wp.array2d(dtype=wp.float32),
):
    """Aggregate minimum signed surface separation for the four finger-tail pairs [m]."""
    contact_id = wp.tid()
    if contact_id >= contact_count[0]:
        return

    shape0 = contact_shape0[contact_id]
    shape1 = contact_shape1[contact_id]
    if shape0 < 0 or shape1 < 0:
        return

    finger_shape = shape0
    finger_slot = shape_slots[shape0, 0]
    tail_arm = shape_slots[shape1, 2]
    if finger_slot < 0 or tail_arm != finger_slot // 2:
        finger_shape = shape1
        finger_slot = shape_slots[shape1, 0]
        tail_arm = shape_slots[shape0, 2]
    if finger_slot < 0 or tail_arm != finger_slot // 2:
        return

    body0 = shape_body[shape0]
    body1 = shape_body[shape1]
    transform0 = wp.transform_identity()
    transform1 = wp.transform_identity()
    if body0 >= 0:
        transform0 = body_q[body0]
    if body1 >= 0:
        transform1 = body_q[body1]
    point0_w = wp.transform_point(transform0, contact_point0[contact_id])
    point1_w = wp.transform_point(transform1, contact_point1[contact_id])
    distance = wp.dot(contact_normal[contact_id], point1_w - point0_w)
    distance -= contact_margin0[contact_id] + contact_margin1[contact_id]
    wp.atomic_min(signed_distance, shape_slots[finger_shape, 1], finger_slot, distance)


class FingerTailContacts:
    """Own shape associations and the four signed distances [m] per environment.

    Args:
        model: Finalized Newton model containing the robot and cable collision shapes.
        body_chains: Left and right cable body indices grouped by environment.

    Attributes:
        signed_distance: Finger-tail surface distances [m], shape [N, 4], or ``None`` before initialization.
    """

    def __init__(self, model: Model, body_chains: list[tuple[list[int], list[int]]]) -> None:
        # Columns are finger slot, environment index, and matching tail arm; -1 means unused.
        self._shape_slots = np.full((len(model.shape_label), 3), -1, dtype=np.int32)
        self._num_envs = len(body_chains)
        self.signed_distance: torch.Tensor | None = None
        body_world = model.body_world.numpy()
        fingers: dict[tuple[int, int, int], list[int]] = {}
        for body, label in enumerate(model.body_label):
            for arm, robot in enumerate(("RobotLeft", "RobotRight")):
                for finger, name in enumerate(("panda_leftfinger", "panda_rightfinger")):
                    if f"/{robot}/" in label and label.rsplit("/", 1)[-1] == name:
                        fingers.setdefault((int(body_world[body]), arm, finger), []).append(body)

        for env_id, (left, right) in enumerate(body_chains):
            # Each arm reaches across to the opposite cable's free end.
            for arm, tail_bodies in enumerate((right[-3:], left[:3])):
                for body in tail_bodies:
                    self._shape_slots[model.body_shapes[body], 2] = arm
                for finger in range(2):
                    bodies = fingers.get((env_id, arm, finger), [])
                    if len(bodies) != 1:
                        raise RuntimeError(f"Expected one finger body for {(env_id, arm, finger)}, got {bodies}")
                    shapes = model.body_shapes[bodies[0]]
                    if not shapes:
                        raise RuntimeError(
                            f"Newton finger body {model.body_label[bodies[0]]!r} has no collision shapes"
                        )
                    self._shape_slots[shapes, :2] = (2 * arm + finger, env_id)

    def initialize(self) -> None:
        """Allocate device buffers and register the post-step observer after model construction."""
        device = NewtonManager.get_model().device
        self._device_slots = wp.array(self._shape_slots, dtype=wp.int32, device=device)
        self._distance = wp.full((self._num_envs, 4), physics.CONTACT_DISTANCE_CAP, device=device)
        self.signed_distance = wp.to_torch(self._distance)
        NewtonManager.unregister_post_step_callback(self.update)
        NewtonManager.register_post_step_callback(self.update)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        """Invalidate cached distances [m] for selected environments, or all when ``None``."""
        if self.signed_distance is not None:
            if env_ids is not None and not isinstance(env_ids, (torch.Tensor, slice)):
                env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.signed_distance.device)
            self.signed_distance[slice(None) if env_ids is None else env_ids] = physics.CONTACT_DISTANCE_CAP

    def update(self) -> None:
        """Aggregate signed surface distances [m] after each physics step, including CUDA capture."""
        contacts = NewtonManager.get_contacts()
        model = NewtonManager.get_model()
        if contacts is None:
            raise RuntimeError("Shoelace contact observation requires an initialized Newton collision buffer")
        self._distance.fill_(physics.CONTACT_DISTANCE_CAP)
        wp.launch(
            _aggregate_finger_tail_signed_distance,
            dim=contacts.rigid_contact_max,
            inputs=[
                NewtonManager.get_state_0().body_q,
                model.shape_body,
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                self._device_slots,
            ],
            outputs=[self._distance],
            device=model.device,
        )


class FingerTailContactSensor(SensorBase):
    """Observe the four finger-tail surface distances [m] through the scene sensor lifecycle."""

    @property
    def data(self) -> torch.Tensor:
        """Signed distances [m], shape [num_envs, 4], refreshed inside the physics CUDA graph."""
        return self._contacts.signed_distance

    def reset(
        self, env_ids: Sequence[int] | torch.Tensor | slice | None = None, env_mask: wp.array | None = None
    ) -> None:
        """Clear cached distances for the reset environments."""
        super().reset(env_ids, env_mask)
        if env_mask is not None:
            self.data[wp.to_torch(env_mask)] = physics.CONTACT_DISTANCE_CAP
        else:
            self._contacts.reset(env_ids)

    def _initialize_impl(self) -> None:
        super()._initialize_impl()
        model = NewtonManager.get_model()
        chains = _find_chains(model.body_label, model.body_world.numpy(), self._num_envs, "edge_body")
        if any(not left or not right for left, right in chains):
            raise RuntimeError("Finger-tail contact sensor requires both shoelace segment chains in every world")
        self._contacts = FingerTailContacts(model, chains)
        self._contacts.initialize()

    def _update_buffers_impl(self, env_mask: wp.array) -> None:
        # The post-step callback also runs inside captured solver steps.
        pass

    def _invalidate_initialize_callback(self, event: object) -> None:
        if hasattr(self, "_contacts"):
            NewtonManager.unregister_post_step_callback(self._contacts.update)
        super()._invalidate_initialize_callback(event)

    def _clear_callbacks(self) -> None:
        if hasattr(self, "_contacts"):
            NewtonManager.unregister_post_step_callback(self._contacts.update)
        super()._clear_callbacks()


def _find_chains(labels: list[str], worlds: list[int], num_envs: int, suffix: str) -> list[tuple[list[int], list[int]]]:
    # Match both sides in one pass and retain authored segment order within every world.
    chains = [([], []) for _ in range(num_envs)]
    for index, (label, world) in enumerate(zip(labels, worlds, strict=True)):
        if not 0 <= world < num_envs:
            continue
        for side, name in enumerate(("Left", "Right")):
            prefix = f"/Shoelace{name}/geometry/mesh_{suffix}_"
            if prefix in label:
                chains[world][side].append((int(label.rsplit(prefix, 1)[1]), index))
    return [tuple([index for _, index in sorted(side)] for side in world) for world in chains]
