# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Policy Ghosts slot layout and the tint the Newton GL panel writes to ghost shapes."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp
from isaaclab_visualizers.newton.newton_policy_ghosts import NewtonPolicyGhostPanel

from isaaclab.visualizers.policy_ghost import PolicyGhostCheckpoint, PolicyGhostLayout

pytestmark = [pytest.mark.unit]

SLICE_SIZE = 2
CHECKPOINT = PolicyGhostCheckpoint(path="/runs/a/model_1.pt", run="a", name="model_1.pt", modified=0.0)


def _panel(layout: PolicyGhostLayout) -> NewtonPolicyGhostPanel:
    return NewtonPolicyGhostPanel(SimpleNamespace(layout=layout))


def _model() -> SimpleNamespace:
    """Model with one shared static shape and one visible shape per world of three slots."""
    worlds = np.array([-1] + list(range(3 * SLICE_SIZE)))
    return SimpleNamespace(
        shape_world=wp.array(worlds, dtype=wp.int32, device="cpu"),
        shape_color=wp.array(np.full((len(worlds), 3), 0.5), dtype=wp.vec3, device="cpu"),
        shape_opacity=None,
    )


def test_only_slots_with_a_checkpoint_are_visible_and_limited_per_slot() -> None:
    layout = PolicyGhostLayout(SLICE_SIZE, 3)
    layout.slots[0].checkpoint = CHECKPOINT
    layout.slots[2].checkpoint = CHECKPOINT

    assert layout.visible_env_ids(1) == [0, 4]
    assert layout.visible_env_ids(SLICE_SIZE + 5) == [0, 1, 4, 5]

    layout.slots[2].visible = False

    assert layout.visible_env_ids(SLICE_SIZE) == [0, 1]


def test_tint_applies_to_active_ghost_worlds_only_and_follows_slot_changes() -> None:
    layout = PolicyGhostLayout(SLICE_SIZE, 3)
    layout.slots[0].checkpoint = CHECKPOINT
    layout.slots[1].checkpoint = CHECKPOINT
    layout.slots[1].opacity = 0.4
    layout.slots[1].tint = 1.0
    panel, model = _panel(layout), _model()

    panel.apply_tint(model)

    colors, opacity = model.shape_color.numpy(), model.shape_opacity.numpy()
    ghost_shapes = [1 + SLICE_SIZE + i for i in range(SLICE_SIZE)]
    assert np.allclose(colors[ghost_shapes], layout.slots[1].color)
    assert np.allclose(opacity[ghost_shapes], 0.4)
    untouched = [0, *range(1, 1 + SLICE_SIZE), *range(1 + 2 * SLICE_SIZE, 1 + 3 * SLICE_SIZE)]
    assert np.allclose(colors[untouched], 0.5)
    assert np.allclose(opacity[untouched], 1.0)

    layout.slots[1].checkpoint = None
    panel.apply_tint(model)
    assert np.allclose(model.shape_color.numpy(), 0.5)
    assert np.allclose(model.shape_opacity.numpy(), 1.0)


def test_slot_changes_are_reported_once() -> None:
    layout = PolicyGhostLayout(SLICE_SIZE, 3)
    panel = _panel(layout)
    assert panel.consume_slots_changed()
    assert not panel.consume_slots_changed()

    layout.slots[1].checkpoint = CHECKPOINT

    assert panel.consume_slots_changed()


def test_tint_strength_blends_between_native_and_ghost_color() -> None:
    layout = PolicyGhostLayout(SLICE_SIZE, 3)
    layout.slots[0].checkpoint = CHECKPOINT
    layout.slots[1].checkpoint = CHECKPOINT
    layout.slots[1].tint = 0.5
    panel, model = _panel(layout), _model()

    panel.apply_tint(model)

    ghost_shape = 1 + SLICE_SIZE
    expected = 0.5 * 0.5 + 0.5 * np.asarray(layout.slots[1].color)
    assert np.allclose(model.shape_color.numpy()[ghost_shape], expected)

    layout.slots[1].tint = 0.0
    panel.apply_tint(model)
    assert np.allclose(model.shape_color.numpy()[ghost_shape], 0.5)


def _viewer_with_batch() -> SimpleNamespace:
    """Viewer with one batch holding a textured, metallic instance for each of the three slots' first worlds."""
    worlds = np.array([0, SLICE_SIZE, 2 * SLICE_SIZE])
    native = np.tile([0.5, 0.7, 0.0, 1.0], (3, 1)).astype(np.float32)
    batch = SimpleNamespace(
        worlds=wp.array(worlds, dtype=wp.int32, device="cpu"),
        materials=wp.array(native, dtype=wp.vec4, device="cpu"),
    )
    return SimpleNamespace(_shape_instances={"batch": batch}, model_changed=False)


def test_full_tint_removes_texture_and_metallic_shading_only_from_that_ghost() -> None:
    layout = PolicyGhostLayout(SLICE_SIZE, 3)
    for slot in layout.slots:
        slot.checkpoint = CHECKPOINT
    layout.slots[1].tint = 1.0
    layout.slots[2].tint = 0.5
    panel, viewer = _panel(layout), _viewer_with_batch()

    panel.apply_materials(viewer)

    materials = viewer._shape_instances["batch"].materials.numpy()
    assert np.allclose(materials[1], [0.5, 0.0, 0.0, 0.0])
    assert np.allclose(materials[[0, 2]], [0.5, 0.7, 0.0, 1.0])
    assert viewer.model_changed

    layout.slots[1].tint = 0.5
    panel.apply_materials(viewer)
    assert np.allclose(viewer._shape_instances["batch"].materials.numpy(), [0.5, 0.7, 0.0, 1.0])
