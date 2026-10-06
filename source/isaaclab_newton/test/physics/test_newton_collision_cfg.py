# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Newton collision pipeline configuration and its use by the Newton manager."""

from __future__ import annotations

from types import SimpleNamespace

import newton
import numpy as np
import pytest
import warp as wp
from isaaclab_newton.physics import NewtonCollisionPipelineCfg, NewtonManager

from isaaclab.physics import PhysicsManager


@pytest.mark.parametrize(
    ("rigid_contact_max", "rigid_contacts_per_world", "world_count", "expected"),
    [(None, None, 3, None), (None, 8, 3, 24), (100, 8, 3, 100), (100, None, None, 100)],
)
def test_rigid_contact_capacity_resolves_from_the_world_count(
    rigid_contact_max, rigid_contacts_per_world, world_count, expected
):
    """An absolute capacity takes precedence; otherwise the per-world capacity scales with the world count."""
    cfg = NewtonCollisionPipelineCfg(
        rigid_contact_max=rigid_contact_max, rigid_contacts_per_world=rigid_contacts_per_world
    )

    assert cfg.resolve_rigid_contact_max(world_count) == expected
    assert cfg.to_pipeline_args(world_count)["rigid_contact_max"] == expected


@pytest.mark.parametrize(("rigid_contacts_per_world", "world_count"), [(0, 3), (8, None)])
def test_rigid_contact_capacity_rejects_unresolvable_per_world_limit(rigid_contacts_per_world, world_count):
    """A per-world capacity must be positive and needs the model's world count."""
    cfg = NewtonCollisionPipelineCfg(rigid_contacts_per_world=rigid_contacts_per_world)

    with pytest.raises(ValueError, match="rigid_contacts_per_world"):
        cfg.to_pipeline_args(world_count)


@pytest.fixture
def two_world_model(monkeypatch: pytest.MonkeyPatch):
    """A CPU model with one resting box per world, installed as the Newton manager's backend model."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    for world in range(2):
        builder.begin_world()
        body = builder.add_body(xform=wp.transform((3.0 * world, 0.0, 0.1), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        builder.end_world()
    model = builder.finalize(device="cpu")
    monkeypatch.setattr(PhysicsManager, "_device", "cpu")
    monkeypatch.setattr(NewtonManager, "backend", SimpleNamespace(model=model))
    monkeypatch.setattr(NewtonManager, "_needs_collision_pipeline", True)
    monkeypatch.setattr(NewtonManager, "_collision_pipeline", None)
    monkeypatch.setattr(NewtonManager, "_contacts", None)
    monkeypatch.setattr(NewtonManager, "_solver", SimpleNamespace())
    monkeypatch.setattr(NewtonManager, "_deterministic_mode", wp.DeterministicMode.NOT_GUARANTEED)
    return model


def test_manager_pipeline_holds_per_world_capacity_and_collision_only_determinism(monkeypatch, two_world_model):
    """The manager sizes contacts for every world and sorts them without a global determinism request."""
    cfg = NewtonCollisionPipelineCfg(rigid_contacts_per_world=7, deterministic=True)
    monkeypatch.setattr(NewtonManager, "_collision_cfg", cfg)

    NewtonManager._initialize_contacts()

    assert NewtonManager._contacts.rigid_contact_max == 14
    assert NewtonManager._collision_pipeline.deterministic


@pytest.mark.parametrize(
    "cfg",
    [
        NewtonCollisionPipelineCfg(rigid_contact_max=4, deterministic=True),
        NewtonCollisionPipelineCfg(rigid_contact_max=4, contact_matching="latest"),
    ],
    ids=["deterministic", "contact_matching"],
)
def test_grown_contact_buffer_matches_the_sorted_pipeline(monkeypatch, two_world_model, cfg):
    """A sorted pipeline is rebuilt, not given a larger contact buffer, when the solver needs more contacts."""
    monkeypatch.setattr(NewtonManager, "_collision_cfg", cfg)
    monkeypatch.setattr(NewtonManager, "_solver", SimpleNamespace(get_max_contact_count=lambda: 64))

    NewtonManager._initialize_contacts()
    NewtonManager._collision_pipeline.collide(two_world_model.state(), NewtonManager._contacts)

    assert NewtonManager._contacts.rigid_contact_max == 64


@pytest.mark.parametrize("contact_matching", ["latest", "sticky"])
@pytest.mark.parametrize("reset_worlds", [[], [0], [0, 1]])
def test_reset_clears_contact_matching_history_of_reset_worlds(
    monkeypatch, two_world_model, contact_matching, reset_worlds
):
    """Contacts of a reset world start unmatched; contacts of other worlds keep their match."""
    model = two_world_model
    cfg = NewtonCollisionPipelineCfg(contact_matching=contact_matching)
    monkeypatch.setattr(NewtonManager, "_collision_cfg", cfg)
    monkeypatch.setattr(
        NewtonManager, "_world_reset_mask", wp.zeros(model.world_count + 1, dtype=wp.bool, device=model.device)
    )
    monkeypatch.setattr(NewtonManager, "_fk_reset_mask", None)
    monkeypatch.setattr(NewtonManager, "_reset_solver_internals_delegate", lambda world_mask: None)
    monkeypatch.setattr(NewtonManager, "_eval_fk", lambda world_mask, fk_mask: None)
    NewtonManager._initialize_contacts()
    state, contacts = model.state(), NewtonManager._contacts

    def collide_and_read_matches() -> tuple[np.ndarray, np.ndarray]:
        NewtonManager._collision_pipeline.collide(state, contacts)
        count = int(contacts.rigid_contact_count.numpy()[0])
        shapes = np.stack(
            [contacts.rigid_contact_shape0.numpy()[:count], contacts.rigid_contact_shape1.numpy()[:count]]
        )
        worlds = model.shape_world.numpy()[shapes].max(axis=0)
        return worlds, contacts.rigid_contact_match_index.numpy()[:count]

    collide_and_read_matches()
    worlds, matches = collide_and_read_matches()
    assert set(worlds) == {0, 1} and (matches >= 0).all()

    reset = np.zeros(model.world_count + 1, dtype=bool)
    reset[reset_worlds] = True
    NewtonManager._world_reset_mask.assign(reset)
    monkeypatch.setattr(NewtonManager, "kinematics_dirty", True)
    NewtonManager.forward()
    worlds, matches = collide_and_read_matches()

    assert not NewtonManager._world_reset_mask.numpy().any()
    assert np.array_equal(matches < 0, reset[worlds])
