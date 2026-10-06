# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton asset views that cover only some worlds, or whose shapes are irregularly spaced between worlds."""

from types import SimpleNamespace

import newton
import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_newton.assets import ArticulationData, RigidObjectData
from isaaclab_newton.assets.view_layout import require_strided_joint_and_body_rows
from isaaclab_newton.envs.mdp.events import randomize_rigid_body_collider_offsets, randomize_rigid_body_material
from isaaclab_newton.physics import NewtonManager
from newton.actuators import DrivePD
from newton.selection import ArticulationView

from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.physics import PhysicsManager

WORLD_COUNT = 4
ARM_WORLDS = (0, 2)


def _build_model(object_shape_counts=(2, 2, 2, 2), extra_shape_counts=(1, 3, 2, 4)) -> newton.Model:
    """Arm in worlds 0 and 2, a two-link arm in worlds 1 and 3, and two free objects in every world.

    ``Object_0`` has ``object_shape_counts[world]`` shapes; ``Object_1`` follows it with
    ``extra_shape_counts[world]`` shapes, so ``Object_0`` shapes are irregularly spaced between worlds.
    """
    scene = newton.ModelBuilder()
    for world in range(WORLD_COUNT):
        builder = newton.ModelBuilder()
        robot = f"/World/envs/env_{world}/Arm" if world in ARM_WORLDS else f"/World/envs/env_{world}/TwoLinkArm"
        parent, joints = -1, []
        for link_index in range(1 if world in ARM_WORLDS else 2):
            link = builder.add_link(label=f"{robot}/link_{link_index}")
            builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)
            joints.append(builder.add_joint_revolute(parent=parent, child=link))
            parent = link
        builder.add_articulation(joints, label=robot)
        for slot, shape_count in enumerate((object_shape_counts[world], extra_shape_counts[world])):
            label = f"/World/envs/env_{world}/Object_{slot}"
            body = builder.add_link(label=label)
            for _ in range(shape_count):
                builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
            builder.add_articulation([builder.add_joint_free(child=body)], label=label)
        scene.add_world(builder)
    return scene.finalize(device="cpu")


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> newton.Model:
    """A CPU model installed as the active Newton backend, with distinct gravity per world."""
    model = _build_model()
    gravity = np.zeros((WORLD_COUNT + 1, 3), dtype=np.float32)
    gravity[:, 2] = -np.arange(1, WORLD_COUNT + 2)
    model.gravity.assign(gravity)
    backend = SimpleNamespace(model=model, state_0=model.state(), state_1=model.state(), control=model.control())
    monkeypatch.setattr(NewtonManager, "backend", backend)
    monkeypatch.setattr(PhysicsManager, "_device", "cpu")
    monkeypatch.setattr(NewtonManager, "_world_reset_mask", wp.zeros(WORLD_COUNT + 1, dtype=wp.bool, device="cpu"))
    monkeypatch.setattr(
        NewtonManager, "_fk_reset_mask", wp.zeros(model.articulation_count, dtype=wp.bool, device="cpu")
    )
    return model


def test_view_of_some_worlds_is_sparse(model: newton.Model):
    """Newton gives the arm view only the arm's worlds, with strided joint and body rows."""
    view = ArticulationView(model, "/World/envs/env_*/Arm", verbose=False)
    assert view.is_sparse
    np.testing.assert_array_equal(view.world_ids.numpy(), ARM_WORLDS)
    require_strided_joint_and_body_rows(view, "/World/envs/env_.*/Arm")


@pytest.mark.parametrize("selector", ["mask", "warp_ids", "torch_ids"])
def test_invalidate_fk_marks_the_model_worlds_of_a_sparse_view(model: newton.Model, selector: str):
    """Resetting the second view world marks model world 2 and that world's arm, not model world 1."""
    view = ArticulationView(model, "/World/envs/env_*/Arm", verbose=False)
    if selector == "mask":
        kwargs = {"env_mask": wp.array([False, True], dtype=wp.bool, device="cpu")}
    elif selector == "warp_ids":
        kwargs = {"env_ids": wp.array([1], dtype=wp.int32, device="cpu")}
    else:
        kwargs = {"env_ids": torch.tensor([1], dtype=torch.long)}
    NewtonManager.invalidate_fk(**kwargs, articulation_ids=view.articulation_ids, world_ids=view.world_ids)

    np.testing.assert_array_equal(NewtonManager._world_reset_mask.numpy(), [False, False, True, False, False])
    expected_fk = np.zeros(model.articulation_count, dtype=bool)
    expected_fk[view.articulation_ids.numpy()[1, 0]] = True
    np.testing.assert_array_equal(NewtonManager._fk_reset_mask.numpy(), expected_fk)


def test_sparse_articulation_data_reads_and_resets_its_own_worlds(model: newton.Model):
    """Gravity follows the view's model worlds through updates, and pose resets mark those worlds."""
    view = ArticulationView(model, "/World/envs/env_*/Arm", verbose=False, exclude_joint_types=[newton.JointType.FREE])
    data = ArticulationData(view, "cpu")
    np.testing.assert_array_equal(data.GRAVITY_VEC_W.torch[:, 2].numpy(), [-1.0, -3.0])

    gravity = model.gravity.numpy()
    gravity[2, 2] = -9.0
    model.gravity.assign(gravity)
    data.update(0.01)
    np.testing.assert_array_equal(data.GRAVITY_VEC_W.torch[:, 2].numpy(), [-1.0, -9.0])

    data._reset_pose(env_ids=wp.array([1], dtype=wp.int32, device="cpu"))
    np.testing.assert_array_equal(NewtonManager._world_reset_mask.numpy(), [False, False, True, False, False])


def test_sparse_rigid_object_data_reads_and_resets_its_own_worlds(model: newton.Model):
    """A rigid object in some worlds reads those worlds' gravity, and velocity resets mark those worlds."""
    selected = [
        index for index, label in enumerate(model.articulation_label) if label.endswith(("2/Object_0", "3/Object_0"))
    ]
    view = ArticulationView(model, selected, verbose=False)
    np.testing.assert_array_equal(view.world_ids.numpy(), [2, 3])
    data = RigidObjectData(view, "cpu")
    np.testing.assert_array_equal(data.GRAVITY_VEC_W.torch[:, 2].numpy(), [-3.0, -4.0])

    data._reset_velocity(env_mask=wp.array([True, False], dtype=wp.bool, device="cpu"))
    np.testing.assert_array_equal(NewtonManager._world_reset_mask.numpy(), [False, False, True, False, False])


def test_irregular_body_rows_are_rejected():
    """Bodies irregularly spaced between worlds would bind gathered copies, so assets reject them."""
    scene = newton.ModelBuilder()
    for world, extra_bodies in enumerate((0, 2, 1)):
        builder = newton.ModelBuilder()
        link = builder.add_link(label=f"/World/envs/env_{world}/Arm")
        builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)
        builder.add_articulation(
            [builder.add_joint_revolute(parent=-1, child=link)], label=f"/World/envs/env_{world}/Arm"
        )
        for index in range(extra_bodies):
            body = builder.add_link(label=f"/World/envs/env_{world}/Extra_{index}")
            builder.add_articulation(
                [builder.add_joint_free(child=body)], label=f"/World/envs/env_{world}/Extra_{index}"
            )
        scene.add_world(builder)
    view = ArticulationView(scene.finalize(device="cpu"), "/World/envs/env_*/Arm", verbose=False)

    with pytest.raises(ValueError, match="not regularly spaced"):
        require_strided_joint_and_body_rows(view, "/World/envs/env_.*/Arm")


@pytest.mark.parametrize("dofs_per_world", [(1, 1), (1, 2)])
def test_native_actuators_need_equal_dofs_per_world(monkeypatch: pytest.MonkeyPatch, dofs_per_world: tuple[int, int]):
    """The native actuator adapter strides its buffers by one DOF count per world, so unequal counts raise."""
    scene = newton.ModelBuilder()
    for world, dof_count in enumerate(dofs_per_world):
        builder = newton.ModelBuilder()
        parent, joints = -1, []
        for link_index in range(dof_count):
            link = builder.add_link(label=f"/World/envs/env_{world}/Arm/link_{link_index}")
            joints.append(builder.add_joint_revolute(parent=parent, child=link))
            builder.add_actuator(DrivePD, index=builder.joint_qd_start[joints[-1]], kp=1.0, kd=0.1)
            parent = link
        builder.add_articulation(joints, label=f"/World/envs/env_{world}/Arm")
        scene.add_world(builder)
    model = scene.finalize(device="cpu")
    monkeypatch.setattr(NewtonManager, "backend", SimpleNamespace(model=model, control=model.control()))
    monkeypatch.setattr(NewtonManager, "_num_envs", len(dofs_per_world))
    monkeypatch.setattr(NewtonManager, "_adapter", None)
    monkeypatch.setattr(NewtonManager, "_use_newton_actuators_active", False, raising=False)
    monkeypatch.setattr(PhysicsManager, "_device", "cpu")

    if dofs_per_world[0] == dofs_per_world[1]:
        NewtonManager.activate_newton_actuator_path()
        assert NewtonManager._adapter is not None
    else:
        with pytest.raises(ValueError, match="equal environment DOF counts"):
            NewtonManager.activate_newton_actuator_path()


def _object_shapes(model: newton.Model, world: int) -> list[int]:
    """Model shape indices of ``Object_0`` in a world."""
    body = model.body_label.index(f"/World/envs/env_{world}/Object_0")
    return np.flatnonzero(model.shape_body.numpy() == body).tolist()


def _event_env(model: newton.Model, view: ArticulationView) -> SimpleNamespace:
    """The parts of an environment that the Newton shape events read."""
    return SimpleNamespace(
        num_envs=view.count,
        device="cpu",
        scene={"object": SimpleNamespace(_root_view=view)},
        sim=SimpleNamespace(physics_manager=NewtonManager),
    )


def test_material_writes_reach_shapes_that_are_irregularly_spaced(model: newton.Model):
    """Selected worlds get new materials through the gathered shape rows; other worlds keep theirs."""
    view = ArticulationView(model, "/World/envs/env_*/Object_0", verbose=False)
    assert view.frequency_layouts[newton.Model.AttributeFrequency.SHAPE].uses_explicit_model_indices
    params = {
        "asset_cfg": SceneEntityCfg("object"),
        "static_friction_range": (0.9, 0.9),
        "dynamic_friction_range": (0.9, 0.9),
        "restitution_range": (0.4, 0.4),
        "num_buckets": 1,
    }
    env = _event_env(model, view)
    term = randomize_rigid_body_material(EventTermCfg(func=randomize_rigid_body_material, params=params), env)
    mu_before = model.shape_material_mu.numpy().copy()
    restitution_before = model.shape_material_restitution.numpy().copy()

    term(env, torch.tensor([1, 3]), **params)

    expected_mu, expected_restitution = mu_before.copy(), restitution_before.copy()
    selected_shapes = _object_shapes(model, 1) + _object_shapes(model, 3)
    expected_mu[selected_shapes] = 0.9
    expected_restitution[selected_shapes] = 0.4
    np.testing.assert_allclose(model.shape_material_mu.numpy(), expected_mu)
    np.testing.assert_allclose(model.shape_material_restitution.numpy(), expected_restitution)


def test_collider_offsets_reach_shapes_that_are_irregularly_spaced(model: newton.Model):
    """Selected worlds get new margins and gaps through the gathered shape rows; other worlds keep theirs."""
    view = ArticulationView(model, "/World/envs/env_*/Object_0", verbose=False)
    params = {
        "asset_cfg": SceneEntityCfg("object"),
        "rest_offset_distribution_params": (0.002, 0.002),
        "contact_offset_distribution_params": (0.01, 0.01),
    }
    env = _event_env(model, view)
    term = randomize_rigid_body_collider_offsets(
        EventTermCfg(func=randomize_rigid_body_collider_offsets, params=params), env
    )
    margin_before = model.shape_margin.numpy().copy()
    gap_before = model.shape_gap.numpy().copy()

    term(env, torch.tensor([2]), **params)

    expected_margin, expected_gap = margin_before.copy(), gap_before.copy()
    expected_margin[_object_shapes(model, 2)] = 0.002
    expected_gap[_object_shapes(model, 2)] = 0.008
    np.testing.assert_allclose(model.shape_margin.numpy(), expected_margin, rtol=1e-6)
    np.testing.assert_allclose(model.shape_gap.numpy(), expected_gap, rtol=1e-6)
