# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent force-balance contracts for NeRD's diagnostic inverse dynamics."""

import newton
import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_newton.physics.nerd_forces import _ForceReconstruction

from isaaclab.test.utils import DeviceScope, test_devices


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_fixed_effort_support_flight_and_reset(device):
    """Contact supports weight minus actual effort; flight cannot invent support or motor effort."""
    builder = newton.ModelBuilder()
    builder.request_state_attributes("body_parent_f")
    builder.request_contact_attributes("force")
    builder.add_ground_plane()
    for world in range(3):
        builder.begin_world()
        base = builder.add_link(mass=2.0, inertia=wp.mat33(0.02, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.02))
        child = builder.add_link(mass=1.0, inertia=wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01))
        slider = builder.add_joint_prismatic(
            -1, base, axis=(0.0, 0.0, 1.0), parent_xform=wp.transform((world * 2.0, 0.0, 0.199), wp.quat_identity())
        )
        mount = builder.add_joint_fixed(base, child, parent_xform=wp.transform((0.0, 0.0, 0.4), wp.quat_identity()))
        builder.add_articulation([slider, mount])
        builder.add_shape_sphere(base, radius=0.2, cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.5))
        builder.end_world()
    model = builder.finalize(device=device)
    state, control = model.state(), model.control()
    wp.to_torch(state.joint_q)[1] = 1.0
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    control.joint_f.assign(np.array([5.0, -3.0, 40.0], dtype=np.float32))
    reconstruction = _ForceReconstruction(model, 0.005)
    collision = newton.CollisionPipeline(model, rigid_contact_max=reconstruction.solver.get_max_contact_count())
    contacts = collision.contacts()
    collision.collide(state, contacts)
    q, qd, effort = [wp.to_torch(x).clone() for x in (state.joint_q, state.joint_qd, control.joint_f)]

    def reconstruct():
        reconstruction.prepare(state, control, contacts)
        reconstruction.update(state)
        reconstruction.solver._populate_contact_observables(contacts, contacts.force, preserve_geometry=True)

    pool = torch.cuda.MemPool()
    with torch.cuda.stream(wp.stream_to_torch(wp.get_stream(device))), torch.cuda.use_mem_pool(pool):
        reconstruct()
        with wp.ScopedCapture(device=device) as capture:
            reconstruct()
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    for original, current in zip((q, qd, effort), (state.joint_q, state.joint_qd, control.joint_f)):
        torch.testing.assert_close(wp.to_torch(current), original, rtol=0, atol=0)
    acceleration = wp.to_torch(reconstruction.solver.mjw_data.qacc)[:, 0]
    expected_acceleration = torch.tensor([0.0, -10.81, 40.0 / 3.0 - 9.81], device=device)
    torch.testing.assert_close(acceleration, expected_acceleration, atol=1e-3, rtol=1e-4)
    # The fixed 1 kg child requires F = m(a + g), independently of contact distribution.
    parent_forces = wp.to_torch(state.body_parent_f).reshape(3, 2, 6)
    torch.testing.assert_close(parent_forces[:, 1, 2], expected_acceleration + 9.81, atol=1e-3, rtol=1e-4)
    torch.testing.assert_close(parent_forces[:, 0, 2], effort, atol=1e-3, rtol=1e-4)
    force = wp.to_torch(contacts.force)
    assert force.abs().max() > 20.0
    assert torch.isfinite(force).all()
    # Masked reset clears incoming wrenches only in the selected world.
    before = parent_forces.clone()
    mask = wp.array([False, True, False, False], dtype=wp.bool, device=device)
    reconstruction.reset(state, mask)
    assert not parent_forces[1].any()
    torch.testing.assert_close(parent_forces[[0, 2]], before[[0, 2]], rtol=0, atol=0)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_reconstruction_contact_capacity_boundaries(device):
    """Independent sliders have known support loads across both GPU solve sizes and no-contact worlds."""
    counts = (0, 1, 6, 12, 20)
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    for world in range(len(counts)):
        builder.begin_world()
        root = builder.add_link()
        joints = [builder.add_joint_fixed(-1, root)]
        for slot in range(20):
            body = builder.add_link(mass=2.0, inertia=wp.mat33(0.02, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.02))
            joints.append(
                builder.add_joint_prismatic(
                    root,
                    body,
                    axis=(0.0, 0.0, 1.0),
                    parent_xform=wp.transform((slot * 0.5, world * 2.0, 0.199), wp.quat_identity()),
                )
            )
            builder.add_shape_sphere(body, radius=0.2, cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.5))
        builder.add_articulation(joints)
        builder.end_world()
    model = builder.finalize(device=device)
    state = model.state()
    positions = wp.to_torch(state.joint_q).reshape(len(counts), 20)
    for world, count in enumerate(counts):
        positions[world, count:] = 1.0
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    reconstruction = _ForceReconstruction(model, 0.005)
    collision = newton.CollisionPipeline(model, rigid_contact_max=reconstruction.solver.get_max_contact_count())
    contacts = collision.contacts()
    collision.collide(state, contacts)
    with torch.cuda.stream(wp.stream_to_torch(wp.get_stream(device))):
        reconstruction.prepare(state, model.control(), contacts)
        reconstruction.update(state)
    wp.synchronize_device(device)
    data = reconstruction.solver.mjw_data
    np.testing.assert_array_equal(data.nefc.numpy(), np.array(counts) * 3)
    for world, count in enumerate(counts):
        expected = torch.full((20,), -9.81, device=device)
        expected[:count] = 0.0
        torch.testing.assert_close(wp.to_torch(data.qacc)[world], expected, atol=2e-3, rtol=1e-4)
        normal = wp.to_torch(data.efc.force)[world, : count * 3 : 3]
        torch.testing.assert_close(normal, torch.full_like(normal, 19.62), atol=2e-3, rtol=1e-4)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_reconstruction_respects_friction_and_rejects_adhesion(device):
    """A sliding sphere saturates the friction cone while root reaction remains zero."""
    builder = newton.ModelBuilder()
    builder.request_state_attributes("body_parent_f")
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
    builder.begin_world()
    body = builder.add_link(
        xform=wp.transform((0.0, 0.0, 0.199), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(0.016, 0.0, 0.0, 0.0, 0.016, 0.0, 0.0, 0.0, 0.016),
    )
    builder.add_articulation([builder.add_joint_free(body)])
    builder.add_shape_sphere(body, radius=0.2, cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.5))
    builder.end_world()
    model = builder.finalize(device=device)
    state = model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    reconstruction = _ForceReconstruction(model, 0.005)
    collision = newton.CollisionPipeline(model, rigid_contact_max=reconstruction.solver.get_max_contact_count())
    contacts = collision.contacts()
    collision.collide(state, contacts)
    with torch.cuda.stream(wp.stream_to_torch(wp.get_stream(device))):
        reconstruction.prepare(state, model.control(), contacts)
        wp.to_torch(state.joint_qd)[0] = 0.1  # Ask for 20 m/s^2 horizontally with zero applied effort.
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        reconstruction.update(state)
    wp.synchronize_device(device)
    data = reconstruction.solver.mjw_data
    assert data.nefc.numpy()[0] == 3
    force = wp.to_torch(data.efc.force)[0, :3]
    assert force[0] > 0
    torch.testing.assert_close(force[1:].norm(), force[0] * 0.5, atol=2e-3, rtol=1e-4)
    # No world-root motor exists: independent Newton-Euler recursion must recover zero reaction.
    torch.testing.assert_close(
        wp.to_torch(state.body_parent_f), torch.zeros_like(wp.to_torch(state.body_parent_f)), atol=2e-3, rtol=0
    )
    model.shape_material_ka.fill_(1.0)
    with pytest.raises(ValueError, match="adhesive"):
        _ForceReconstruction(model, 0.005)
