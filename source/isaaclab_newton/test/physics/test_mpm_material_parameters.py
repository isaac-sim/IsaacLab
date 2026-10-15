# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime MPM material selection, density units, and validation contracts."""

from types import SimpleNamespace

import newton
import pytest
import torch
import warp as wp
from isaaclab_newton.physics import MPMSolverCfg, NewtonManager, NewtonMPMManager
from newton.solvers import SolverImplicitMPM


@pytest.fixture
def material_model(monkeypatch):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    for i in range(4):
        builder.add_particle(pos=(float(i), 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=0.0 if i == 1 else 1.0, radius=0.05)
    model = builder.finalize(device="cpu")
    solver = NewtonMPMManager._create_solver(model, MPMSolverCfg(grid_type="fixed", grid_padding=1, max_iterations=1))
    monkeypatch.setattr(NewtonManager, "_solver", solver)
    monkeypatch.setattr(NewtonManager, "backend", SimpleNamespace(model=model))
    monkeypatch.setattr(NewtonManager, "_model_changes", set())
    return model


def test_selected_material_updates_preserve_kinematic_particles(material_model):
    """Density writes preserve zero masses and use cubic particle volume, without changing other particles."""
    model = material_model
    friction_before = wp.to_torch(model.mpm.friction).clone()
    NewtonMPMManager.set_particle_material_parameters(
        {
            "density": wp.array([1200.0, 1800.0], dtype=wp.float32, device=material_model.device),
            "friction": 0.65,
            "damping": 0.02,
        },
        slice(1, None, 2),
    )
    torch.testing.assert_close(wp.to_torch(model.particle_mass), torch.tensor([1.0, 0.0, 1.0, 1.8]))
    torch.testing.assert_close(wp.to_torch(model.particle_inv_mass), torch.tensor([1.0, 0.0, 1.0, 1 / 1.8]))
    torch.testing.assert_close(wp.to_torch(model.mpm.friction)[::2], friction_before[::2])
    values = NewtonMPMManager.get_particle_material_parameters(
        wp.array([3, 1], dtype=wp.int64, device=material_model.device), parameters=["density", "friction", "damping"]
    )
    torch.testing.assert_close(values["density"].torch, torch.tensor([1800.0, 0.0]))
    torch.testing.assert_close(values["damping"].torch, torch.tensor([0.02, 0.02]))
    torch.testing.assert_close(values["friction"].torch, torch.tensor([0.65, 0.65]))
    NewtonMPMManager.set_particle_material_parameters({"density": values["density"].warp}, [3, 1])
    torch.testing.assert_close(wp.to_torch(model.particle_mass), torch.tensor([1.0, 0.0, 1.0, 1.8]))
    values["friction"].warp.zero_()
    torch.testing.assert_close(wp.to_torch(model.mpm.friction)[1::2], torch.tensor([0.65, 0.65]))
    NewtonMPMManager.set_particle_material_parameters({"friction": 0.4}, slice(None))
    torch.testing.assert_close(wp.to_torch(model.mpm.friction), torch.full((4,), 0.4))


def test_invalid_material_batch_does_not_partially_write(material_model):
    """A bad second parameter must not leave the first parameter partially updated."""
    before = NewtonMPMManager.get_particle_material_parameters()
    with pytest.raises(ValueError, match="poisson_ratio"):
        NewtonMPMManager.set_particle_material_parameters({"friction": 0.1, "poisson_ratio": 0.5})
    with pytest.raises(ValueError, match="shape"):
        NewtonMPMManager.set_particle_material_parameters(
            {"friction": wp.ones(3, dtype=wp.float32, device=material_model.device)}, [0, 2]
        )
    with pytest.raises(ValueError, match="Unknown MPM"):
        NewtonMPMManager.set_particle_material_parameters({"friction": 0.1, "particle_stress": 0.0})
    with pytest.raises(IndexError, match="outside"):
        NewtonMPMManager.set_particle_material_parameters(
            {"friction": 0.1}, wp.array([0, 4], dtype=wp.int32, device=material_model.device)
        )
    with pytest.raises(IndexError, match="outside"):
        NewtonMPMManager.get_particle_material_parameters([-1], parameters=["friction"])
    with pytest.raises(ValueError, match="density"):
        NewtonMPMManager.set_particle_material_parameters({"density": float("nan")})
    with pytest.raises(ValueError, match="repeated"):
        NewtonMPMManager.set_particle_material_parameters({"friction": 0.2}, [0, 0])
    after = NewtonMPMManager.get_particle_material_parameters()
    for name in before:
        torch.testing.assert_close(after[name].torch, before[name].torch)


def test_material_write_snapshots_overlapping_input(material_model):
    """A reversed scatter of a live model array must retain all original values."""
    NewtonMPMManager.set_particle_material_parameters(
        {"friction": wp.array([0.1, 0.2, 0.3, 0.4], dtype=float, device="cpu")}
    )
    NewtonMPMManager.set_particle_material_parameters({"friction": material_model.mpm.friction}, [3, 2, 1, 0])
    torch.testing.assert_close(wp.to_torch(material_model.mpm.friction), torch.tensor([0.4, 0.3, 0.2, 0.1]))


@pytest.mark.parametrize("coupled", [False, True])
def test_material_notifications_preserve_only_compatible_graphs(material_model, monkeypatch, coupled):
    """The real manager step retains value-only graphs and invalidates changed buffers or branches."""
    from isaaclab.physics import PhysicsManager

    from isaaclab_contrib.coupling.coupler import NewtonCouplerManager

    solver = NewtonManager._solver
    manager = NewtonCouplerManager if coupled else NewtonMPMManager
    changes = []
    callbacks = []

    def notify(flags):
        changes.append(flags)
        callbacks[-1]()

    if coupled:
        root = SimpleNamespace(entry_names=lambda: ["sand"], solver=lambda name: solver, notify_model_changed=notify)
        monkeypatch.setattr(NewtonManager, "_solver", root)
    else:
        monkeypatch.setattr(solver, "notify_model_changed", notify)
    monkeypatch.setattr(PhysicsManager, "_sim", SimpleNamespace(is_playing=lambda: True))
    monkeypatch.setattr(PhysicsManager, "_device", "cpu")
    monkeypatch.setattr(PhysicsManager, "_cfg", SimpleNamespace(use_cuda_graph=False))
    monkeypatch.setattr(PhysicsManager, "_sim_time", 0.0)
    monkeypatch.setattr(NewtonManager, "forward", classmethod(lambda cls: None))
    monkeypatch.setattr(NewtonManager, "_simulate_full", classmethod(lambda cls: None))
    monkeypatch.setattr(NewtonManager, "_is_all_graphable", classmethod(lambda cls: True))
    monkeypatch.setattr(NewtonManager, "_check_solver_status", classmethod(lambda cls: None))
    monkeypatch.setattr(NewtonManager, "_graph_capture_pending", False)
    monkeypatch.setattr(NewtonManager, "_graph", None)
    material = solver._mpm_model
    original_buffer = material.material_parameters.friction

    def replace_buffer():
        material.material_parameters.friction = wp.clone(original_buffer)

    def change_branch():
        material.has_viscosity = not material.has_viscosity

    for notification, compatible in [
        (lambda: original_buffer.fill_(0.6), True),
        (replace_buffer, False),
        (change_branch, False),
    ]:
        graph = object()
        NewtonManager._graph = graph
        callbacks.append(notification)
        NewtonMPMManager.set_particle_material_parameters({"friction": 0.6})
        manager.step()
        assert (NewtonManager._graph is graph) == compatible
        assert not NewtonManager._model_changes
    assert changes == [newton.ModelFlags.MODEL_PROPERTIES] * 3
