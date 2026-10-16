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
    after = NewtonMPMManager.get_particle_material_parameters()
    for name in before:
        torch.testing.assert_close(after[name].torch, before[name].torch)


@pytest.mark.parametrize("distribution", ["uniform", "log_uniform"])
def test_material_event_only_updates_selected_environments(material_model, distribution):
    """Event sampling routes environment IDs to the corresponding particles without changing other objects."""
    from isaaclab_newton.assets import MPMObject
    from isaaclab_newton.envs.mdp.events import randomize_mpm_material

    from isaaclab.managers import EventTermCfg, SceneEntityCfg

    # The event needs only the asset's particle layout; the material model is real.
    asset = object.__new__(MPMObject)
    asset._num_instances = 2
    asset._particles_per_object = 2
    asset._particle_offsets = wp.array([0, 2], dtype=wp.int32, device="cpu")
    env = SimpleNamespace(num_envs=2, device="cpu", scene={"sand": asset})
    asset_cfg = SceneEntityCfg("sand")
    cfg = EventTermCfg(func=randomize_mpm_material, mode="reset", params={"asset_cfg": asset_cfg})
    term = randomize_mpm_material(cfg, env)
    before = NewtonMPMManager.get_particle_material_parameters(parameters=["friction"])["friction"].torch
    term(env, torch.tensor([1]), asset_cfg, {"friction": (0.3, 0.8)}, distribution)
    after = NewtonMPMManager.get_particle_material_parameters(parameters=["friction"])["friction"].torch
    torch.testing.assert_close(after[:2], before[:2])
    assert 0.3 <= float(after[2]) <= 0.8
    torch.testing.assert_close(after[2], after[3])
    torch.testing.assert_close(term.material_parameters["friction"][1], after[2])
    with pytest.raises(ValueError, match="Invalid"):
        term(env, slice(None), asset_cfg, {"friction": (0.0, 0.8)}, "log_uniform")
    torch.testing.assert_close(
        NewtonMPMManager.get_particle_material_parameters(parameters=["friction"])["friction"].torch, after
    )
