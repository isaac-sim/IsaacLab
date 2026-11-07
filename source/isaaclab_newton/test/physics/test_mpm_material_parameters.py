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
from isaaclab_newton.physics import (
    MPMSolver,
    MPMSolverCfg,
    NewtonBackend,
    NewtonBackendCfg,
    NewtonCfg,
    get_particle_material_parameters,
    set_particle_material_parameters,
)
from newton.solvers import SolverImplicitMPM


@pytest.fixture
def material_backend():
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    for i in range(4):
        builder.add_particle(pos=(float(i), 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=0.0 if i == 1 else 1.0, radius=0.05)
    model = builder.finalize(device="cpu")
    solver = MPMSolver(
        MPMSolverCfg(grid_type="fixed", grid_padding=1, max_iterations=1),
        model,
        wp.DeterministicMode.NOT_GUARANTEED,
    ).native
    return SimpleNamespace(model=model, solver=solver, model_changes=set())


def test_selected_material_updates_preserve_kinematic_particles(material_backend):
    """Density writes preserve zero masses and use cubic particle volume, without changing other particles."""
    model = material_backend.model
    friction_before = wp.to_torch(model.mpm.friction).clone()
    set_particle_material_parameters(
        material_backend,
        {
            "density": wp.array([1200.0, 1800.0], dtype=wp.float32, device=material_backend.model.device),
            "friction": 0.65,
            "damping": 0.02,
        },
        slice(1, None, 2),
    )
    torch.testing.assert_close(wp.to_torch(model.particle_mass), torch.tensor([1.0, 0.0, 1.0, 1.8]))
    torch.testing.assert_close(wp.to_torch(model.particle_inv_mass), torch.tensor([1.0, 0.0, 1.0, 1 / 1.8]))
    torch.testing.assert_close(wp.to_torch(model.mpm.friction)[::2], friction_before[::2])
    values = get_particle_material_parameters(
        material_backend,
        wp.array([3, 1], dtype=wp.int64, device=material_backend.model.device),
        parameters=["density", "friction", "damping"],
    )
    torch.testing.assert_close(values["density"].torch, torch.tensor([1800.0, 0.0]))
    torch.testing.assert_close(values["damping"].torch, torch.tensor([0.02, 0.02]))
    torch.testing.assert_close(values["friction"].torch, torch.tensor([0.65, 0.65]))
    set_particle_material_parameters(material_backend, {"density": values["density"].warp}, [3, 1])
    torch.testing.assert_close(wp.to_torch(model.particle_mass), torch.tensor([1.0, 0.0, 1.0, 1.8]))
    values["friction"].warp.zero_()
    torch.testing.assert_close(wp.to_torch(model.mpm.friction)[1::2], torch.tensor([0.65, 0.65]))
    set_particle_material_parameters(material_backend, {"friction": 0.4}, slice(None))
    torch.testing.assert_close(wp.to_torch(model.mpm.friction), torch.full((4,), 0.4))


def test_invalid_material_batch_does_not_partially_write(material_backend):
    """A bad second parameter must not leave the first parameter partially updated."""
    before = get_particle_material_parameters(
        material_backend,
    )
    with pytest.raises(ValueError, match="poisson_ratio"):
        set_particle_material_parameters(material_backend, {"friction": 0.1, "poisson_ratio": 0.5})
    with pytest.raises(ValueError, match="shape"):
        set_particle_material_parameters(
            material_backend, {"friction": wp.ones(3, dtype=wp.float32, device=material_backend.model.device)}, [0, 2]
        )
    with pytest.raises(ValueError, match="Unknown MPM"):
        set_particle_material_parameters(material_backend, {"friction": 0.1, "particle_stress": 0.0})
    with pytest.raises(IndexError, match="outside"):
        set_particle_material_parameters(
            material_backend, {"friction": 0.1}, wp.array([0, 4], dtype=wp.int32, device=material_backend.model.device)
        )
    with pytest.raises(IndexError, match="outside"):
        get_particle_material_parameters(material_backend, [-1], parameters=["friction"])
    with pytest.raises(ValueError, match="density"):
        set_particle_material_parameters(material_backend, {"density": float("nan")})
    with pytest.raises(ValueError, match="repeated"):
        set_particle_material_parameters(material_backend, {"friction": 0.2}, [0, 0])
    after = get_particle_material_parameters(
        material_backend,
    )
    for name in before:
        torch.testing.assert_close(after[name].torch, before[name].torch)


def test_material_write_snapshots_overlapping_input(material_backend):
    """A reversed scatter of a live model array must retain all original values."""
    set_particle_material_parameters(
        material_backend, {"friction": wp.array([0.1, 0.2, 0.3, 0.4], dtype=float, device="cpu")}
    )
    set_particle_material_parameters(material_backend, {"friction": material_backend.model.mpm.friction}, [3, 2, 1, 0])
    torch.testing.assert_close(wp.to_torch(material_backend.model.mpm.friction), torch.tensor([0.4, 0.3, 0.2, 0.1]))


@pytest.mark.parametrize("coupled", [False, True])
def test_material_updates_reuse_and_invalidate_cuda_graphs(coupled):
    """Update real MPM child data and replay captured steps after material changes."""
    if not wp.is_cuda_available():
        pytest.skip("CUDA graph replay requires a CUDA device.")
    from isaaclab_newton.physics import XPBDSolverCfg

    from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -1.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.begin_world()
    for i in range(4):
        builder.add_particle(pos=(0.04 * i, 0.0, 0.1), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.02)
    builder.end_world()
    mpm_cfg = MPMSolverCfg(
        grid_type="fixed",
        grid_padding=1,
        max_active_cell_count=128,
        max_iterations=2,
        voxel_size=0.1,
        solver="jacobi",
        warmstart_mode="none",
    )
    solver_cfg = (
        CouplerProxyCfg(
            entries=[
                CouplerEntryCfg(name="other", solver_cfg=XPBDSolverCfg(), particles=[0]),
                CouplerEntryCfg(name="sand", solver_cfg=mpm_cfg, particles=[1, 2, 3], in_place=True),
            ]
        )
        if coupled
        else mpm_cfg
    )
    backend = NewtonBackend(
        NewtonBackendCfg(physics_cfg=NewtonCfg(solver_cfg=solver_cfg, use_cuda_graph=True), device="cuda:0"),
        builder,
        dt=0.001,
    )
    backend.init_solver()
    mpm = backend.solver.solver("sand") if coupled else backend.solver
    try:
        backend.step()
        assert backend.graphs
        previous_graphs = backend.graphs.copy()
        initial_positions = backend.state_0.particle_q.numpy().copy()
        set_particle_material_parameters(backend, {"friction": 0.9, "density": 1500.0}, [1, 2, 3])
        backend.step()
        assert backend.graphs == previous_graphs
        assert not backend.model_changes
        torch.testing.assert_close(wp.to_torch(mpm.model.mpm.friction)[1:], torch.full((3,), 0.9, device="cuda:0"))
        torch.testing.assert_close(
            wp.to_torch(mpm._mpm_model.particle_density)[1:], torch.full((3,), 1500.0, device="cuda:0")
        )
        assert (backend.state_0.particle_q.numpy() != initial_positions).any()
        if coupled:
            assert mpm.model.particle_mass.numpy()[0] == 0.0

        # Enabling viscosity changes a Python-side solver branch and requires a fresh capture.
        set_particle_material_parameters(backend, {"viscosity": 0.1}, [1, 2, 3])
        backend.notify_model_changes()
        assert not backend.graphs
        assert mpm._mpm_model.has_viscosity
        backend.step()
        assert backend.graphs
        assert backend.graphs != previous_graphs

        # Replacing an array must also release the old graph, even with unchanged values.
        mpm.model.mpm.friction = wp.clone(mpm.model.mpm.friction)
        backend.model_changes.add(newton.ModelFlags.MODEL_PROPERTIES)
        backend.notify_model_changes()
        assert not backend.graphs
        backend.step()
        assert backend.graphs
        assert torch.isfinite(wp.to_torch(backend.state_0.particle_q)).all()
    finally:
        backend.close()
