# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import subprocess
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

newton = pytest.importorskip("newton")
from isaaclab_newton.assets import MPMObject
from isaaclab_newton.physics import NewtonManager
from isaaclab_newton.physics.newton_manager import NewtonSceneDataBackend
from isaaclab_newton.sim.spawners.mpm import MPMGridCfg, MPMParticleMaterialCfg, MPMPointsCfg
from newton.solvers import SolverImplicitMPM

from pxr import UsdGeom, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.cloner import UsdReplicateContext, make_clone_plan

pytestmark = pytest.mark.unit


@pytest.fixture
def stage():
    stage = sim_utils.create_new_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    UsdPhysics.Scene.Define(stage, "/physicsScene")
    return stage


def test_mpm_points_author_and_import_through_usd(stage, monkeypatch):
    material = MPMParticleMaterialCfg(
        young_modulus=2500.0,
        damping=0.125,
        hardening_rate=2.0,
        softening_rate=3.0,
    )
    cfg = MPMPointsCfg(
        positions=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        velocities=((0.1, 0.0, 0.0), (0.0, 0.2, 0.0)),
        mass=(0.3, 0.4),
        radius=(0.05, 0.06),
        material=material,
    )
    cfg.func(
        "/World/Media",
        cfg,
        translation=(2.0, 3.0, 4.0),
        orientation=(0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)),
    )

    points = UsdGeom.Points(stage.GetPrimAtPath("/World/Media/geometry/points"))
    points_prim = points.GetPrim()
    assert {"NewtonPointsDeformableSimAPI", "PhysicsDeformableBodyAPI"} <= set(
        points_prim.GetPrimTypeInfo().GetAppliedAPISchemas()
    )
    assert points_prim.GetRelationship("physics:simulationOwner").GetTargets() == [
        stage.GetPrimAtPath("/physicsScene").GetPath()
    ]
    np.testing.assert_allclose(points_prim.GetAttribute("physics:masses").Get(), cfg.mass)
    np.testing.assert_allclose(points.GetWidthsAttr().Get(), (0.1, 0.12))

    binding = UsdShade.MaterialBindingAPI(points_prim).GetDirectBinding("physics")
    material_prim = stage.GetPrimAtPath(binding.GetMaterialPath())
    assert material_prim.GetAttribute("newton:mpm:elasticDamping").Get() == pytest.approx(312.5)
    assert material_prim.GetAttribute("newton:mpm:hardeningRate").Get() == pytest.approx(2.0)
    assert material_prim.GetAttribute("newton:mpm:softeningRate").Get() == pytest.approx(3.0)

    builder = newton.ModelBuilder(up_axis="Z")
    SolverImplicitMPM.register_custom_attributes(builder)
    result = builder.add_usd(stage, root_path="/World/Media")

    assert result["path_particle_map"] == {"/World/Media/geometry/points": (0, 2)}
    np.testing.assert_allclose(builder.particle_q, ((2.0, 4.0, 4.0), (1.0, 3.0, 4.0)), atol=2.0e-7)
    np.testing.assert_allclose(builder.particle_qd, ((0.0, 0.1, 0.0), (-0.2, 0.0, 0.0)), atol=3.0e-8)
    np.testing.assert_allclose(builder.particle_mass, cfg.mass)
    np.testing.assert_allclose(builder.particle_radius, cfg.radius)
    visual_points = UsdGeom.Points(stage.GetPrimAtPath("/World/Media/Particles"))
    assert not visual_points.GetResetXformStack()
    np.testing.assert_allclose(visual_points.GetPointsAttr().Get(), cfg.positions)
    np.testing.assert_allclose(visual_points.GetWidthsAttr().Get(), (0.1, 0.12))
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(visual_points.GetPrim())
    np.testing.assert_allclose(
        [transform.Transform(point) for point in visual_points.GetPointsAttr().Get()], builder.particle_q, atol=2.0e-7
    )
    shared_cfg = cfg.copy()
    shared_cfg.func("/World/Shared", shared_cfg, translation=(-2.0, 0.0, 0.0))
    usd = SimpleNamespace(
        instances=(
            (0, "/World/Media", "/Scene/copy_{}/Media", np.array([12, 7])),
            (1, "/World/Other", "/Scene/copy_{}/Media", np.array([99])),
            (2, "/World/Shared", "/World/Shared", np.array([-1])),
        ),
        global_paths=("/World/Shared",),
    )
    monkeypatch.setattr(
        sim_utils.SimulationContext, "instance", lambda: SimpleNamespace(clone_contexts={UsdReplicateContext: usd})
    )
    state = SimpleNamespace(particle_q=wp.zeros(12, dtype=wp.vec3f, device="cpu"))
    monkeypatch.setattr(NewtonSceneDataBackend, "state", property(lambda self: state))
    backend = NewtonSceneDataBackend()
    backend.initialize_geometry(make_clone_plan((), ((),), 0), usd.instances)
    monkeypatch.setattr(NewtonManager, "_scene_data_backend", backend)
    asset = SimpleNamespace(
        cfg=SimpleNamespace(prim_path="/Scene/copy_[^/]+/Media", spawn=cfg),
        _recorded_particle_offsets=[4, 10],
        _particles_per_object=2,
    )
    MPMObject._bind_particle_visualization(asset)
    asset.cfg.prim_path = "/World/Shared"
    asset.cfg.spawn = shared_cfg
    asset._recorded_particle_offsets = [0]
    MPMObject._bind_particle_visualization(asset)
    assert not stage.GetPrimAtPath("/Scene/copy_12/Media/Particles")
    [(publication, ranges)] = backend.get_geometry_batches()
    assert publication.points is state.particle_q
    assert ranges == {
        "/Scene/copy_12/Media/Particles": (4, 2),
        "/Scene/copy_7/Media/Particles": (10, 2),
        "/World/Shared/Particles": (0, 2),
    }
    assert visual_points.GetPrim().GetAttribute("isaaclab:pointsUpdateFrequency").Get() == cfg.visual_update_frequency


@pytest.mark.parametrize("visible", [False, True])
def test_mpm_grid_authors_explicit_particles(stage, visible):
    cfg = MPMGridCfg(
        lower=(0.0, 0.0, 0.0),
        upper=(0.2, 0.2, 0.2),
        voxel_size=0.1,
        particles_per_cell=1.0,
        particle_placement="cell_center",
        jitter=0.0,
        visible=visible,
    )
    cfg.func("/World/Media", cfg)

    points = UsdGeom.Points(stage.GetPrimAtPath("/World/Media/geometry/points"))
    assert len(points.GetPointsAttr().Get()) == 8
    assert len(points.GetPrim().GetAttribute("physics:masses").Get()) == 8
    assert bool(stage.GetPrimAtPath("/World/Media/Particles")) is visible


def test_mpm_config_imports_do_not_load_pxr():
    code = textwrap.dedent(
        """
        import sys

        from isaaclab_newton.assets import MPMObjectCfg
        from isaaclab_newton.sim.spawners.mpm import MPMGridCfg, MPMParticleMaterialCfg, MPMPointsCfg

        MPMObjectCfg(
            prim_path="{ENV_REGEX_NS}/Sand",
            spawn=MPMGridCfg(
                lower=(0.0, 0.0, 0.0),
                upper=(0.1, 0.1, 0.1),
                voxel_size=0.1,
                material=MPMParticleMaterialCfg(),
            ),
        )

        loaded_pxr_modules = [module for module in sys.modules if module == "pxr" or module.startswith("pxr.")]
        if loaded_pxr_modules:
            raise SystemExit("pxr loaded before SimulationApp: " + ", ".join(loaded_pxr_modules[:20]))
        """
    )

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "module",
    [
        "examples.mpm.newton_mpm_granular",
        "examples.mpm.newton_mpm_twoway_coupling",
        "examples.demos.snowball_smash",
        "examples.demos.teapot_fill",
    ],
)
def test_mpm_program_configs_do_not_load_pxr_before_simulation_launch(module):
    """Every MPM program must delay USD imports until after ``AppLauncher`` starts."""
    code = textwrap.dedent(
        f"""
        import importlib
        import sys

        sys.argv = ["program.py", "--max_steps", "0", "--visualizer", "none", "--device", "cuda:0"]
        program = importlib.import_module({module!r})
        program.create_sim_cfg()

        loaded_pxr_modules = [name for name in sys.modules if name == "pxr" or name.startswith("pxr.")]
        if loaded_pxr_modules:
            raise SystemExit("pxr loaded before simulation launch: " + ", ".join(loaded_pxr_modules[:20]))
        """
    )

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr
