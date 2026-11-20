# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX USD export: a PhysX import of the exported file in a new process reproduces the initialized engine, for a
scene that changes the exported values and for registered tasks."""

from isaaclab.test.utils import launch_test_simulation

launch_test_simulation()

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from pxr import UsdPhysics  # noqa: E402

# The engine reader lives next to this test, which also runs it in a separate Isaac Sim process.
sys.path.insert(0, str(Path(__file__).parent))

_DRIVELESS_GAINS = {"damping": r".*/Robot/joints/\w+:angular", "max_force": r".*/Robot/joints/\w+:angular"}

TASKS = {
    # Franka USD physics variant, mimic finger, rigid object
    "Isaac-Lift-Franka": {},
    # a second articulation (cabinet with a prismatic drawer); edits under instanced assets
    "Isaac-Open-Drawer-Franka": {},
    # floating base, LSTM network actuator, rough terrain
    "Isaac-Velocity-Rough-AnymalD": {},
    # explicit DC motor actuator
    "Isaac-Velocity-Flat-UnitreeGo2": {},
    # Isaac Lab gives joints without a USD drive gains, which PhysX does not apply and the export leaves out
    "Isaac-Ant": _DRIVELESS_GAINS,
    # the export of a DirectRLEnv
    "Isaac-Cartpole-Direct": {},
    # the export of a DirectMARLEnv
    "Isaac-Pendulum-MARL-Direct": {},
}
"""Task -> values the import holds differently: reading name -> regular expression of entities."""


def _read_live(scene, path: str) -> tuple:
    """The exported stage, its articulation roots and other rigid bodies, and the live engine's readings of them."""
    from usd_export_read_import import read, readings

    from isaaclab.test.utils.usd_export import read_live_engine

    view = scene.sim.backend_manager.get_physics_sim_view()
    return read_live_engine(scene, path, lambda stage, *prims: readings(stage, read(view, *prims)))


def _read_import(stage, articulations: list[str], bodies: list[str]) -> dict:
    """Readings of a PhysX import of ``stage`` in a new Isaac Sim process."""
    from usd_export_read_import import read_import, readings

    return readings(stage, read_import(stage, articulations, bodies))


def test_importing_the_export_reproduces_the_initialized_engine(tmp_path):
    """Every exported value of a PhysX import in a new process equals the live engine's, including config overrides."""
    from isaaclab_isaacsim.physics import PhysxCfg

    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationCfg, build_simulation_context
    from isaaclab.test.utils.usd_export import assert_differences_match, find_reading_differences, make_export_scene_cfg

    sim_cfg = SimulationCfg(device="cpu", dt=1 / 120, gravity=(0.2, -0.1, -4.0), physics=PhysxCfg())
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene = InteractiveScene(make_export_scene_cfg(tmp_path))
        sim.reset()
        stage, articulations, bodies, live = _read_live(scene, scene.export_to_usd(str(tmp_path / "exported.usda")))
        gravity = np.asarray(sim.backend_manager.get_gravity())

    hinge = live["/World/envs/env_0/Robot/Hinge", "angular"]
    assert hinge["position"] == pytest.approx(0.21)
    assert hinge["stiffness"] == pytest.approx(83.0)
    assert hinge["armature"] == pytest.approx(0.023)
    assert_differences_match(find_reading_differences(live, _read_import(stage, articulations, bodies)), {})
    scene_prim = UsdPhysics.Scene(stage.GetPrimAtPath("/physicsScene"))
    exported = np.asarray(scene_prim.GetGravityDirectionAttr().Get()) * scene_prim.GetGravityMagnitudeAttr().Get()
    np.testing.assert_allclose(exported, gravity, rtol=1e-5)


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="PhysX simulates deformables on the GPU")
def test_importing_the_export_starts_a_volume_deformable_at_its_reset_state(tmp_path):
    """A PhysX import starts a volume deformable at the nodal state Isaac Lab resets it to, which the export states.

    The reset state is moved off the authored points, so that only the stated one reproduces it.
    """
    from isaaclab_isaacsim.physics import PhysxCfg
    from isaaclab_isaacsim.sim.schemas import PhysxDeformableBodyCfg
    from isaaclab_isaacsim.sim.spawners.materials import PhysxDeformableBodyMaterialCfg
    from usd_export_read_import import read_import_nodal_positions

    import isaaclab.sim as sim_utils
    from isaaclab.assets import DeformableObjectCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg, build_simulation_context
    from isaaclab.sim.usd_export.base_usd_exporter import BaseUsdExporter

    output = str(tmp_path / "exported.usda")
    cfg = InteractiveSceneCfg(num_envs=1, env_spacing=1.0, export_usd_path=output)
    cfg.beam = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Beam",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, -0.3, 1.0)),
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.3, 0.04, 0.04),
            volume_deformable_props=[sim_utils.OmniPhysicsDeformableBodyCfg(), PhysxDeformableBodyCfg()],
            physics_material=PhysxDeformableBodyMaterialCfg(youngs_modulus=2e5, poissons_ratio=0.3, density=500.0),
        ),
    )
    with build_simulation_context(sim_cfg=SimulationCfg(device="cuda:0", physics=PhysxCfg())) as sim:
        scene = InteractiveScene(cfg)
        sim.reset()
        beam = scene.deformable_objects["beam"]
        beam.data.default_nodal_state_w.torch[..., 2] += 0.05  # positions, then velocities, per node
        beam.write_nodal_state_to_sim_index(beam.data.default_nodal_state_w)
        scene.export_to_usd(output)
        path = beam.root_view.prim_paths[0]
        reset = beam.data.default_nodal_state_w.torch[0, :, :3].cpu().numpy()
    loaded = read_import_nodal_positions(BaseUsdExporter.open_stage(output), [path])

    # The reader sees the import after the warm-up step GPU tensor access needs, in which gravity lowers the nodes by
    # under a millimeter (measured 0.36 mm); the stated reset state is 50 mm off the authored one.
    np.testing.assert_allclose(loaded[path], reset, atol=1e-3)


@pytest.mark.parametrize("task", TASKS)
def test_importing_the_export_of_a_task_reproduces_the_live_engine(tmp_path, task):
    """A PhysX import of a task's export holds the live engine's values and initial joint state."""
    from isaaclab.test.utils.usd_export import (
        assert_differences_match,
        export_task,
        find_reading_differences,
        reading_pairs,
    )

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.hydra import resolve_presets
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    env_cfg = resolve_presets(load_cfg_from_registry(task, "env_cfg_entry_point"), selected=("isaacsim_physx",))
    env_cfg.sim.device = "cuda:0"
    path = str(tmp_path / "exported.usda")
    stage, articulations, bodies, live = export_task(env_cfg, task, path, lambda scene: _read_live(scene, path))
    loaded = _read_import(stage, articulations, bodies)
    assert_differences_match(find_reading_differences(live, loaded), TASKS[task], reading_pairs(live))
