# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVPhysX USD export: an OVPhysX import of the exported file reproduces the initialized engine, for a scene that
changes the exported values and for registered tasks."""

import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from isaaclab_ov.physics import OvPhysxManager

from pxr import Sdf, Usd, UsdPhysics

pytestmark = [pytest.mark.kitless, pytest.mark.ov]

# The file loader runs in a spawned process and must import by module name.
sys.path.insert(0, str(Path(__file__).parent))

# The USD schema registry misses schemas registered after its first use, which a test's scene files make.
OvPhysxManager._ensure_physx_schemas_registered()

_DRIVELESS_GAINS = {"damping": r".*/Robot/joints/\w+:angular", "max_force": r".*/Robot/joints/\w+:angular"}

TASKS = {
    # Franka USD physics variant, mimic finger, rigid object
    "Isaac-Lift-Franka": {},
    # a second articulation (cabinet with a prismatic drawer); edits under instanced assets
    "Isaac-Open-Drawer-Franka": {},
    # floating base, LSTM network actuator
    "Isaac-Velocity-Flat-AnymalD": {},
    # explicit DC motor actuator
    "Isaac-Velocity-Flat-UnitreeGo2": {},
    # Isaac Lab gives joints without a USD drive gains, which PhysX does not apply and the export leaves out
    "Isaac-Ant": _DRIVELESS_GAINS,
    # the export of a DirectRLEnv
    "Isaac-Cartpole-Direct": {},
    # the export of a DirectMARLEnv
    "Isaac-Pendulum-MARL-Direct": {},
}
"""Task -> values the import holds differently: reading name -> regular expression of entities.

Tasks whose reset state has contacts (e.g. a cube held in a hand, feet on rough terrain) are left out: OVPhysX resolves
those contacts when it attaches a stage, so an import's joint state then differs from the live engine's, which Isaac
Lab writes after attaching.
"""


def _read_live(scene, path: str) -> tuple:
    """The exported stage, its articulation roots and other rigid bodies, and the live engine's readings of them."""
    from usd_export_read_import import read, readings

    from isaaclab.test.utils.usd_export import read_live_engine

    physx = scene.sim.backend_manager.get_physx_instance()
    return read_live_engine(scene, path, lambda stage, *prims: readings(stage, read(physx, *prims, scene.device)))


@pytest.fixture(scope="module")
def importer():
    """A spawned process that imports this module's exports one at a time, each into a new OVPhysX instance: a
    process has one attached simulation."""
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        yield pool


def _read_import(importer, stage, articulations: list[str], bodies: list[str]) -> dict:
    """Readings of a fresh OVPhysX import of ``stage``, in the ``importer`` process."""
    from usd_export_read_import import read_import, readings

    raw = importer.submit(read_import, stage.Flatten().ExportToString(), articulations, bodies).result()
    return readings(stage, raw)


def _round_trip(importer, cfg, sim_cfg, before_reset=None):
    """Export the initialized scene of ``cfg``, then read the live engine and a fresh OVPhysX import of the file.

    ``before_reset`` changes the stage before the engine loads it.

    Returns:
        The exported stage, the live and the imported readings by prim, and the live gravity [m/s^2].
    """
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import build_simulation_context

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene = InteractiveScene(cfg)
        if before_reset is not None:
            before_reset(sim.stage)
        sim.reset()
        stage, articulations, bodies, live = _read_live(scene, scene.export_to_usd(cfg.export_usd_path))
        gravity = np.asarray(sim.backend_manager.get_gravity())
    return stage, live, _read_import(importer, stage, articulations, bodies), gravity


# Without a box per environment, environment 1 exists in physics only.
def test_importing_the_export_reproduces_the_initialized_engine(tmp_path, importer):
    """Every exported value of a fresh OVPhysX import equals the live engine's, including config overrides.

    A value the engine derives is not restated: the box collider's contact offset, which its USD leaves to PhysX
    (-inf).
    """
    from isaaclab_ov.physics import OvPhysxCfg

    from isaaclab.sim import SimulationCfg
    from isaaclab.test.utils.usd_export import assert_differences_match, find_reading_differences, make_export_scene_cfg

    box = "/World/envs/env_0/Box"

    def derived_box_contact_offset(stage):
        collider = next(p for p in Usd.PrimRange(stage.GetPrimAtPath(box)) if p.HasAPI(UsdPhysics.CollisionAPI))
        collider.AddAppliedSchema("PhysxCollisionAPI")
        collider.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(float("-inf"))
        return collider.GetPath()

    sim_cfg = SimulationCfg(device="cpu", dt=1 / 120, gravity=(0.2, -0.1, -4.0), physics=OvPhysxCfg())
    colliders = []
    stage, live, loaded, gravity = _round_trip(
        importer,
        make_export_scene_cfg(tmp_path),
        sim_cfg,
        lambda stage: colliders.append(derived_box_contact_offset(stage)),
    )

    hinge = live["/World/envs/env_0/Robot/Hinge", "angular"]
    assert hinge["position"] == pytest.approx(0.21)
    assert hinge["stiffness"] == pytest.approx(83.0)
    assert hinge["armature"] == pytest.approx(0.023)
    exported_hinge = stage.GetPrimAtPath("/World/envs/env_0/Robot/Hinge")
    assert exported_hinge.GetAttribute("physxJointAxis:angular:armature").HasAuthoredValue()
    # Zero joint friction equals the schema default, so the export leaves it unset.
    assert not exported_hinge.GetAttribute("physxJointAxis:angular:staticFrictionEffort").HasAuthoredValue()
    assert np.all(np.isfinite(live[box, None]["contact_offset"]))  # derived by the engine
    assert stage.GetPrimAtPath(colliders[0]).GetAttribute("physxCollision:contactOffset").Get() == float("-inf")
    assert_differences_match(find_reading_differences(live, loaded), {})
    scene_prim = UsdPhysics.Scene(stage.GetPrimAtPath("/physicsScene"))
    exported = np.asarray(scene_prim.GetGravityDirectionAttr().Get()) * scene_prim.GetGravityMagnitudeAttr().Get()
    np.testing.assert_allclose(exported, gravity, rtol=1e-5)


@pytest.mark.parametrize("task", TASKS)
def test_importing_the_export_of_a_task_reproduces_the_live_engine(tmp_path, importer, task):
    """A fresh OVPhysX import of a task's export holds the live engine's values and initial joint state."""
    from isaaclab.test.utils.usd_export import (
        assert_differences_match,
        export_task,
        find_reading_differences,
        reading_pairs,
    )

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.hydra import resolve_presets
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    env_cfg = resolve_presets(load_cfg_from_registry(task, "env_cfg_entry_point"), selected=("ovphysx",))
    env_cfg.sim.device = "cuda:0"
    path = str(tmp_path / "exported.usda")
    stage, articulations, bodies, live = export_task(env_cfg, task, path, lambda scene: _read_live(scene, path))
    loaded = _read_import(importer, stage, articulations, bodies)
    assert_differences_match(find_reading_differences(live, loaded), TASKS[task], reading_pairs(live))


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="OVPhysX simulates deformables on the GPU")
def test_importing_the_export_starts_a_volume_deformable_at_its_reset_state(tmp_path):
    """An OVPhysX import starts a volume deformable at the nodal state Isaac Lab resets it to, which the export states.

    The reset state is moved off the authored points, so that only the stated one reproduces it. The PhysX backends
    share this export, which writes no other deformable value.
    """
    from isaaclab_isaacsim.sim.schemas import PhysxDeformableBodyCfg
    from isaaclab_isaacsim.sim.spawners.materials import PhysxDeformableBodyMaterialCfg
    from isaaclab_ov.physics import OvPhysxCfg
    from usd_export_read_import import NODAL_POSITION, read_nodal_positions

    import isaaclab.sim as sim_utils
    from isaaclab.assets import DeformableObjectCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg, build_simulation_context

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
    with build_simulation_context(sim_cfg=SimulationCfg(device="cuda:0", physics=OvPhysxCfg())) as sim:
        scene = InteractiveScene(cfg)
        sim.reset()
        beam = scene.deformable_objects["beam"]
        beam.data.default_nodal_state_w.torch[..., 2] += 0.05  # positions, then velocities, per node
        beam.write_nodal_state_to_sim_index(beam.data.default_nodal_state_w)
        scene.export_to_usd(output)
        path = beam.root_view.prim_paths[0]
        live = beam.root_view.get_attribute(NODAL_POSITION).numpy().copy()
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        loaded = pool.submit(read_nodal_positions, Usd.Stage.Open(output).Flatten().ExportToString(), [path]).result()

    np.testing.assert_allclose(loaded[path], live, atol=1e-5)


@pytest.mark.parametrize(
    "schema,authored,actuator,live_armature",
    [
        # Isaac Lab migrates these legacy attributes before the engine loads the stage; OVPhysX 0.7 ignores them.
        pytest.param(
            "PhysxJointAPI",
            {"physxJoint:armature": 0.3, "physxJoint:maxJointVelocity": 120.0},
            {"armature": None, "joint_velocity_limit": None},
            0.3,
            id="legacy-physx",
        ),
        # OVPhysX 0.7 applies a per-axis value only when positive and otherwise keeps an authored Newton value.
        pytest.param("NewtonJointAPI", {"newton:armature": 0.3}, {"armature": 0.0}, 0.0, id="newton"),
    ],
)
def test_joint_values_ovphysx_reads_from_other_attributes_are_restated(
    tmp_path, importer, schema, authored, actuator, live_armature
):
    """A joint value OVPhysX 0.7 reads from another attribute than the live engine's source is restated as the
    attribute an import reads."""
    from isaaclab_ov.physics import OvPhysxCfg

    from isaaclab.sim import SimulationCfg
    from isaaclab.test.utils.usd_export import make_export_scene_cfg

    cfg = make_export_scene_cfg(tmp_path)
    robot = Usd.Stage.Open(cfg.robot.spawn.usd_path)
    hinge = robot.GetPrimAtPath("/Robot/Hinge")
    hinge.AddAppliedSchema(schema)
    for name, value in authored.items():
        hinge.CreateAttribute(name, Sdf.ValueTypeNames.Float).Set(value)
    robot.GetRootLayer().Save()
    for name, value in actuator.items():
        setattr(cfg.robot.actuators["hinge"], name, value)
    _, live, loaded, _ = _round_trip(importer, cfg, SimulationCfg(device="cpu", dt=1 / 120, physics=OvPhysxCfg()))

    key = "/World/envs/env_0/Robot/Hinge", "angular"
    assert live[key]["armature"] == pytest.approx(live_armature)
    for name in ("armature", "max_velocity"):
        assert loaded[key][name] == pytest.approx(live[key][name], rel=1e-5), name
