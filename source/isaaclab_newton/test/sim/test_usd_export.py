# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton USD export: loading the exported file reproduces the initialized model, for a scene that changes the
exported values and for registered tasks."""

import re

import newton
import numpy as np
import pytest
import warp as wp
from newton.usd import PrimType, SchemaResolverNewton

from pxr import Gf, Sdf, Usd, UsdPhysics, UsdShade

pytestmark = pytest.mark.kitless


def _export(
    tmp_path,
    solver_cfg,
    device,
    hinge_actuator=None,
    num_substeps=1,
    edit_robot=None,
    edit_cfg=None,
    before_export=None,
    default_shape_cfg=None,
    world_hook=None,
):
    """Initialize the fixture scene, export it, and capture the live model for comparison.

    ``edit_robot`` edits the robot's USD and ``edit_cfg`` the scene configuration before the scene is built;
    ``world_hook`` changes each world's Newton builder before it is finalized; ``before_export`` changes the
    initialized scene.
    """
    import contextlib
    from unittest import mock

    from isaaclab_newton.cloner import newton_builder_world_hook
    from isaaclab_newton.physics import NewtonCfg
    from isaaclab_newton.sim.usd_export import UsdExporter

    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationCfg, build_simulation_context
    from isaaclab.test.utils.usd_export import make_export_scene_cfg

    cfg = make_export_scene_cfg(tmp_path)
    if edit_robot is not None:
        robot = Usd.Stage.Open(cfg.robot.spawn.usd_path)
        edit_robot(robot)
        robot.GetRootLayer().Save()
    if hinge_actuator is not None:
        cfg.robot.actuators = {"hinge": hinge_actuator}
    if edit_cfg is not None:
        edit_cfg(cfg)
    sim_cfg = SimulationCfg(
        device=device,
        dt=1 / 120,
        gravity=(0.2, -0.1, -4.0),
        physics=NewtonCfg(solver_cfg=solver_cfg, num_substeps=num_substeps),
    )
    if default_shape_cfg is not None:
        sim_cfg.physics.default_shape_cfg = default_shape_cfg
    output = str(tmp_path / "exported.usda")
    hook = newton_builder_world_hook(world_hook) if world_hook is not None else contextlib.nullcontext()

    def without_writing(detect):
        def detect_without_writing(exporter, stage):
            before = stage.GetRootLayer().ExportToString()
            edits = detect(exporter, stage)
            assert stage.GetRootLayer().ExportToString() == before  # detection leaves writing to the edits
            return edits

        return detect_without_writing

    detection = contextlib.ExitStack()
    for name in ("compute_diff", "_compute_pd_drive_edits"):
        detection.enter_context(mock.patch.object(UsdExporter, name, without_writing(getattr(UsdExporter, name))))
    with build_simulation_context(sim_cfg=sim_cfg) as sim, hook, detection:
        scene = InteractiveScene(cfg)
        sim.reset()
        if before_export is not None:
            before_export(scene)
        scene.export_to_usd(output)
        live = _capture_live(scene)
    return output, live


def _capture_live(scene) -> dict:
    """:func:`_capture` of the live model, after Isaac Lab's reset writes the initial joint state the export states."""
    from isaaclab.test.utils.usd_export import write_default_joint_state

    write_default_joint_state(scene)
    backend = scene.sim.backend_manager.backend
    return _capture(backend.model, backend.state_0.body_q, backend.state_0.joint_q, backend.state_0.joint_qd)


def _newton_import(output, solver, device):
    """Newton's own USD import of the exported file, as a consumer without Isaac Lab does it."""
    from newton.usd import SchemaResolverMjc, SchemaResolverPhysx

    builder = newton.ModelBuilder()
    resolvers = [SchemaResolverNewton(), SchemaResolverPhysx()]
    if solver == "mujoco":
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        resolvers.append(SchemaResolverMjc())
    builder.add_usd(Usd.Stage.Open(output), schema_resolvers=resolvers)
    return builder.finalize(device=device)


def _capture(model, body_q, joint_q=None, joint_qd=None):
    """Every float model array, and the joint target modes, of world 0's bodies, colliding shapes, joints, DOFs and
    coordinates.

    Keyed by attribute name and entity label; a DOF or coordinate is labelled ``<joint label>:<index in the
    joint>``. Root body poses come from ``body_q`` and the joint state from ``joint_q`` and ``joint_qd`` (default:
    the model's); other poses follow the joint state. The world's gravity is keyed ``("gravity", "")``. Model
    values the export does not carry show up as differences.
    """
    frequency = newton.Model.AttributeFrequency
    labels = {
        frequency.BODY: list(model.body_label),
        frequency.SHAPE: list(model.shape_label),
        frequency.JOINT: list(model.joint_label),
        frequency.JOINT_DOF: [None] * model.joint_dof_count,
        frequency.JOINT_COORD: [None] * model.joint_coord_count,
    }
    worlds = {
        frequency.BODY: model.body_world.numpy(),
        frequency.SHAPE: model.shape_world.numpy(),
        frequency.JOINT: model.joint_world.numpy(),
        frequency.JOINT_DOF: np.empty(model.joint_dof_count, dtype=int),
        frequency.JOINT_COORD: np.empty(model.joint_coord_count, dtype=int),
    }
    types = model.joint_type.numpy()
    # Free joints of floating bodies are generated, under generated names; their bodies carry the pose.
    worlds[frequency.JOINT][types == int(newton.JointType.FREE)] = -2
    # Visual shapes, and shape colors, are not physics: an import loads them as it is configured to.
    worlds[frequency.SHAPE][(model.shape_flags.numpy() & int(newton.ShapeFlags.COLLIDE_SHAPES)) == 0] = -2
    for entity, starts in ((frequency.JOINT_DOF, model.joint_qd_start), (frequency.JOINT_COORD, model.joint_q_start)):
        starts = starts.numpy()
        for joint, label in enumerate(model.joint_label):
            for axis, index in enumerate(range(starts[joint], starts[joint + 1])):
                labels[entity][index] = f"{label}:{axis}"
                worlds[entity][index] = worlds[frequency.JOINT][joint]
    state = {"body_q": body_q, "joint_q": joint_q, "joint_qd": joint_qd}
    parents, children = model.joint_parent.numpy(), model.joint_child.numpy()
    roots = {model.body_label[child] for parent, child in zip(parents, children) if parent == -1}
    values = {}
    for name, entity in model.attribute_frequency.items():
        namespace, _, attribute = name.rpartition(":")
        array = getattr(getattr(model, namespace) if namespace else model, attribute, None)
        if entity not in labels or not isinstance(array, wp.array):
            continue
        array = (array if state.get(name) is None else state[name]).numpy()
        if (array.dtype.kind != "f" and name != "joint_target_mode") or name == "shape_color":
            continue
        for index in np.flatnonzero(np.isin(worlds[entity], (-1, 0))):
            if name != "body_q" or labels[entity][index] in roots:
                values[name, labels[entity][index]] = array[index]
    gravity = model.gravity.numpy()
    values["gravity", ""] = gravity[0] if gravity.ndim == 2 else gravity
    return values


def _differences(live: dict, loaded: dict) -> set[tuple[str, str]]:
    """The ``(value name, entity label)`` pairs of two :func:`_capture` results that differ.

    A pose's rotation compares up to the quaternion's sign; an entity only one side has differs as ``"missing"``.
    """
    differences = {("missing", label) for _, label in live.keys() ^ loaded.keys()}
    for key in live.keys() & loaded.keys():
        a, b = np.asarray(live[key], dtype=float), np.asarray(loaded[key], dtype=float)
        if key[0] in ("body_q", "joint_X_p", "joint_X_c"):
            sign = 1.0 if np.dot(a[3:], b[3:]) >= 0 else -1.0
            same = np.allclose(b[:3], a[:3], atol=1e-5) and np.allclose(sign * b[3:], a[3:], atol=1e-5)
        else:
            same = a.shape == b.shape and np.allclose(b, a, rtol=1e-4, atol=1e-7)
        if not same:
            differences.add(key)
    return differences


@pytest.mark.parametrize("solver,device", [("xpbd", "cpu"), ("mujoco", "cuda:0")])
def test_importing_the_export_reproduces_the_initialized_model(tmp_path, caplog, solver, device):
    """Newton's own import of the file reproduces every live model value, including config overrides.

    The exported layer references the robot file instead of copying it, and, with substeps, states the solver step
    rate, which Newton's import returns as its step.
    """
    from isaaclab_newton.physics import MJWarpSolverCfg, XPBDSolverCfg

    from isaaclab.test.utils.usd_export import assert_differences_match

    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("MuJoCo Warp requires CUDA.")
    solver_cfg = (
        XPBDSolverCfg(iterations=7) if solver == "xpbd" else MJWarpSolverCfg(iterations=17, enable_multiccd=True)
    )
    num_substeps = 2 if solver == "xpbd" else 1  # the XPBD import is not stepped, so substeps change no comparison
    output, live = _export(tmp_path, solver_cfg, device, num_substeps=num_substeps)

    layer = Sdf.Layer.FindOrOpen(output)
    assert str(tmp_path / "robot.usda") in layer.externalReferences
    assert layer.GetPrimAtPath("/World/envs/env_0/Robot").referenceList.GetAddedOrExplicitItems()
    # The robot's own prims come from the reference; only edits are authored over them.
    link = layer.GetPrimAtPath("/World/envs/env_0/Robot/Link")
    assert not link or link.specifier == Sdf.SpecifierOver
    result = newton.ModelBuilder().add_usd(Usd.Stage.Open(output), schema_resolvers=[SchemaResolverNewton()])
    assert result["physics_dt"] == pytest.approx(1 / 120 / num_substeps)

    model = _newton_import(output, solver, device)
    loaded = _capture(model, model.body_q)
    if solver == "mujoco":
        # MuJoCo reads its options from the physics scene, so the default solver runs the configured one.
        imported_solver = newton.solvers.SolverMuJoCo(model)
        assert imported_solver.mjw_model.opt.iterations == 17
        # Configuration flags have their own USD names; disable flags are stated inverted.
        assert bool(model.mujoco.flag_multiccd.numpy()[0]) and bool(model.mujoco.flag_contact.numpy()[0])
        # Options equal to Newton's import defaults are left to the importer.
        stage = Usd.Stage.Open(output)
        scene = next(prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene))
        assert scene.GetAttribute("mjc:option:iterations").HasAuthoredValue()
        assert not scene.GetAttribute("mjc:option:cone").HasAuthoredValue()
        assert not scene.GetAttribute("mjc:flag:contact").HasAuthoredValue()
    # The actuator configuration differs from the USD drive, and the configured initial state from the USD's (zero),
    # so the export must carry them.
    hinge = "/World/envs/env_0/Robot/Hinge:0"
    assert live["joint_q", hinge] == pytest.approx(0.21)
    assert live["joint_target_ke", hinge] == pytest.approx(83.0)
    assert live["joint_armature", hinge] == pytest.approx(0.023)
    assert live["joint_target_mode", hinge] == int(newton.JointTargetMode.POSITION_VELOCITY)
    if solver != "mujoco":
        # The import infers position mode, which XPBD does not apply, so the export does not warn about it.
        assert "target mode" not in caplog.text
        assert "USD does not state the settings of XPBDSolverCfg" in caplog.text
        del live["joint_target_mode", hinge], loaded["joint_target_mode", hinge]
    assert_differences_match(_differences(live, loaded), {})


TASKS = {
    # Franka USD physics variant, mimic finger, position-velocity joints as MuJoCo velocity actuators, rigid object
    "Isaac-Lift-Franka": ("newton_mjwarp", {}),
    # a second articulation (cabinet with a prismatic drawer); edits under instanced assets
    "Isaac-Open-Drawer-Franka": ("newton_mjwarp", {}),
    # floating base, LSTM network actuator (re-saved next to the export), rough terrain
    "Isaac-Velocity-Rough-AnymalD": ("newton_mjwarp", {}),
    # explicit DC motor actuator, which stays a NewtonActuator prim
    "Isaac-Velocity-Flat-UnitreeGo2": ("newton_mjwarp", {}),
    # joints Isaac Lab gives gains without a USD drive
    "Isaac-Ant": ("newton_mjwarp", {}),
    # the export of a DirectRLEnv
    "Isaac-Cartpole-Direct": ("newton_mjwarp", {}),
    # the export of a DirectMARLEnv; two hands with fixed tendons
    "Isaac-Shadow-Handover-Direct": ("newton_mjwarp", {}),
    # Kamino solver options and constraint stabilization gains; the import infers another target mode for the joints
    # whose gains USD cannot tell apart, which the export warns about
    "IsaacContrib-DrLegs-Walk": (
        "newton_kamino",
        {"joint_target_mode": r".*/Robot/Joints/(j1_[lr]_i|j2_[lr]_[io]|j6_[lr]_i|j7_[lr]_[io]):0"},
    ),
    # cable: Newton's import gives the segments it generates its own contact gap (Known gaps C3)
    "Isaac-Lift-Cable-Franka": ("newton_mjwarp_vbd_proxy", {"shape_gap": r".*_edge_capsule_\d+"}),
}
"""Task -> (physics preset, values the import holds differently: value name -> regular expression of entities)."""


@pytest.mark.parametrize("task", TASKS)
def test_importing_the_export_of_a_task_builds_its_live_model(tmp_path, task):
    """Newton's import of a task's export, with the solver's schema resolvers, holds the live model's values and
    initial joint state."""
    from isaaclab_newton.physics.newton_builder import register_builder_attributes, usd_schema_resolvers

    from isaaclab.sim.usd_export.base_usd_exporter import BaseUsdExporter
    from isaaclab.test.utils.usd_export import assert_differences_match, export_task

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.hydra import resolve_presets
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    preset, expected = TASKS[task]
    env_cfg = resolve_presets(load_cfg_from_registry(task, "env_cfg_entry_point"), selected=(preset,))
    env_cfg.sim.device = "cuda:0" if wp.is_cuda_available() else "cpu"
    output = str(tmp_path / "exported.usda")
    solver_cfg, live = export_task(
        env_cfg, task, output, lambda scene: (scene.sim.cfg.physics.solver_cfg, _capture_live(scene))
    )
    builder = newton.ModelBuilder()
    register_builder_attributes(builder, solver_cfg)
    builder.add_usd(
        BaseUsdExporter.open_stage(output),
        schema_resolvers=usd_schema_resolvers(solver_cfg),
        allow_unsafe_torch_checkpoints=True,  # the network actuators the export copied next to the file
    )
    model = builder.finalize(device="cpu")
    assert_differences_match(_differences(live, _capture(model, model.body_q)), expected, live)


def test_importing_an_exported_cable_reproduces_it_but_its_segment_gaps(tmp_path, caplog):
    """Newton's import builds a cable from its USD curve; the segments it generates take its own contact gap.

    A segment friction Isaac Lab changes is stated on the curve's material, which the import reads; a changed gain of
    a cable joint or mass of a segment has no prim to state it on, and the export warns.
    """
    from isaaclab_newton.physics import NewtonCfg, VBDSolverCfg
    from newton.usd import SchemaResolverPhysx

    from isaaclab.assets import CableObjectCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg, UsdPhysicsCollisionCfg, build_simulation_context
    from isaaclab.sim.spawners.materials import CableMaterialCfg
    from isaaclab.sim.spawners.shapes import CableCfg
    from isaaclab.test.utils.usd_export import assert_differences_match

    cfg = InteractiveSceneCfg(num_envs=1, env_spacing=3.0)
    cfg.cable = CableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cable",
        spawn=CableCfg(
            positions=[(0.1 * index, 0.0, 0.5) for index in range(4)],
            physics_material=CableMaterialCfg(thickness=0.01, density=100.0, stretch_stiffness=1e6, bend_stiffness=1e3),
            collision_props=UsdPhysicsCollisionCfg(collision_enabled=True),
        ),
    )
    sim_cfg = SimulationCfg(device="cpu", dt=1 / 120, physics=NewtonCfg(solver_cfg=VBDSolverCfg()))
    output = str(tmp_path / "exported.usda")
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene = InteractiveScene(cfg)
        sim.reset()
        model = sim.backend_manager.backend.model
        model.shape_material_mu.assign(np.full(model.shape_count, 0.7, dtype=np.float32))
        stiffness, masses, starts = model.joint_target_ke.numpy(), model.body_mass.numpy(), model.joint_qd_start.numpy()
        types = model.joint_type.numpy()
        cable_joint = next(j for j in range(model.joint_count) if types[j] != int(newton.JointType.FREE))
        stiffness[starts[cable_joint]] *= 2.0
        masses[0] *= 2.0
        model.joint_target_ke.assign(stiffness)
        model.body_mass.assign(masses)
        scene.export_to_usd(output)
        live = _capture(model, sim.backend_manager.backend.state_0.body_q)
    builder = newton.ModelBuilder()
    newton.solvers.SolverVBD.register_custom_attributes(builder)
    builder.add_usd(Usd.Stage.Open(output), schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()])
    model = builder.finalize(device="cpu")
    loaded = _capture(model, model.body_q)

    segments = sorted(label for name, label in live if name == "shape_gap")
    assert len(segments) == 3
    joint, body = f"{model.joint_label[cable_joint]}:0", model.body_label[0]
    expected = {"shape_gap": "|".join(map(re.escape, segments)), "joint_target_ke": re.escape(joint)}
    assert_differences_match(_differences(live, loaded), expected | {"body_mass": re.escape(body)}, live)
    assert "contact values" in caplog.text
    assert "joints it generates" in caplog.text and "bodies it generates" in caplog.text


def test_inertia_newton_derives_on_import_is_not_written(tmp_path):
    """Inertia Newton's import derives is left to it when Isaac Lab changes a body's mass: one the import corrects
    (the link's moments break the triangle inequality), and one it derives from the geometry (the box's USD states
    none)."""
    from isaaclab_newton.physics import XPBDSolverCfg

    link, box = "/World/envs/env_0/Robot/Link", "/World/envs/env_0/Box"

    def unbalanced_link(stage):
        UsdPhysics.MassAPI(stage.GetPrimAtPath("/Robot/Link")).GetDiagonalInertiaAttr().Set(Gf.Vec3f(0.01, 0.01, 0.5))

    def heavier_link_and_box(scene):
        model = scene.sim.backend_manager.backend.model
        masses, labels = model.body_mass.numpy(), list(model.body_label)
        masses[labels.index(link)] = 4.0
        masses[labels.index(box)] *= 2.0
        model.body_mass.assign(masses)

    output, live = _export(
        tmp_path, XPBDSolverCfg(), "cpu", edit_robot=unbalanced_link, before_export=heavier_link_and_box
    )

    assert live["body_inertia", link][0, 0] > 0.01  # the import corrected it
    exported = Sdf.Layer.FindOrOpen(output)
    for name in ("physics:diagonalInertia", "physics:principalAxes", "newton:inertia"):
        assert exported.GetPropertyAtPath(f"{link}.{name}") is None, name
    assert exported.GetAttributeAtPath(f"{link}.physics:mass").default == pytest.approx(4.0)
    model = _newton_import(output, "xpbd", "cpu")
    loaded = _capture(model, model.body_q)
    np.testing.assert_allclose(loaded["body_inertia", link], live["body_inertia", link], rtol=1e-5)
    for name in ("body_mass", "body_inertia"):
        np.testing.assert_allclose(loaded[name, box], live[name, box], rtol=1e-5, err_msg=name)


def _ball_hinge(stage):
    stage.GetPrimAtPath("/Robot/Hinge").SetTypeName("PhysicsSphericalJoint")


def _zero_joint_state(cfg):
    cfg.robot.init_state.joint_pos = {".*": 0.0}


def _add_body(builder, world, position, orientation):
    body = builder.add_body(label=f"/World/envs/env_{world}/Extra")
    builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1, label=f"/World/envs/env_{world}/Extra/Box")


def _switch_off_link_collisions(builder, world, position, orientation):
    for shape, label in enumerate(builder.shape_label):
        if label.endswith("/Robot/Link/Collision") and builder.shape_world[shape] == world:
            builder.shape_flags[shape] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)


@pytest.mark.parametrize(
    "change,refused",
    [
        # A joint other than revolute, prismatic and rotational D6 keeps its USD values.
        pytest.param(
            {"edit_robot": _ball_hinge, "edit_cfg": _zero_joint_state, "changed_joint": True},
            "Joint '/World/envs/env_0/Robot/Hinge' changed",
            id="joint-type",
        ),
        # A world builder hook changes the model outside the USD, so an import lacks what it adds or switches.
        pytest.param(
            {"world_hook": _add_body},
            re.escape("outside the USD: ['/World/envs/env_0/Extra', '/World/envs/env_0/Extra/Box']"),
            id="hook-added",
        ),
        pytest.param(
            {"world_hook": _switch_off_link_collisions},
            re.escape("outside the USD: ['/World/envs/env_0/Robot/Link/Collision']"),
            id="hook-switched",
        ),
    ],
)
def test_changes_an_import_cannot_reproduce_are_refused(tmp_path, change, refused):
    """A change no USD the export writes would reproduce raises instead of writing a file that differs."""
    from isaaclab_newton.physics import XPBDSolverCfg

    from isaaclab.actuators import ImplicitActuatorCfg

    kwargs = dict(change)
    if kwargs.pop("changed_joint", False):
        kwargs["hinge_actuator"] = ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=83.0, damping=4.5)
    with pytest.raises(NotImplementedError, match=refused):
        _export(tmp_path, XPBDSolverCfg(), "cpu", **kwargs)


@pytest.mark.parametrize("solver", ["mujoco", "xpbd"])
def test_builder_hook_and_shape_default_changes_are_written(tmp_path, solver):
    """Changes Isaac Lab makes while building the model, which Newton's import alone does not make, are written.

    A world builder hook sets a link's contact stiffness and rolling friction before finalization, and Isaac Lab's
    shape defaults reach colliders with and without a physics material; a collider without one, or bound only to a
    render material, gets a new material that states them. Writing them keeps what an import reads elsewhere: the
    base's material, which its spawner adds to the stage, references the link's, and the base's bumper is an asset
    instance whose layer authors a size beneath it, which the instance ignores.
    """
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonShapeCfg, XPBDSolverCfg
    from isaaclab_newton.sim.usd_export.usd_exporter import MATERIAL_ATTRIBUTES

    from pxr import UsdGeom

    link_collider = "/World/envs/env_0/Robot/Link/Collision"

    def instance_bumper(robot):
        robot.CreateClassPrim("/BumperProto")
        shape = UsdGeom.Cube.Define(robot, "/BumperProto/Shape")
        shape.CreateSizeAttr().Set(0.1)
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        bumper = robot.DefinePrim("/Robot/Base/Bumper", "Xform")
        bumper.GetReferences().AddInternalReference("/BumperProto")
        UsdShade.MaterialBindingAPI.Apply(bumper).Bind(UsdShade.Material.Define(robot, "/Robot/Looks/Render"))
        robot.GetPrimAtPath("/Robot/Base/Bumper/Shape").GetAttribute("size").Set(0.3)
        bumper.SetInstanceable(True)

    def override_base_material(cfg):
        spawn = cfg.robot.spawn.func

        def spawn_with_override(prim_path, spawn_cfg, *args, **kwargs):
            root = spawn(prim_path, spawn_cfg, *args, **kwargs)
            override = UsdShade.Material.Define(root.GetStage(), f"{root.GetPath()}/Materials/BaseOverride")
            override.GetPrim().GetReferences().AddInternalReference(f"{root.GetPath()}/Materials/Link")
            UsdPhysics.MaterialAPI.Apply(override.GetPrim()).CreateDynamicFrictionAttr().Set(0.3)
            collider = root.GetStage().GetPrimAtPath(f"{root.GetPath()}/Base/Collision")
            UsdShade.MaterialBindingAPI(collider).Bind(override, materialPurpose="physics")
            return root

        cfg.robot.spawn.func = spawn_with_override

    def stiffer_link(builder, world, position, orientation):
        for shape, label in enumerate(builder.shape_label):
            if label.endswith("/Robot/Link/Collision"):
                builder.shape_material_ke[shape] = 7.0e3
                builder.shape_material_mu_rolling[shape] = 0.02

    solver_cfg = MJWarpSolverCfg() if solver == "mujoco" else XPBDSolverCfg()
    shape_cfg = NewtonShapeCfg(gap=0.02, ke=5.0e3, kd=200.0, mu=0.8)
    output, live = _export(
        tmp_path,
        solver_cfg,
        "cpu",
        edit_robot=instance_bumper,
        edit_cfg=override_base_material,
        default_shape_cfg=shape_cfg,
        world_hook=stiffer_link,
    )

    assert live["shape_material_ke", link_collider] == pytest.approx(7.0e3)
    table, ground = "/World/envs/env_0/Table/geometry/mesh", "/World/Ground/geometry/mesh"
    stage = Usd.Stage.Open(output)
    bound = [
        UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(c)).ComputeBoundMaterial("physics")[0] for c in (table, ground)
    ]
    assert bound[0] and bound[0].GetPath() == bound[1].GetPath()  # equal values share one new material
    # Only what an unset attribute would not give: the changed friction and gains.
    assert {a.GetName() for a in bound[0].GetPrim().GetAuthoredAttributes()} == {
        "physics:dynamicFriction",
        "newton:contactStiffness",
        "newton:contactDamping",
    }
    model = _newton_import(output, solver, "cpu")
    loaded = _capture(model, model.body_q)
    bumper, base_collider = "/World/envs/env_0/Robot/Base/Bumper/Shape", "/World/envs/env_0/Robot/Base/Collision"
    assert live["shape_scale", bumper] == pytest.approx((0.05, 0.05, 0.05))
    for collider in (table, link_collider, base_collider, bumper):
        for name in (*MATERIAL_ATTRIBUTES, "shape_gap", "shape_scale"):
            np.testing.assert_allclose(
                loaded[name, collider], live[name, collider], rtol=1e-5, err_msg=f"{collider} {name}"
            )


def test_a_joint_actuated_without_a_usd_drive_gets_one(tmp_path):
    """Newton's import actuates only axes with a USD drive; an axis Isaac Lab actuates gets one, also at zero gains."""
    from isaaclab_newton.physics import XPBDSolverCfg

    from isaaclab.actuators import ImplicitActuatorCfg

    def undriven(stage):
        stage.GetPrimAtPath("/Robot/Hinge").RemoveAPI(UsdPhysics.DriveAPI, "angular")

    actuator = ImplicitActuatorCfg(joint_names_expr=["Hinge"], stiffness=0.0, damping=0.0)
    output, live = _export(tmp_path, XPBDSolverCfg(), "cpu", hinge_actuator=actuator, edit_robot=undriven)

    model = _newton_import(output, "xpbd", "cpu")
    key = "joint_target_mode", "/World/envs/env_0/Robot/Hinge:0"
    assert live[key] == int(newton.JointTargetMode.EFFORT)
    assert _capture(model, model.body_q)[key] == live[key]


def test_explicit_pd_actuator_is_exported_as_a_joint_drive(tmp_path, caplog):
    """An explicit PD actuator, whose gains the engine does not hold, becomes the joint's USD drive, a force drive as
    the actuator's PD law is. With substeps, the export warns that Isaac Lab runs the actuator once per physics step."""
    from isaaclab_newton.physics import XPBDSolverCfg

    from isaaclab.actuators import IdealPDActuatorCfg

    def acceleration_drive(stage):
        drive = UsdPhysics.DriveAPI(stage.GetPrimAtPath("/Robot/Hinge"), "angular")
        drive.CreateTypeAttr().Set(UsdPhysics.Tokens.acceleration)

    actuator = IdealPDActuatorCfg(joint_names_expr=["Hinge"], stiffness=83.0, damping=4.5, actuator_effort_limit=12.0)
    output, _ = _export(
        tmp_path, XPBDSolverCfg(), "cpu", hinge_actuator=actuator, num_substeps=2, edit_robot=acceleration_drive
    )

    assert "Isaac Lab runs its collisions and actuators once per physics step" in caplog.text

    stage = Usd.Stage.Open(output)
    hinge = UsdPhysics.DriveAPI(stage.GetPrimAtPath("/World/envs/env_0/Robot/Hinge"), "angular")
    assert hinge.GetTypeAttr().Get() == UsdPhysics.Tokens.force
    assert hinge.GetStiffnessAttr().Get() == pytest.approx(np.deg2rad(83.0))
    assert hinge.GetDampingAttr().Get() == pytest.approx(np.deg2rad(4.5))
    assert hinge.GetMaxForceAttr().Get() == pytest.approx(12.0)
    # the drive replaces the actuator prim, so importers reading both apply the PD once
    assert not [prim for prim in stage.Traverse() if prim.GetTypeName() == "NewtonActuator"]
    model = _newton_import(output, "xpbd", "cpu")
    hinge_dof = model.joint_qd_start.numpy()[model.joint_label.index("/World/envs/env_0/Robot/Hinge")]
    assert model.joint_target_ke.numpy()[hinge_dof] == pytest.approx(83.0, rel=1e-5)
    assert model.joint_target_kd.numpy()[hinge_dof] == pytest.approx(4.5, rel=1e-5)


@pytest.mark.parametrize("exports", [False, True], ids=["no export path", "export path"])
def test_only_an_exporting_scene_reads_the_engine_while_simulating(tmp_path, monkeypatch, exports):
    """Only a scene that exports reads the engine at startup, which costs PhysX a view and a copy per asset; the read
    is dropped when the simulation stops and taken again when it is ready."""
    from isaaclab_newton.physics import NewtonCfg, XPBDSolverCfg

    from isaaclab.physics import PhysicsEvent
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationCfg, build_simulation_context
    from isaaclab.sim.usd_export.base_usd_exporter import BaseUsdExporter
    from isaaclab.test.utils.usd_export import make_export_scene_cfg

    reads = []  # each read returns a new base, so that a later read is told apart from the first
    monkeypatch.setattr(BaseUsdExporter, "read_engine_base", lambda self: reads.append(object()) or reads[-1])
    cfg = make_export_scene_cfg(tmp_path)
    if not exports:
        cfg.export_usd_path = None
    sim_cfg = SimulationCfg(device="cpu", dt=1 / 120, physics=NewtonCfg(solver_cfg=XPBDSolverCfg()))
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene = InteractiveScene(cfg)
        sim.reset()
        assert len(reads) == int(exports) and scene._usd_export_base is (reads[0] if exports else None)
        if exports:
            sim.backend_manager.dispatch_event(PhysicsEvent.STOP, payload={})
            assert scene._usd_export_base is None
            sim.backend_manager.dispatch_event(PhysicsEvent.PHYSICS_READY, payload={})
            assert scene._usd_export_base is reads[1]


def test_model_without_joint_dofs_has_no_changed_values():
    """A model without joint DOFs, e.g. one with only fixed joints, compares to empty builder lists."""
    from types import SimpleNamespace

    from isaaclab_newton.sim.usd_export import UsdExporter

    changed = UsdExporter._compute_changed_masks({"joint_armature": np.zeros(0)}, SimpleNamespace(joint_armature=[]))
    assert changed["joint_armature"].shape == (0,)


def test_joints_the_import_lacks_or_holds_with_other_dofs_are_unmatched():
    """A live joint is compared only with the import's joint of its label and DOF count; a repeated label is refused."""
    from isaaclab_newton.sim.usd_export.usd_exporter import _AlignedBase

    def model(joints):
        builder = newton.ModelBuilder()
        for label, add in joints:
            add(builder, -1, builder.add_link(label=f"{label}Body"), label=label)
        return builder.finalize(device="cpu")

    revolute, ball = newton.ModelBuilder.add_joint_revolute, newton.ModelBuilder.add_joint_ball
    live = model([("A", revolute), ("B", revolute), ("C", revolute)])
    assert _AlignedBase(live, model([("A", revolute), ("B", ball)]), {}).unmatched["joint"] == {1, 2}
    with pytest.raises(ValueError, match="2 entities labeled 'A'"):
        _AlignedBase(live, model([("A", revolute), ("A", revolute)]), {})


def test_merged_joint_dofs_map_to_their_usd_joints():
    """Newton merges single-axis joints between two bodies into one D6 joint; each DOF maps to its own USD joint."""
    from types import SimpleNamespace

    from isaaclab_newton.sim.usd_export import UsdExporter

    stage = Usd.Stage.CreateInMemory()
    for path in ("/Slide", "/HingeX", "/HingeY"):
        (UsdPhysics.PrismaticJoint if path == "/Slide" else UsdPhysics.RevoluteJoint).Define(stage, path)

    def array(values):
        return SimpleNamespace(numpy=lambda: np.asarray(values))

    model = SimpleNamespace(
        joint_qd_start=array([0, 3]),
        joint_type=array([int(newton.JointType.D6)]),
        joint_dof_dim=array([[1, 2]]),
        mujoco=SimpleNamespace(joint_dof_label=["/Slide", "/HingeX", "/HingeY"]),
    )
    units = UsdExporter._map_joint_dofs_to_prims(stage, model, 0, stage.GetPrimAtPath("/HingeX"))
    assert [(str(prim.GetPath()), axes) for prim, axes, _ in units] == [
        ("/Slide", [(0, "linear")]),
        ("/HingeX", [(1, "angular")]),
        ("/HingeY", [(2, "angular")]),
    ]
    del model.mujoco
    with pytest.raises(NotImplementedError, match="merges USD joints"):
        UsdExporter._map_joint_dofs_to_prims(stage, model, 0, stage.GetPrimAtPath("/HingeX"))


def test_d6_limit_gains_go_to_a_stated_joint_wide_attribute():
    """Newton reads a joint-wide limit gain ahead of per-axis ones, so a joint stating it is written there."""
    from isaaclab_newton.sim.usd_export import UsdExporter
    from isaaclab_newton.sim.usd_export.usd_exporter import D6_DOF_ATTRIBUTES

    stage = Usd.Stage.CreateInMemory()
    joint = UsdPhysics.Joint.Define(stage, "/Ball").GetPrim()
    per_axis = D6_DOF_ATTRIBUTES["joint_limit_ke"][0]
    assert UsdExporter._select_limit_gain_attribute(joint, "joint_limit_ke", per_axis) == per_axis
    joint.CreateAttribute("newton:limitStiffness", Sdf.ValueTypeNames.Float).Set(100.0)
    assert UsdExporter._select_limit_gain_attribute(joint, "joint_limit_ke", per_axis) == "newton:limitStiffness"


def test_changed_limits_are_written_with_the_mujoco_reference_and_force_range(tmp_path):
    """Newton's import subtracts a joint's MuJoCo reference from its USD limits and keeps the tighter of the drive
    force limit and the joint's MuJoCo force range, so changed limits import as live."""
    import torch
    from isaaclab_newton.physics import MJWarpSolverCfg

    def add_reference(stage):
        hinge = stage.GetPrimAtPath("/Robot/Hinge")
        hinge.CreateAttribute("mjc:ref", Sdf.ValueTypeNames.Double).Set(10.0)
        hinge.CreateAttribute("mjc:actuatorfrcrange:min", Sdf.ValueTypeNames.Double).Set(-4.0)
        hinge.CreateAttribute("mjc:actuatorfrcrange:max", Sdf.ValueTypeNames.Double).Set(4.0)

    def change_limits(scene):
        scene["robot"].write_joint_position_limit_to_sim_index(limits=torch.tensor([[[-0.6, 0.9]]], device="cuda:0"))

    output, _ = _export(tmp_path, MJWarpSolverCfg(), "cuda:0", edit_robot=add_reference, before_export=change_limits)
    model = _newton_import(output, "mujoco", "cpu")
    dof = model.joint_qd_start.numpy()[model.joint_label.index("/World/envs/env_0/Robot/Hinge")]
    assert model.joint_limit_lower.numpy()[dof] == pytest.approx(-0.6, abs=1e-5)
    assert model.joint_limit_upper.numpy()[dof] == pytest.approx(0.9, abs=1e-5)
    assert model.joint_effort_limit.numpy()[dof] == pytest.approx(19.0)  # the actuator's, above the USD range


def test_kamino_constraint_gains_are_stated_on_the_scene():
    """Kamino reads its constraint stabilization gains from the physics scene, so the export states those that differ
    from Newton's defaults."""
    from types import SimpleNamespace

    from isaaclab_newton.physics import KaminoConstraintsCfg, KaminoPADMMSolverCfg
    from isaaclab_newton.sim.usd_export import UsdExporter

    exporter = UsdExporter.__new__(UsdExporter)
    physics = SimpleNamespace(num_substeps=1)
    exporter.scene = SimpleNamespace(
        physics_scene_path="/physicsScene",
        sim=SimpleNamespace(cfg=SimpleNamespace(physics=physics), get_physics_dt=lambda: 0.01),
    )
    cfg = KaminoPADMMSolverCfg(constraints=KaminoConstraintsCfg(alpha=0.1, beta=0.2, gamma=0.01))
    edits = {edit.attribute: edit.value for edit in exporter._compute_scene_edits(cfg)}
    assert edits["newton:kamino:constraints:beta"] == pytest.approx(0.2)
    assert "newton:kamino:constraints:gamma" not in edits  # Newton's default


def test_attribute_tables_cover_newton_importer():
    """Every attribute Newton's importer maps is exported or deliberately excluded."""
    from isaaclab_newton.sim.usd_export.usd_exporter import (
        DOF_ATTRIBUTES,
        MATERIAL_ATTRIBUTES,
        SHAPE_ATTRIBUTES,
    )

    exported = {attribute for attribute, _, _ in DOF_ATTRIBUTES.values()}
    exported |= set(SHAPE_ATTRIBUTES.values()) | set(MATERIAL_ATTRIBUTES.values())
    excluded = {
        # joint state, written as JOINT_STATE_ATTRIBUTES unless the USD states these
        *(
            f"newton:{axis}:{kind}"
            for axis in ("angular", "linear", "rotX", "rotY", "rotZ")
            for kind in ("position", "velocity")
        ),
        # per-axis limit gains of multi-axis joints, which the export rejects when changed
        *(
            f"newton:{axis}:limit{gain}"
            for axis in ("linear", "angular", "rotX", "rotY", "rotZ")
            for gain in ("Stiffness", "Damping")
        ),
        # deprecated collider contact gains; the export states them on the collider's physics material
        *(SchemaResolverNewton.mapping[PrimType.SHAPE][key].name for key in ("ke", "kd", "kf", "ka")),
        # collision geometry generation, fixed at import
        "newton:maxHullVertices",
        "newton:sdfMaxResolution",
        "newton:sdfNarrowBandInner",
        "newton:sdfNarrowBandOuter",
        "newton:sdfTargetVoxelSize",
        "newton:sdfTextureFormat",
        "newton:sdfPadding",
        "newton:hydroelasticEnabled",
        "newton:hydroelasticStiffness",
        "newton:massModel",
        "newton:shellThickness",
    }
    mapped = {
        spec.name
        for prim_type in (PrimType.JOINT, PrimType.SHAPE, PrimType.MATERIAL)
        for spec in SchemaResolverNewton.mapping[prim_type].values()
    }
    assert mapped - exported - excluded == set()
