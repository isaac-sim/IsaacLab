# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Round-trip tests for exporting a Newton model back to USD.

The contract under test is model idempotence, not USD fidelity::

    m1 = load(source.usda)
    export(m1) -> out.usda
    m2 = load(out.usda)
    assert m1 == m2

The exported stage is deliberately *not* compared against the source stage. Newton's importer
normalizes as it reads (unit conversion, shape-scale baking, fixed-joint body collapsing), so the
two files differ by construction. What must hold is that a second import observes no further
change -- the exported stage is a fixed point of the import.
"""

from __future__ import annotations

import newton
import numpy as np
import pytest
from isaaclab_newton.sim.usd_export import export_model_to_usd, resolve_world_prim_paths

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

# Model array attributes compared between the two imports. Grouped by concern so a failure names
# the concern rather than a bare attribute list.
BODY_ATTRS = ("body_mass", "body_inertia", "body_com", "body_q")
JOINT_ATTRS = (
    "joint_type",
    "joint_axis",
    "joint_target_ke",
    "joint_target_kd",
    "joint_limit_lower",
    "joint_limit_upper",
    "joint_armature",
    "joint_friction",
    "joint_effort_limit",
    "joint_velocity_limit",
)
SHAPE_ATTRS = ("shape_material_mu", "shape_material_restitution")


def _author_source_stage(
    path: str,
    root: str = "/World",
    principal_axes: Gf.Quatf | None = None,
    drive_gains: tuple[float, float] = (120.0, 7.5),
    max_force: float = 33.0,
    joint_local_pos0: tuple[float, float, float] | None = None,
    visual_only_shape: bool = False,
    mesh_shape: bool = False,
) -> None:
    """Author a minimal two-body articulation with a driven revolute joint.

    Values are deliberately non-default and non-round so a dropped or mis-scaled attribute cannot
    coincide with a default.

    Args:
        path: Destination path for the stage.
        root: Prim path the articulation is rooted at. Varied by tests because real assets do not
            root their bodies under ``/World``.
        principal_axes: Optional rotation for the inertia frame. A non-identity rotation produces
            products of inertia in the body frame.
    """
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    scene_prim = UsdPhysics.Scene.Define(stage, f"{root}/physicsScene").GetPrim()
    # Non-default solver settings: the defaults (1000 Hz, -1 iterations) would round-trip even if
    # the exporter dropped them entirely.
    scene_prim.CreateAttribute("newton:timeStepsPerSecond", Sdf.ValueTypeNames.Int).Set(240)
    scene_prim.CreateAttribute("newton:maxSolverIterations", Sdf.ValueTypeNames.Int).Set(37)

    world = UsdGeom.Xform.Define(stage, root)
    # Newton requires every joint to belong to an articulation; mark the root explicitly.
    UsdPhysics.ArticulationRootAPI.Apply(world.GetPrim())

    for name, pos, mass in (("Base", (0.0, 0.0, 0.5), 3.25), ("Link", (0.0, 0.0, 1.25), 1.75)):
        prim_path = f"{root}/{name}"
        cube = UsdGeom.Cube.Define(stage, prim_path)
        cube.GetSizeAttr().Set(0.4)
        cube.AddTranslateOp().Set(Gf.Vec3d(*pos))

        prim = cube.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.GetMassAttr().Set(mass)
        mass_api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(0.11, 0.22, 0.33))
        if principal_axes is not None:
            mass_api.GetPrincipalAxesAttr().Set(principal_axes)

    joint = UsdPhysics.RevoluteJoint.Define(stage, f"{root}/Link/joint")
    joint.GetBody0Rel().SetTargets([f"{root}/Base"])
    joint.GetBody1Rel().SetTargets([f"{root}/Link"])
    joint.GetAxisAttr().Set("Z")
    # Degrees on the stage; Newton converts to radians on import.
    joint.GetLowerLimitAttr().Set(-45.0)
    joint.GetUpperLimitAttr().Set(75.0)

    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.GetStiffnessAttr().Set(drive_gains[0])
    drive.GetDampingAttr().Set(drive_gains[1])
    drive.GetMaxForceAttr().Set(max_force)

    if joint_local_pos0 is not None:
        joint.GetLocalPos0Attr().Set(Gf.Vec3f(*joint_local_pos0))

    if visual_only_shape:
        # A gprim under a body with no CollisionAPI: Newton imports it as a visible, non-colliding
        # shape, which the exporter must not turn into a collider.
        visual = UsdGeom.Cube.Define(stage, f"{root}/Link/visual_only")
        visual.GetSizeAttr().Set(0.2)

    if mesh_shape:
        # A visible mesh with a collision approximation: the importer turns it into a collision-only
        # convex shape and adds a visual-only copy of the mesh without a prim of its own.
        mesh = UsdGeom.Mesh.Define(stage, f"{root}/Link/mesh")
        h = 0.1
        mesh.CreatePointsAttr([Gf.Vec3f(x, y, z) for x in (-h, h) for y in (-h, h) for z in (-h, h)])
        mesh.CreateFaceVertexCountsAttr([4] * 6)
        mesh.CreateFaceVertexIndicesAttr([0, 1, 3, 2, 4, 6, 7, 5, 0, 4, 5, 1, 2, 3, 7, 6, 0, 2, 6, 4, 1, 5, 7, 3])
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set(UsdPhysics.Tokens.convexHull)

    # Newton-specific extras, read through SchemaResolverNewton.
    joint_prim = joint.GetPrim()
    joint_prim.CreateAttribute("newton:armature", Sdf.ValueTypeNames.Float).Set(0.017)
    joint_prim.CreateAttribute("newton:friction", Sdf.ValueTypeNames.Float).Set(0.045)

    stage.GetRootLayer().Save()


def _load(path: str) -> tuple[newton.Model, dict]:
    """Import ``path`` into a Newton model, returning the model and the importer's result maps."""
    builder = newton.ModelBuilder()
    stage_info = builder.add_usd(str(path))
    return builder.finalize(), stage_info


def _assert_arrays_equal(m1: newton.Model, m2: newton.Model, attrs: tuple[str, ...], group: str) -> None:
    """Assert every attribute in ``attrs`` matches between the two models."""
    mismatched = []
    for attr in attrs:
        a, b = getattr(m1, attr, None), getattr(m2, attr, None)
        if a is None and b is None:
            continue
        if a is None or b is None:
            mismatched.append(f"{attr}: present on only one model")
            continue
        arr_a, arr_b = np.asarray(a.numpy()), np.asarray(b.numpy())
        if arr_a.shape != arr_b.shape:
            mismatched.append(f"{attr}: shape {arr_a.shape} != {arr_b.shape}")
        elif not np.allclose(arr_a, arr_b, rtol=1e-5, atol=1e-6, equal_nan=True):
            delta = np.max(np.abs(arr_a - arr_b))
            mismatched.append(f"{attr}: max|delta|={delta:.6g}\n  reimport={arr_a}\n  export  ={arr_b}")
    assert not mismatched, f"{group} differ after export/reimport:\n" + "\n".join(mismatched)


@pytest.fixture
def source_stage(tmp_path):
    """Path to a freshly authored source stage."""
    path = tmp_path / "source.usda"
    _author_source_stage(str(path))
    return path


def test_round_trip_preserves_body_properties(source_stage, tmp_path):
    """Body mass, inertia, COM and pose survive export and reimport."""
    m1, stage_info = _load(source_stage)
    out = tmp_path / "exported.usda"
    export_model_to_usd(m1, str(out), stage_info=stage_info)
    m2, _ = _load(out)

    assert m1.body_count == m2.body_count, f"body_count {m1.body_count} != {m2.body_count}"
    _assert_arrays_equal(m1, m2, BODY_ATTRS, "body properties")


def test_round_trip_preserves_joint_properties(source_stage, tmp_path):
    """Joint drive gains, limits, armature and friction survive export and reimport."""
    m1, stage_info = _load(source_stage)
    out = tmp_path / "exported.usda"
    export_model_to_usd(m1, str(out), stage_info=stage_info)
    m2, _ = _load(out)

    assert m1.joint_count == m2.joint_count, f"joint_count {m1.joint_count} != {m2.joint_count}"
    _assert_arrays_equal(m1, m2, JOINT_ATTRS, "joint properties")


def test_round_trip_preserves_shape_materials(source_stage, tmp_path):
    """Shape friction and restitution survive export and reimport."""
    m1, stage_info = _load(source_stage)
    out = tmp_path / "exported.usda"
    export_model_to_usd(m1, str(out), stage_info=stage_info)
    m2, _ = _load(out)

    assert m1.shape_count == m2.shape_count, f"shape_count {m1.shape_count} != {m2.shape_count}"
    _assert_arrays_equal(m1, m2, SHAPE_ATTRS, "shape materials")


def test_round_trip_preserves_solver_settings(source_stage, tmp_path):
    """Solver settings survive the round-trip.

    Solver configuration lives on the physics scene prim rather than in the model, so exporting the
    model alone would silently drop it. The fixture authors non-default values so this cannot pass
    by defaulting.
    """
    m1, stage_info = _load(source_stage)
    out = tmp_path / "exported.usda"
    export_model_to_usd(m1, str(out), stage_info=stage_info)
    _, stage_info_2 = _load(out)

    assert stage_info_2["physics_dt"] == pytest.approx(stage_info["physics_dt"]), "physics timestep lost on export"
    assert stage_info_2["max_solver_iterations"] == stage_info["max_solver_iterations"], (
        "max solver iterations lost on export"
    )


def test_export_is_a_fixed_point(source_stage, tmp_path):
    """Exporting the reimported model reproduces the first export byte-for-byte.

    A difference here means the exporter carries hidden state: the second export saw a model that
    the first export failed to fully describe.
    """
    m1, stage_info = _load(source_stage)
    first = tmp_path / "first.usda"
    export_model_to_usd(m1, str(first), stage_info=stage_info)

    m2, stage_info_2 = _load(first)
    second = tmp_path / "second.usda"
    export_model_to_usd(m2, str(second), stage_info=stage_info_2)

    assert first.read_text() == second.read_text(), "export is not idempotent; exporter carries hidden state"


def test_round_trip_when_bodies_are_not_under_world(tmp_path):
    """Assets that root their bodies outside ``/World`` still reimport.

    Regression test: the articulation root was previously hardcoded to ``/World``, so exports of
    assets rooted elsewhere (``/cartpole``, ``/env``) produced a stage whose articulation root
    contained no bodies, and every reimport failed with "joints not belonging to any articulation".
    """
    source = tmp_path / "rooted_elsewhere.usda"
    _author_source_stage(str(source), root="/robot")
    m1, stage_info = _load(source)

    out = tmp_path / "exported.usda"
    export_model_to_usd(m1, str(out), stage_info=stage_info)
    m2, _ = _load(out)

    assert m1.joint_count == m2.joint_count
    _assert_arrays_equal(m1, m2, JOINT_ATTRS, "joint properties")


def test_round_trip_preserves_products_of_inertia(tmp_path):
    """A body inertia tensor with off-diagonal terms survives the round-trip.

    Regression test: ``UsdPhysics.MassAPI`` stores inertia as a diagonal in a rotated frame, so
    authoring only ``diagonalInertia`` silently dropped the products of inertia.
    """
    source = tmp_path / "rotated_inertia.usda"
    # A principal frame rotated 45 degrees about Z produces off-diagonal terms in the body frame.
    _author_source_stage(str(source), principal_axes=Gf.Quatf(0.9238795, 0.0, 0.0, 0.3826834))
    m1, stage_info = _load(source)

    inertia = m1.body_inertia.numpy().reshape(m1.body_count, 3, 3)
    off_diagonal = max(abs(inertia[i][j][k]) for i in range(m1.body_count) for j, k in ((0, 1), (0, 2), (1, 2)))
    assert off_diagonal > 1e-6, "fixture failed to produce products of inertia; test would be vacuous"

    out = tmp_path / "exported.usda"
    export_model_to_usd(m1, str(out), stage_info=stage_info)
    m2, _ = _load(out)

    _assert_arrays_equal(m1, m2, ("body_inertia",), "body inertia")


def test_round_trip_preserves_effort_limit_without_drive_gains(tmp_path):
    """A torque-controlled joint keeps its effort limit.

    Regression test: the drive was only authored when stiffness or damping was non-zero, so
    torque-controlled joints (zero gains, finite effort limit -- e.g. the shipped Cartpole asset)
    silently lost their effort limit and reimported at Newton's unlimited default.
    """
    source = tmp_path / "torque_controlled.usda"
    _author_source_stage(str(source), drive_gains=(0.0, 0.0), max_force=1000.0)
    m1, stage_info = _load(source)

    assert not m1.joint_target_ke.numpy().any(), "fixture authored drive gains; test would be vacuous"

    out = tmp_path / "exported.usda"
    export_model_to_usd(m1, str(out), stage_info=stage_info)
    m2, _ = _load(out)

    _assert_arrays_equal(m1, m2, ("joint_effort_limit",), "joint effort limit")


def test_aliased_prim_paths_export_one_prim_per_shape(source_stage, tmp_path):
    """Several prim paths referring to one shape export as a single collider.

    Regression test: the importer's path maps are many-to-one -- nested prims such as
    ``.../mesh`` and ``.../mesh/mesh`` resolve to the same shape index. Authoring one prim per
    *path* emitted more colliders than the model held (the shipped Rizon4s asset reimported with 27
    shapes instead of 21).
    """
    m1, stage_info = _load(source_stage)
    # Alias every shape under a deeper nested path, as the real assets do.
    aliased = dict(stage_info["path_shape_map"])
    for path, index in list(stage_info["path_shape_map"].items()):
        aliased[f"{path}/mesh"] = index
    stage_info = {**stage_info, "path_shape_map": aliased}
    assert len(aliased) > len(set(aliased.values())), "fixture failed to alias paths; test would be vacuous"

    out = tmp_path / "exported.usda"
    export_model_to_usd(m1, str(out), stage_info=stage_info)
    m2, _ = _load(out)

    assert m1.shape_count == m2.shape_count, f"shape_count {m1.shape_count} != {m2.shape_count}"


def test_round_trip_preserves_joint_frames(tmp_path):
    """Joint attachment frames survive the round-trip.

    Regression test: ``joint_X_p`` / ``joint_X_c`` were never authored, so every joint collapsed to
    its body origin on reimport and the articulation reassembled in the wrong pose.
    """
    source = tmp_path / "offset_joint.usda"
    _author_source_stage(str(source), joint_local_pos0=(0.37, -0.11, 0.05))
    m1, stage_info = _load(source)

    assert np.abs(m1.joint_X_p.numpy()[:, :3]).max() > 1e-6, "fixture produced no joint offset; test would be vacuous"

    out = tmp_path / "exported.usda"
    export_model_to_usd(m1, str(out), stage_info=stage_info)
    m2, _ = _load(out)

    _assert_arrays_equal(m1, m2, ("joint_X_p", "joint_X_c"), "joint frames")


def test_visual_only_shapes_do_not_become_colliders(tmp_path):
    """A visible, non-colliding shape stays non-colliding through the round-trip.

    Regression test: every exported shape was given a ``CollisionAPI``, so visual meshes reimported
    as colliders and silently enlarged the collision set.
    """
    source = tmp_path / "with_visual.usda"
    _author_source_stage(str(source), visual_only_shape=True)
    m1, stage_info = _load(source)

    collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
    assert any(not (int(f) & collide) for f in m1.shape_flags.numpy()), (
        "fixture produced no visual-only shape; test would be vacuous"
    )

    out = tmp_path / "exported.usda"
    export_model_to_usd(m1, str(out), stage_info=stage_info)
    m2, _ = _load(out)

    assert m1.shape_count == m2.shape_count
    np.testing.assert_array_equal(
        m1.shape_flags.numpy(), m2.shape_flags.numpy(), err_msg="shape flags changed on reimport"
    )


def test_visual_twins_of_approximated_meshes_round_trip(tmp_path):
    """A mesh the importer approximates for collision keeps its visual copy through the round-trip.

    ``add_usd`` runs ``approximate_meshes(keep_visual_shapes=True)`` on visible meshes that carry a
    collision approximation: the prim becomes a collision-only convex shape and a visual-only copy
    labelled ``<path>_visual`` is added with no prim of its own. The copy is exported as a visual
    sibling, so the stage still renders what the model renders, and a reimport recreates both shapes.
    """
    root = "/World"
    source = tmp_path / "with_mesh.usda"
    _author_source_stage(str(source), root=root, mesh_shape=True)
    m1, stage_info = _load(source)

    labels = list(m1.shape_label)
    twin_path = f"{root}/Link/mesh_visual"
    assert twin_path in labels and twin_path not in stage_info["path_shape_map"], (
        "fixture produced no visual twin; the test would be vacuous"
    )
    paths = resolve_world_prim_paths(m1, stage_info)
    assert paths.shapes[labels.index(twin_path)] == twin_path

    out = tmp_path / "exported.usda"
    export_model_to_usd(m1, str(out), stage_info=stage_info)
    exported = Usd.Stage.Open(str(out))
    twin, collider = exported.GetPrimAtPath(twin_path), exported.GetPrimAtPath(f"{root}/Link/mesh")
    assert twin.IsA(UsdGeom.Mesh) and not twin.HasAPI(UsdPhysics.CollisionAPI), "twin exported as a collider"
    assert UsdGeom.Imageable(twin).ComputeVisibility() == UsdGeom.Tokens.inherited, "twin exported invisible"
    assert collider.HasAPI(UsdPhysics.CollisionAPI), "approximated mesh lost its collider"
    assert UsdGeom.Imageable(collider).ComputeVisibility() == UsdGeom.Tokens.invisible, "collider exported visible"

    m2, stage_info2 = _load(out)
    assert m2.shape_count == m1.shape_count
    assert sorted(m1.shape_flags.numpy().tolist()) == sorted(m2.shape_flags.numpy().tolist())
    assert set(resolve_world_prim_paths(m2, stage_info2).shapes.values()) == set(paths.shapes.values())


def _rollout(model, frames: int = 40, fps: int = 60, substeps: int = 8) -> np.ndarray:
    """Simulate ``model`` and return its body-pose trajectory, shape [frames, bodies, 7]."""
    solver = newton.solvers.SolverXPBD(model)
    state_in, state_out = model.state(), model.state()
    control = model.control()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

    dt = 1.0 / fps / substeps
    trajectory = []
    for _ in range(frames):
        for _ in range(substeps):
            state_in.clear_forces()
            pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, dt)
            state_in, state_out = state_out, state_in
        trajectory.append(state_in.body_q.numpy().copy())
    return np.asarray(trajectory)


def test_exported_stage_simulates_identically(tmp_path):
    """The exported stage *behaves* like the one it was exported from.

    This is the verification that matters: array equality shows the numbers match, but only
    simulating both stages shows that the export reproduces the same physics. It is also less
    brittle than comparing every model array, because it is unaffected by rebuilt acceleration
    structures and by shape orderings that differ without changing behavior.

    The joint carries an offset frame so that the rollout is actually sensitive to the joint
    geometry -- with everything at the origin the trajectory cannot distinguish a correct export
    from one that drops joint frames entirely.
    """
    source = tmp_path / "rollout_source.usda"
    _author_source_stage(str(source), joint_local_pos0=(0.37, -0.11, 0.05))
    m1, stage_info = _load(source)
    out = tmp_path / "exported.usda"
    export_model_to_usd(m1, str(out), stage_info=stage_info)
    m2, _ = _load(out)

    traj_1, traj_2 = _rollout(m1), _rollout(m2)
    assert traj_1.shape == traj_2.shape, f"trajectory shape {traj_1.shape} != {traj_2.shape}"
    assert np.abs(traj_1).sum() > 0.0, "fixture never moved; test would be vacuous"

    position_error = np.abs(traj_1[..., :3] - traj_2[..., :3]).max()
    # A quaternion and its negation are the same rotation, so compare |dot| against 1.
    rotation_error = (1.0 - np.abs((traj_1[..., 3:7] * traj_2[..., 3:7]).sum(-1))).max()
    assert position_error < 1e-5, f"body positions diverged after export: max |delta| = {position_error:.3e}"
    assert rotation_error < 1e-5, f"body rotations diverged after export: max error = {rotation_error:.3e}"


def _first_articulated_joint(model) -> int:
    """Return the index of the first joint that is not a free joint."""
    free = int(newton.JointType.FREE)
    for index, joint_type in enumerate(model.joint_type.numpy()):
        if int(joint_type) != free:
            return index
    raise AssertionError("model has no articulated joint")


def test_export_captures_post_load_overrides(source_stage, tmp_path):
    """The export reflects values overridden after load, not the values in the source file.

    This is the property the feature exists for. Isaac Lab applies most configuration by writing
    into the model after the stage is parsed (``write_joint_stiffness_to_sim`` and friends), so an
    exporter that re-derived its output from the source USD would reproduce the *asset* rather than
    what is being simulated -- and would pass every other test here, because all of them start from
    an unmodified load.
    """
    m1, stage_info = _load(source_stage)

    # Only DOFs backed by an actual joint can be exported: a free joint is expressed in USD by the
    # absence of a joint prim, so its DOFs have nowhere to carry drive gains or armature.
    dof_start = int(m1.joint_qd_start.numpy()[_first_articulated_joint(m1)])

    # Stand in for Isaac Lab's config overrides: write distinctive values into the model, exactly as
    # the asset classes do. The values are chosen not to coincide with the fixture's.
    target_ke = m1.joint_target_ke.numpy().copy()
    target_ke[dof_start] = 4242.0
    armature = m1.joint_armature.numpy().copy()
    armature[dof_start] = 0.0731
    overrides = {
        "joint_target_ke": target_ke,
        "joint_armature": armature,
        "body_mass": m1.body_mass.numpy() * 0.0 + 9.875,
    }
    for name, value in overrides.items():
        getattr(m1, name).assign(value)

    out = tmp_path / "exported.usda"
    export_model_to_usd(m1, str(out), stage_info=stage_info)
    m2, _ = _load(out)

    mismatched = []
    for name, expected in overrides.items():
        actual = getattr(m2, name).numpy()
        if not np.allclose(actual, expected, rtol=1e-4, atol=1e-6):
            mismatched.append(f"{name}: expected {expected}, exported {actual}")
    assert not mismatched, "overrides applied after load were not captured by the export:\n" + "\n".join(mismatched)


def test_exported_stage_preserves_source_prim_paths(source_stage, tmp_path):
    """Bodies are exported at their original prim paths, not synthesized ones."""
    m1, stage_info = _load(source_stage)
    out = tmp_path / "exported.usda"
    export_model_to_usd(m1, str(out), stage_info=stage_info)

    exported = Usd.Stage.Open(str(out))
    for path in stage_info["path_body_map"]:
        assert exported.GetPrimAtPath(path).IsValid(), f"source prim path {path} missing from export"


def _replicate(source_path: str, offsets: tuple[float, ...]) -> tuple[newton.Model, dict]:
    """Clone ``source_path`` into one world per offset, as the Isaac Lab cloner does.

    Mirrors ``isaaclab_newton.cloner.newton_clone_utils``, which replicates a prototype builder
    across worlds. Every world shares the source asset's prim paths, so only the world offsets
    distinguish them.
    """
    prototype = newton.ModelBuilder()
    stage_info = prototype.add_usd(str(source_path))
    xforms = np.zeros((len(offsets), 7), dtype=np.float32)
    xforms[:, 6] = 1.0
    xforms[:, 0] = offsets
    builder = newton.ModelBuilder()
    builder.replicate(prototype, len(offsets), xforms=xforms)
    return builder.finalize(), stage_info


def test_each_cloned_world_round_trips_its_own_state(source_stage, tmp_path):
    """A cloned model exports the selected world, carrying that world's state and no other's.

    Isaac Lab always clones the scene across environments, so the model holds several worlds that
    share one set of prim paths. Exporting must select one rather than emit the source paths while
    reading whichever indices happen to come first.
    """
    offsets = (0.0, 2.0, 4.0)
    model, stage_info = _replicate(source_stage, offsets)
    assert model.world_count == len(offsets), "fixture failed to build one world per offset"

    world_starts = model.body_world_start.numpy()
    for world, offset in enumerate(offsets):
        out = tmp_path / f"world_{world}.usda"
        export_model_to_usd(model, str(out), stage_info=stage_info, world=world)

        reimported, _ = _load(out)
        assert reimported.body_count == len(stage_info["path_body_map"]), (
            f"world {world} exported {reimported.body_count} bodies, expected"
            f" {len(stage_info['path_body_map'])}: the export dropped or duplicated bodies"
        )
        expected = model.body_q.numpy()[world_starts[world]]
        np.testing.assert_allclose(reimported.body_q.numpy()[0], expected, rtol=1e-5, atol=1e-6)
        assert abs(reimported.body_q.numpy()[0][0] - offset) < 1e-5, (
            f"world {world} carries the placement of a different world"
        )


def test_export_rejects_a_world_outside_the_model(source_stage, tmp_path):
    """Selecting a world the model does not have fails instead of exporting an empty stage."""
    model, stage_info = _replicate(source_stage, (0.0, 2.0))
    with pytest.raises(ValueError, match="out of range"):
        export_model_to_usd(model, str(tmp_path / "out.usda"), stage_info=stage_info, world=2)


def test_export_rejects_provenance_that_misses_entities(source_stage, tmp_path):
    """Prim paths covering fewer bodies than the world holds fail rather than exporting a subset.

    Without this the export silently describes part of the scene, and the shortfall is invisible
    until the reimported stage is compared against the model it came from.
    """
    model, stage_info = _load(source_stage)
    partial = dict(stage_info)
    partial["path_body_map"] = dict(list(stage_info["path_body_map"].items())[:1])
    with pytest.raises(ValueError, match="silently drop"):
        export_model_to_usd(model, str(tmp_path / "out.usda"), stage_info=partial)


def test_ground_plane_round_trips(tmp_path):
    """A finite ground plane survives export and reimport with its extent and orientation.

    Every Isaac Lab task spawns one, so a plane the exporter cannot author would fail the export of
    every real scene rather than an exotic one.
    """
    source = tmp_path / "ground.usda"
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    plane = UsdGeom.Plane.Define(stage, "/World/ground")
    plane.GetAxisAttr().Set(UsdGeom.Tokens.z)
    plane.GetWidthAttr().Set(100.0)
    plane.GetLengthAttr().Set(60.0)
    UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
    stage.GetRootLayer().Save()

    m1, stage_info = _load(source)
    assert m1.shape_count == 1 and newton.GeoType(int(m1.shape_type.numpy()[0])) is newton.GeoType.PLANE, (
        "fixture did not import as a plane; the test would be vacuous"
    )
    out = tmp_path / "exported.usda"
    export_model_to_usd(m1, str(out), stage_info=stage_info)

    m2, _ = _load(out)
    assert m2.shape_count == 1
    assert newton.GeoType(int(m2.shape_type.numpy()[0])) is newton.GeoType.PLANE
    np.testing.assert_allclose(m2.shape_scale.numpy()[0], m1.shape_scale.numpy()[0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(m2.shape_transform.numpy()[0], m1.shape_transform.numpy()[0], rtol=1e-6, atol=1e-6)


def test_sites_are_not_counted_against_the_asset(source_stage, tmp_path):
    """A cloner-added site has no prim and must not trip the provenance guard.

    Isaac Lab adds one site per sensor frame after importing the asset. They are shapes in the model
    but describe nothing USD can hold, so the export must skip them rather than refuse the scene.
    """
    builder = newton.ModelBuilder()
    stage_info = builder.add_usd(str(source_stage))
    builder.add_site(body=0, label="imu_frame")
    model = builder.finalize()
    assert model.shape_count == len(_canonical_paths_count(stage_info)) + 1, "fixture failed to add a site"

    out = tmp_path / "exported.usda"
    export_model_to_usd(model, str(out), stage_info=stage_info)
    reimported, _ = _load(out)
    assert reimported.shape_count == model.shape_count - 1, "the site was exported as geometry"


def _canonical_paths_count(stage_info) -> set[int]:
    return set(stage_info["path_shape_map"].values())


def test_loop_closing_joint_round_trips(tmp_path):
    """A joint marked ``excludeFromArticulation`` survives export, so a closed loop reimports.

    A four-bar linkage closes its loop with a joint that USD keeps out of the articulation tree and
    Newton imports with no articulation membership. Authoring it as an ordinary joint makes the
    reimport reject the model with a cycle.
    """
    source = tmp_path / "loop.usda"
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = "/World"
    UsdPhysics.ArticulationRootAPI.Apply(UsdGeom.Xform.Define(stage, root).GetPrim())
    UsdPhysics.Scene.Define(stage, f"{root}/physicsScene")
    for name, x in (("A", 0.0), ("B", 1.0), ("C", 2.0)):
        body = UsdGeom.Xform.Define(stage, f"{root}/{name}")
        body.AddTranslateOp().Set(Gf.Vec3d(x, 0.0, 0.0))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdPhysics.MassAPI.Apply(body.GetPrim()).GetMassAttr().Set(1.0)
        cube = UsdGeom.Cube.Define(stage, f"{root}/{name}/col")
        cube.GetSizeAttr().Set(0.1)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdPhysics.FixedJoint.Define(stage, f"{root}/anchor").GetBody1Rel().SetTargets([f"{root}/A"])
    for name, b0, b1 in (("ab", "A", "B"), ("bc", "B", "C"), ("ca", "C", "A")):
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"{root}/{name}")
        joint.GetBody0Rel().SetTargets([f"{root}/{b0}"])
        joint.GetBody1Rel().SetTargets([f"{root}/{b1}"])
        joint.GetAxisAttr().Set("Z")
    stage.GetPrimAtPath(f"{root}/ca").GetAttribute("physics:excludeFromArticulation").Set(True)
    stage.GetRootLayer().Save()

    m1, stage_info = _load(source)
    assert int(m1.joint_articulation.numpy()[-1]) < 0, (
        "fixture's loop joint was not excluded; the test would be vacuous"
    )
    out = tmp_path / "exported.usda"
    export_model_to_usd(m1, str(out), stage_info=stage_info)

    m2, _ = _load(out)  # would raise "Joint graph contains a cycle" without the attribute
    assert m2.joint_count == m1.joint_count
    np.testing.assert_array_equal(m2.joint_articulation.numpy(), m1.joint_articulation.numpy())
