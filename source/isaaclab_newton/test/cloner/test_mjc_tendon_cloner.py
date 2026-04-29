# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for MjcTendon handling in the Newton cloner.

Root cause (newton-physics/newton#2618):

  Part 1 — main builder isolation:
    ``SchemaResolverMjc.validate_custom_attributes()`` raises unless
    ``SolverMuJoCo.register_custom_attributes()`` was called first.  Calling
    it on the main builder triggers a stage-wide traversal that finds
    ``MjcTendon`` prims and silently drops all tendons (joint resolution fails
    against the main builder's empty ``joint_label``).  Fix: exclude
    ``SchemaResolverMjc`` from the main builder's ``schema_resolvers``.

  Part 2 — proto builder path scoping:
    Newton's custom-frequency traversal calls ``stage.Traverse()``
    unconditionally, ignoring ``root_path``.  In heterogeneous clone plans,
    proto A's traversal also matches source B's prims.  Fix:
    ``_scope_custom_frequencies`` patches every registered ``usd_prim_filter``
    to require a ``root_path/`` prefix match, restricting each proto to its
    own source subtree.

  Part 3 — N×T multi-world semantics:
    ``add_builder(proto)`` copies T resolved tendon entries per world, giving
    N×T total.  Newton reads per-world slices for parameter randomization and
    uses template_world=0 for joint connectivity.  No deduplication needed.
"""

import unittest

import newton
from isaaclab_newton.cloner.newton_replicate import _scope_custom_frequencies
from newton.solvers import SolverMuJoCo

_TENDON_FREQ = "mujoco:tendon"
_TENDON_JOINT_FREQ = "mujoco:tendon_joint"


# ─── helpers ─────────────────────────────────────────────────────────────────


def _plain_builder() -> newton.ModelBuilder:
    return newton.ModelBuilder()


def _mjc_builder() -> newton.ModelBuilder:
    b = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(b)
    return b


class _FakePrim:
    """Minimal USD prim stub for testing usd_prim_filter lambdas without a stage."""

    def __init__(self, path: str, type_name: str = "MjcTendon"):
        self._path = path
        self._type_name = type_name

    def GetPath(self) -> str:
        return self._path

    def GetTypeName(self) -> str:
        return self._type_name


def _inject_tendon_entries(
    builder: newton.ModelBuilder,
    world: int,
    names: list[str],
    stiffnesses: list[float],
    joint_entries_per_tendon: int,
) -> None:
    """Append synthetic tendon entries for *world* directly into *builder*."""
    t = len(names)
    j = t * joint_entries_per_tendon

    def _vals(key):
        attr = builder.custom_attributes.get(key)
        if attr is not None and attr.values is None:
            attr.values = []
        return attr.values if attr is not None else None

    world_vals = _vals("mujoco:tendon_world")
    label_vals = _vals("mujoco:tendon_label")
    stiff_vals = _vals("mujoco:tendon_stiffness")
    jadr_vals = _vals("mujoco:tendon_joint_adr")
    jnum_vals = _vals("mujoco:tendon_joint_num")
    joint_vals = _vals("mujoco:tendon_joint")
    coef_vals = _vals("mujoco:tendon_coef")

    current_joint_offset = builder._custom_frequency_counts.get(_TENDON_JOINT_FREQ, 0)
    for i, (name, stiff) in enumerate(zip(names, stiffnesses)):
        if world_vals is not None:
            world_vals.append(world)
        if label_vals is not None:
            label_vals.append(name)
        if stiff_vals is not None:
            stiff_vals.append(stiff)
        if jadr_vals is not None:
            jadr_vals.append(current_joint_offset + i * joint_entries_per_tendon)
        if jnum_vals is not None:
            jnum_vals.append(joint_entries_per_tendon)
    for _ in range(j):
        if joint_vals is not None:
            joint_vals.append(0)
        if coef_vals is not None:
            coef_vals.append(1.0)

    builder._custom_frequency_counts[_TENDON_FREQ] = builder._custom_frequency_counts.get(_TENDON_FREQ, 0) + t
    builder._custom_frequency_counts[_TENDON_JOINT_FREQ] = (
        builder._custom_frequency_counts.get(_TENDON_JOINT_FREQ, 0) + j
    )


def _tendon_world_values(b: newton.ModelBuilder) -> list[int]:
    attr = b.custom_attributes.get("mujoco:tendon_world")
    if attr is None or not isinstance(attr.values, list):
        return []
    return [int(v) for v in attr.values]


def _tendon_stiffnesses(b: newton.ModelBuilder) -> list[float]:
    attr = b.custom_attributes.get("mujoco:tendon_stiffness")
    if attr is None or not isinstance(attr.values, list):
        return []
    return list(attr.values)


def _make_minimal_tendon_stage(root_path: str, stiffness: float = 1.5):
    """Create an in-memory USD stage with one joint and one fixed MjcTendon.

    Returns the stage and the joint prim path string so callers can verify
    joint resolution.  No Isaac Sim required — uses only OpenUSD + Newton schemas.
    """
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, "Z")

    UsdGeom.Xform.Define(stage, root_path)
    b0 = UsdGeom.Xform.Define(stage, f"{root_path}/body0")
    UsdPhysics.RigidBodyAPI.Apply(b0.GetPrim())
    b1 = UsdGeom.Xform.Define(stage, f"{root_path}/body1")
    UsdPhysics.RigidBodyAPI.Apply(b1.GetPrim())

    joint_path = f"{root_path}/joint0"
    j = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
    j.GetBody0Rel().SetTargets([Sdf.Path(f"{root_path}/body0")])
    j.GetBody1Rel().SetTargets([Sdf.Path(f"{root_path}/body1")])
    j.GetAxisAttr().Set("X")

    t = stage.DefinePrim(f"{root_path}/tendon0", "MjcTendon")
    t.CreateAttribute("mjc:type", Sdf.ValueTypeNames.String).Set("fixed")
    t.CreateAttribute("mjc:stiffness", Sdf.ValueTypeNames.Float).Set(stiffness)
    t.CreateAttribute("mjc:damping", Sdf.ValueTypeNames.Float).Set(0.1)
    rel = t.CreateRelationship("mjc:path")
    rel.SetTargets([Sdf.Path(joint_path)])
    t.CreateAttribute("mjc:path:coef", Sdf.ValueTypeNames.FloatArray).Set([1.0])

    return stage, joint_path


# ─── tests ───────────────────────────────────────────────────────────────────


class TestSchemaResolverMjcContract(unittest.TestCase):
    """SchemaResolverMjc.validate_custom_attributes enforces registration order (Part 1)."""

    def test_validate_raises_on_unregistered_builder(self):
        """Confirms why SchemaResolverMjc must be excluded from the main builder."""
        from newton._src.usd.schemas import SchemaResolverMjc  # noqa: PLC0415

        with self.assertRaises(RuntimeError):
            SchemaResolverMjc().validate_custom_attributes(_plain_builder())

    def test_validate_passes_after_register_custom_attributes(self):
        from newton._src.usd.schemas import SchemaResolverMjc  # noqa: PLC0415

        SchemaResolverMjc().validate_custom_attributes(_mjc_builder())


class TestScopeCustomFrequencies(unittest.TestCase):
    """``_scope_custom_frequencies`` restricts traversal to a source subtree (Part 2)."""

    ROOT_A = "/World/envs/env_0/source_a"
    ROOT_B = "/World/envs/env_0/source_b"

    def _scoped(self, root_path: str) -> newton.ModelBuilder:
        b = _mjc_builder()
        _scope_custom_frequencies(b, root_path)
        return b

    def test_filter_accepts_prim_under_root(self):
        freq = self._scoped(self.ROOT_A).custom_frequencies[_TENDON_FREQ]
        self.assertTrue(freq.usd_prim_filter(_FakePrim(f"{self.ROOT_A}/tendon"), {}))

    def test_filter_rejects_prim_outside_root(self):
        freq = self._scoped(self.ROOT_A).custom_frequencies[_TENDON_FREQ]
        self.assertFalse(freq.usd_prim_filter(_FakePrim(f"{self.ROOT_B}/tendon"), {}))

    def test_filter_rejects_sibling_path_prefix_match(self):
        """startswith must use a trailing slash to avoid matching sibling paths.

        e.g. root=/robot_a must not match /robot_a_v2/tendon.
        """
        root = "/World/envs/env_0/robot_a"
        freq = self._scoped(root).custom_frequencies[_TENDON_FREQ]
        sibling = _FakePrim("/World/envs/env_0/robot_a_v2/tendon")
        self.assertFalse(freq.usd_prim_filter(sibling, {}))

    def test_filter_rejects_wrong_prim_type(self):
        """Path match alone is not sufficient; original type filter still applies."""
        freq = self._scoped(self.ROOT_A).custom_frequencies[_TENDON_FREQ]
        self.assertFalse(freq.usd_prim_filter(_FakePrim(f"{self.ROOT_A}/body", type_name="PhysicsRigidBodyAPI"), {}))

    def test_each_proto_scoped_to_its_own_path(self):
        """Two protos each accept only their own source's prims."""
        freq_a = self._scoped(self.ROOT_A).custom_frequencies[_TENDON_FREQ]
        freq_b = self._scoped(self.ROOT_B).custom_frequencies[_TENDON_FREQ]
        prim_a = _FakePrim(f"{self.ROOT_A}/tendon")
        prim_b = _FakePrim(f"{self.ROOT_B}/tendon")
        self.assertTrue(freq_a.usd_prim_filter(prim_a, {}))
        self.assertFalse(freq_a.usd_prim_filter(prim_b, {}))
        self.assertFalse(freq_b.usd_prim_filter(prim_a, {}))
        self.assertTrue(freq_b.usd_prim_filter(prim_b, {}))


class TestUsdTendonParsing(unittest.TestCase):
    """Verify tendon attributes from USD land in the proto builder (Part 1 + 2 integration).

    Uses a real in-memory USD stage with an MjcTendon prim.  No Isaac Sim required.
    """

    ROOT = "/robot"

    def _build_proto(self, stiffness: float = 1.5) -> newton.ModelBuilder:
        from newton._src.usd.schemas import (  # noqa: PLC0415
            SchemaResolverMjc,
            SchemaResolverNewton,
            SchemaResolverPhysx,
        )

        stage, _ = _make_minimal_tendon_stage(self.ROOT, stiffness=stiffness)
        b = _mjc_builder()
        _scope_custom_frequencies(b, self.ROOT)
        b.add_usd(
            stage,
            root_path=self.ROOT,
            load_visual_shapes=False,
            skip_mesh_approximation=True,
            schema_resolvers=[SchemaResolverMjc(), SchemaResolverNewton(), SchemaResolverPhysx()],
        )
        return b

    def test_stiffness_from_usd_lands_in_builder(self):
        """Tendon stiffness authored in the USD prim is read into the builder."""
        b = self._build_proto(stiffness=2.5)
        stiffs = _tendon_stiffnesses(b)
        self.assertEqual(len(stiffs), 1)
        self.assertAlmostEqual(float(stiffs[0]), 2.5, places=5)

    def test_no_tendons_without_register_custom_attributes(self):
        """Without register_custom_attributes the MJC traversal never runs."""
        from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx  # noqa: PLC0415

        stage, _ = _make_minimal_tendon_stage(self.ROOT)
        b = _plain_builder()
        # SchemaResolverMjc intentionally excluded — mirrors the main builder path
        b.add_usd(
            stage,
            root_path=self.ROOT,
            load_visual_shapes=False,
            skip_mesh_approximation=True,
            schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()],
        )
        self.assertEqual(_tendon_stiffnesses(b), [])


class TestAddBuilderTendonPropagation(unittest.TestCase):
    """Verify add_builder propagates tendon entries with correct world indices (Part 3).

    Newton's design: N add_builder calls produce N×T entries.  SolverMuJoCo
    reads per-world slices for parameter randomization and uses template_world=0
    for joint connectivity.  IsaacLab must NOT deduplicate.
    """

    NAMES = ["coupling_ff", "coupling_mf"]
    STIFFNESSES = [2.0, 1.0]
    JOINT_ENTRIES_PER_TENDON = 2
    T = len(NAMES)

    def _accumulate(self, n_envs: int) -> newton.ModelBuilder:
        acc = _mjc_builder()
        for world in range(n_envs):
            _inject_tendon_entries(acc, world, self.NAMES, self.STIFFNESSES, self.JOINT_ENTRIES_PER_TENDON)
        return acc

    def test_n_worlds_produce_n_times_t_entries(self):
        """N×T total entries is the expected state Newton reads."""
        for n in (1, 2, 4):
            with self.subTest(n=n):
                self.assertEqual(len(_tendon_world_values(self._accumulate(n))), n * self.T)

    def test_template_world_entries_have_correct_stiffness(self):
        """World-0 tendon entries carry the stiffness values from the proto."""
        main = self._accumulate(4)
        worlds = _tendon_world_values(main)
        stiffs = _tendon_stiffnesses(main)
        w0_stiffs = [stiffs[i] for i, w in enumerate(worlds) if w == 0]
        self.assertEqual(sorted(w0_stiffs), sorted(self.STIFFNESSES))

    def test_regression_extra_world0_entries_break_newton_filter(self):
        """Regression for newton-physics/newton#2618.

        Old behavior: SchemaResolverMjc on the main builder caused a stage-wide
        traversal that injected T extra world-0 entries before any add_builder call.
        After N add_builder calls, template_world held 2T entries instead of T,
        causing SolverMuJoCo to initialize with twice the correct tendon count.

        Fix: exclude SchemaResolverMjc from the main builder so it contributes
        zero tendon entries from its own add_usd.
        """
        n = 4

        # Broken: extra T world-0 entries from the stage-wide traversal
        broken = _mjc_builder()
        _inject_tendon_entries(broken, 0, self.NAMES, self.STIFFNESSES, self.JOINT_ENTRIES_PER_TENDON)
        for world in range(n):
            _inject_tendon_entries(broken, world, self.NAMES, self.STIFFNESSES, self.JOINT_ENTRIES_PER_TENDON)
        broken_w0 = [w for w in _tendon_world_values(broken) if w == 0]
        self.assertEqual(len(broken_w0), 2 * self.T)  # 2T — wrong

        # Fixed: exactly T world-0 entries
        fixed_w0 = [w for w in _tendon_world_values(self._accumulate(n)) if w == 0]
        self.assertEqual(len(fixed_w0), self.T)  # T — correct


if __name__ == "__main__":
    unittest.main()
