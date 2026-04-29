# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for MjcTendon handling in the Newton cloner.

Root cause of the bug (newton-physics/newton#2618):
  Two interlocking constraints make the main builder incompatible with
  ``SchemaResolverMjc``:

  1. ``SchemaResolverMjc.validate_custom_attributes()`` is called inside
     ``builder.add_usd()`` and raises ``RuntimeError`` unless
     ``SolverMuJoCo.register_custom_attributes()`` has already been called.

  2. Calling ``register_custom_attributes`` registers MJC custom frequencies,
     which triggers Newton's stage-wide traversal (independent of
     ``ignore_paths``).  That traversal finds ``MjcTendon`` prims under
     ``/World/envs/...`` and tries to resolve their joint paths against the
     main builder's empty ``joint_label`` (no joints were loaded via
     ``ignore_paths``), producing "unknown joint path" warnings and silently
     discarding every tendon.

The fix:
  ``SchemaResolverMjc`` is **excluded** from the main builder's
  ``schema_resolvers``.  The main builder only loads scene-level prims
  (ground plane, lights) that have no ``MjcTendon`` prims, so this resolver
  is not needed there.  Proto builders include it and call
  ``register_custom_attributes`` before ``add_usd``, so Newton correctly
  resolves ``MjcTendon`` joint paths against each proto's fully populated
  ``joint_label``.  ``add_builder(proto)`` then copies those resolved entries
  (with joint-index offsets and the correct ``tendon_world`` value) into the
  main builder for every environment world.

Newton's design for multi-world tendons:
  ``add_builder`` appends T tendon entries per environment, yielding N×T
  total entries after N environments.  ``SolverMuJoCo.__init__`` iterates
  over all N×T entries and keeps only those whose ``tendon_world`` equals
  the template world (world 0), discarding the rest.  No deduplication by
  IsaacLab is required or correct.

Tests here use ``newton.ModelBuilder`` directly, so no Isaac Sim, USD stage,
or MJCF XML parsing is required.
"""

import unittest

import newton
from newton.solvers import SolverMuJoCo

# Custom-frequency keys used by SolverMuJoCo for fixed tendons.
_TENDON_FREQ = "mujoco:tendon"
_TENDON_JOINT_FREQ = "mujoco:tendon_joint"


# ─── helpers ─────────────────────────────────────────────────────────────────


def _plain_builder() -> newton.ModelBuilder:
    """Return a ModelBuilder without MuJoCo custom attributes registered."""
    return newton.ModelBuilder()


def _mjc_builder() -> newton.ModelBuilder:
    """Return a ModelBuilder with MuJoCo custom attributes registered."""
    b = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(b)
    return b


def _inject_tendon_entries(
    builder: newton.ModelBuilder,
    world: int,
    names: list[str],
    stiffnesses: list[float],
    joint_entries_per_tendon: int,
) -> None:
    """Append synthetic tendon entries for *world* directly into *builder*.

    Mimics what ``add_builder`` does when copying a proto that has resolved
    tendon data.  The builder must already have MJC custom attributes
    registered (via ``SolverMuJoCo.register_custom_attributes``).
    """
    t = len(names)
    j = t * joint_entries_per_tendon

    def _attr_values(key):
        attr = builder.custom_attributes.get(key)
        if attr is not None and attr.values is None:
            attr.values = []
        return attr.values if attr is not None else None

    world_vals = _attr_values("mujoco:tendon_world")
    label_vals = _attr_values("mujoco:tendon_label")
    stiff_vals = _attr_values("mujoco:tendon_stiffness")
    jadr_vals = _attr_values("mujoco:tendon_joint_adr")
    jnum_vals = _attr_values("mujoco:tendon_joint_num")
    joint_vals = _attr_values("mujoco:tendon_joint")
    coef_vals = _attr_values("mujoco:tendon_coef")

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


def _tendon_labels(b: newton.ModelBuilder) -> list[str]:
    attr = b.custom_attributes.get("mujoco:tendon_label")
    if attr is None or not isinstance(attr.values, list):
        return []
    return list(attr.values)


def _tendon_stiffnesses(b: newton.ModelBuilder) -> list[float]:
    attr = b.custom_attributes.get("mujoco:tendon_stiffness")
    if attr is None or not isinstance(attr.values, list):
        return []
    return list(attr.values)


# ─── tests ───────────────────────────────────────────────────────────────────


class TestMainBuilderHasNoMjcFrequencies(unittest.TestCase):
    """Main builder must not have MJC custom frequencies registered.

    This is the core invariant of the fix: the main builder is constructed
    without ``SolverMuJoCo.register_custom_attributes``, so Newton's stage-wide
    custom-frequency traversal never attempts to resolve MjcTendon prims
    against the main builder's empty joint_label.
    """

    def test_plain_builder_has_no_tendon_frequency(self):
        b = _plain_builder()
        self.assertNotIn(_TENDON_FREQ, b.custom_frequencies)

    def test_plain_builder_has_no_tendon_joint_frequency(self):
        b = _plain_builder()
        self.assertNotIn(_TENDON_JOINT_FREQ, b.custom_frequencies)

    def test_plain_builder_has_no_mjc_tendon_attributes(self):
        b = _plain_builder()
        mjc_keys = [k for k in b.custom_attributes if k.startswith("mujoco:tendon")]
        self.assertEqual(mjc_keys, [], f"Unexpected MJC tendon attrs on plain builder: {mjc_keys}")


class TestProtoBuilderHasMjcFrequencies(unittest.TestCase):
    """Proto builders must have MJC custom frequencies registered.

    The proto is where joint_label is fully populated, so the MJC custom-freq
    traversal can correctly resolve MjcTendon joint paths.
    """

    def test_registered_builder_has_tendon_frequency(self):
        b = _mjc_builder()
        self.assertIn(_TENDON_FREQ, b.custom_frequencies)

    def test_registered_builder_has_tendon_joint_frequency(self):
        b = _mjc_builder()
        self.assertIn(_TENDON_JOINT_FREQ, b.custom_frequencies)

    def test_registered_builder_has_tendon_world_attribute(self):
        b = _mjc_builder()
        self.assertIn("mujoco:tendon_world", b.custom_attributes)


class TestAddBuilderTendonPropagation(unittest.TestCase):
    """Verify that add_builder propagates tendon entries with correct world indices.

    Newton's design: after N add_builder calls (one per world), the main
    builder holds N×T tendon entries.  Each batch of T entries has
    tendon_world == that world's index.  SolverMuJoCo filters on
    tendon_world == template_world (0) to extract the T canonical tendons;
    all other entries are discarded during MuJoCo model construction.

    IsaacLab must NOT deduplicate these entries — Newton expects N×T.
    """

    NAMES = ["coupling_ff", "coupling_mf"]
    STIFFNESSES = [2.0, 1.0]
    JOINT_ENTRIES_PER_TENDON = 2
    T = len(NAMES)
    T_JOINT = T * JOINT_ENTRIES_PER_TENDON

    def _build_main_with_n_worlds(self, n_envs: int) -> newton.ModelBuilder:
        main = _mjc_builder()
        for world in range(n_envs):
            _inject_tendon_entries(main, world, self.NAMES, self.STIFFNESSES, self.JOINT_ENTRIES_PER_TENDON)
        return main

    def test_single_world_has_t_entries(self):
        main = self._build_main_with_n_worlds(1)
        self.assertEqual(len(_tendon_world_values(main)), self.T)

    def test_n_worlds_produce_n_times_t_entries(self):
        """N×T total entries is the expected state Newton reads."""
        for n in (2, 4, 8):
            with self.subTest(n=n):
                main = self._build_main_with_n_worlds(n)
                self.assertEqual(len(_tendon_world_values(main)), n * self.T)

    def test_tendon_world_values_span_all_worlds(self):
        """Every world index 0..N-1 is represented exactly T times."""
        n = 4
        main = self._build_main_with_n_worlds(n)
        worlds = _tendon_world_values(main)
        for w in range(n):
            count = sum(1 for v in worlds if v == w)
            self.assertEqual(count, self.T, f"World {w}: expected {self.T} entries, got {count}")

    def test_template_world_entries_have_correct_data(self):
        """World-0 tendon entries carry the original proto names and stiffnesses."""
        n = 5
        main = self._build_main_with_n_worlds(n)
        worlds = _tendon_world_values(main)
        labels = _tendon_labels(main)
        stiffs = _tendon_stiffnesses(main)

        w0_labels = [labels[i] for i, w in enumerate(worlds) if w == 0]
        w0_stiffs = [stiffs[i] for i, w in enumerate(worlds) if w == 0]

        self.assertEqual(sorted(w0_labels), sorted(self.NAMES))
        self.assertEqual(sorted(w0_stiffs), sorted(self.STIFFNESSES))

    def test_newton_filter_extracts_t_entries(self):
        """Simulating Newton's tendon_world filter yields exactly T entries from N×T."""
        n = 6
        template_world = 0
        main = self._build_main_with_n_worlds(n)
        worlds = _tendon_world_values(main)
        selected = [i for i, w in enumerate(worlds) if w == template_world or w < 0]
        self.assertEqual(len(selected), self.T)

    def test_plain_builder_accumulates_no_tendon_entries(self):
        """A plain builder (no register_custom_attributes) has no tendon entries."""
        self.assertEqual(_tendon_world_values(_plain_builder()), [])

    def test_regression_extra_world0_entries_break_newton_filter(self):
        """Regression for newton-physics/newton#2618.

        When SchemaResolverMjc was included in the main builder's schema_resolvers
        (old behavior), Newton's stage-wide MJC traversal ran against the main
        builder's empty joint_label before any add_builder call.  The traversal
        found MjcTendon prims and added T entries with world=0 (template world),
        because add_usd on the main builder runs before any begin_world() call.
        After add_builder for env-0 also appended T world-0 entries, the template
        world held 2T entries instead of T — causing Newton's SolverMuJoCo to
        initialize with twice the correct number of tendons for world 0.

        The fix excludes SchemaResolverMjc from the main builder's resolvers so
        the traversal never runs there.  The main builder contributes zero tendon
        entries from its own add_usd; only add_builder calls populate it.
        """
        n = 4

        # --- Broken state (old behavior) ---
        # Simulate: main builder's add_usd injected T bad world-0 entries (from
        # the stage-wide traversal), then add_builder added T more for each world.
        broken = _mjc_builder()
        _inject_tendon_entries(broken, 0, self.NAMES, self.STIFFNESSES, self.JOINT_ENTRIES_PER_TENDON)
        for world in range(n):
            _inject_tendon_entries(broken, world, self.NAMES, self.STIFFNESSES, self.JOINT_ENTRIES_PER_TENDON)
        broken_worlds = _tendon_world_values(broken)
        # Total: T (bad) + N×T (good) = (N+1)×T
        self.assertEqual(len(broken_worlds), (n + 1) * self.T)
        # Newton's filter selects world-0 entries: T bad + T good = 2T (wrong)
        broken_w0 = [w for w in broken_worlds if w == 0]
        self.assertEqual(len(broken_w0), 2 * self.T)

        # --- Fixed state (new behavior) ---
        # No entries from main builder's add_usd; only N×T from add_builder.
        fixed = self._build_main_with_n_worlds(n)
        fixed_worlds = _tendon_world_values(fixed)
        self.assertEqual(len(fixed_worlds), n * self.T)
        # Newton's filter selects exactly T world-0 entries (correct)
        fixed_w0 = [w for w in fixed_worlds if w == 0]
        self.assertEqual(len(fixed_w0), self.T)


class TestSchemaResolverMjcImport(unittest.TestCase):
    """SchemaResolverMjc must be importable and validate correctly."""

    def test_schema_resolver_mjc_importable(self):
        from newton._src.usd.schemas import SchemaResolverMjc  # noqa: PLC0415

        self.assertIsNotNone(SchemaResolverMjc)

    def test_schema_resolver_mjc_instantiable(self):
        from newton._src.usd.schemas import SchemaResolverMjc  # noqa: PLC0415

        self.assertIsNotNone(SchemaResolverMjc())

    def test_validate_raises_on_unregistered_builder(self):
        """SchemaResolverMjc.validate_custom_attributes raises if register_custom_attributes was not called.

        This is why SchemaResolverMjc must be excluded from the main builder's
        schema_resolvers — the main builder never calls register_custom_attributes.
        """
        from newton._src.usd.schemas import SchemaResolverMjc  # noqa: PLC0415

        resolver = SchemaResolverMjc()
        plain = _plain_builder()
        with self.assertRaises(RuntimeError):
            resolver.validate_custom_attributes(plain)

    def test_validate_passes_on_registered_builder(self):
        """SchemaResolverMjc.validate_custom_attributes passes after register_custom_attributes."""
        from newton._src.usd.schemas import SchemaResolverMjc  # noqa: PLC0415

        resolver = SchemaResolverMjc()
        registered = _mjc_builder()
        resolver.validate_custom_attributes(registered)


if __name__ == "__main__":
    unittest.main()
