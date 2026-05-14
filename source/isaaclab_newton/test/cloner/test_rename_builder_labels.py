# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for ``_rename_builder_labels``.

Covers both passes of the rewrite:

  * Pass 1 — built-in label arrays (``body``, ``joint``, ``shape``,
    ``articulation``, ``constraint_mimic``, ``equality_constraint``).
  * Pass 2 — any string-typed custom-attribute column whose frequency declares a
    sibling ``references="world"`` companion (e.g. ``mujoco:tendon_label``).

The contract under test: every label whose row maps to a world in ``env_ids``
and whose value starts with the source root is rewritten to the destination
template's per-env path; everything else is left alone.
"""

import unittest

import newton
import torch
from isaaclab_newton.cloner.newton_replicate import _rename_builder_labels
from newton.solvers import SolverMuJoCo

_TENDON_FREQ = "mujoco:tendon"
_SRC = "/Sources/protoA"
_DST = "/World/envs/env_{}"


# ─── helpers ─────────────────────────────────────────────────────────────────


def _inject_builtins(builder: newton.ModelBuilder, types: tuple[str, ...], src_path: str, worlds: list[int]) -> None:
    """Append ``len(worlds)`` synthetic entries to each built-in ``*_label``/``*_world`` pair."""
    for t in types:
        labels = getattr(builder, f"{t}_label")
        worlds_arr = getattr(builder, f"{t}_world")
        for w in worlds:
            labels.append(f"{src_path}/{t}_{w}")
            worlds_arr.append(w)


def _inject_tendon_strings(builder: newton.ModelBuilder, src_path: str, worlds: list[int]) -> None:
    """Append synthetic ``mujoco:tendon_label`` + ``mujoco:tendon_world`` rows."""
    label_attr = builder.custom_attributes["mujoco:tendon_label"]
    world_attr = builder.custom_attributes["mujoco:tendon_world"]
    if label_attr.values is None:
        label_attr.values = []
    if world_attr.values is None:
        world_attr.values = []
    for w in worlds:
        label_attr.values.append(f"{src_path}/Tendon_{w}")
        world_attr.values.append(w)
    builder._custom_frequency_counts[_TENDON_FREQ] = builder._custom_frequency_counts.get(_TENDON_FREQ, 0) + len(worlds)


def _make_builder_with_entries(worlds: list[int]) -> newton.ModelBuilder:
    """Builder pre-populated with one row per world for every label class under test."""
    b = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(b)
    _inject_builtins(
        b, ("body", "joint", "shape", "articulation", "constraint_mimic", "equality_constraint"), _SRC, worlds
    )
    _inject_tendon_strings(b, _SRC, worlds)
    return b


# ─── tests ───────────────────────────────────────────────────────────────────


class TestRenameBuilderLabels(unittest.TestCase):
    """Both passes rewrite to the same per-env destination pattern."""

    def setUp(self):
        self.worlds = [0, 1, 2]
        self.env_ids = torch.tensor(self.worlds, dtype=torch.int32)
        self.mapping = torch.ones(1, len(self.worlds), dtype=torch.bool)

    def _rename(self, builder):
        _rename_builder_labels(builder, [_SRC], [_DST], self.env_ids, self.mapping)

    # Pass 1 ---------------------------------------------------------------

    def test_builtin_labels_rewritten_per_world(self):
        b = _make_builder_with_entries(self.worlds)
        self._rename(b)
        for t in ("body", "joint", "shape", "articulation", "constraint_mimic", "equality_constraint"):
            labels = getattr(b, f"{t}_label")
            worlds_arr = getattr(b, f"{t}_world")
            for k, w in enumerate(worlds_arr):
                self.assertEqual(
                    labels[k],
                    f"{_DST.format(int(w))}/{t}_{int(w)}",
                    msg=f"{t}_label[{k}] not rewritten correctly",
                )

    # Pass 2 ---------------------------------------------------------------

    def test_tendon_label_string_custom_attr_rewritten(self):
        b = _make_builder_with_entries(self.worlds)
        self._rename(b)
        labels = b.custom_attributes["mujoco:tendon_label"].values
        worlds_arr = b.custom_attributes["mujoco:tendon_world"].values
        for k, w in enumerate(worlds_arr):
            self.assertEqual(labels[k], f"{_DST.format(int(w))}/Tendon_{int(w)}")

    # Cross-pass consistency ----------------------------------------------

    def test_all_renamed_labels_share_the_per_env_root(self):
        """Every label written by either pass must live under ``/World/envs/env_<world>/``."""
        b = _make_builder_with_entries(self.worlds)
        self._rename(b)
        per_world = {int(w): _DST.format(int(w)) + "/" for w in self.env_ids.tolist()}
        for t in ("body", "joint", "shape", "articulation", "constraint_mimic", "equality_constraint"):
            for label, w in zip(getattr(b, f"{t}_label"), getattr(b, f"{t}_world")):
                self.assertTrue(label.startswith(per_world[int(w)]), msg=f"{t}: {label!r}")
        tendon_labels = b.custom_attributes["mujoco:tendon_label"].values
        tendon_worlds = b.custom_attributes["mujoco:tendon_world"].values
        for label, w in zip(tendon_labels, tendon_worlds):
            self.assertTrue(label.startswith(per_world[int(w)]), msg=f"tendon: {label!r}")

    # Guards ---------------------------------------------------------------

    def test_non_path_string_left_untouched(self):
        """Strings that don't start with ``src_path`` must pass through unchanged."""
        b = _make_builder_with_entries(self.worlds)
        # Inject one tendon row whose label is an opaque identifier, not a path.
        b.custom_attributes["mujoco:tendon_label"].values.append("named_tendon")
        b.custom_attributes["mujoco:tendon_world"].values.append(self.worlds[0])
        self._rename(b)
        self.assertEqual(b.custom_attributes["mujoco:tendon_label"].values[-1], "named_tendon")

    def test_world_outside_env_ids_left_untouched(self):
        """A row whose world is not in ``env_ids`` must keep its original label."""
        b = _make_builder_with_entries(self.worlds)
        # Inject one extra row tagged with a world id not present in env_ids.
        b.body_label.append(f"{_SRC}/body_99")
        b.body_world.append(99)
        self._rename(b)
        self.assertEqual(b.body_label[-1], f"{_SRC}/body_99")

    def test_sparse_env_ids(self):
        """Non-contiguous ``env_ids`` (e.g. [10, 20, 30]) must rewrite using the right per-env root."""
        worlds = [10, 20, 30]
        b = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(b)
        _inject_builtins(b, ("body",), _SRC, worlds)
        env_ids = torch.tensor(worlds, dtype=torch.int32)
        mapping = torch.ones(1, len(worlds), dtype=torch.bool)
        _rename_builder_labels(b, [_SRC], [_DST], env_ids, mapping)
        for k, w in enumerate(b.body_world):
            self.assertEqual(b.body_label[k], f"/World/envs/env_{int(w)}/body_{int(w)}")


class TestRenamePass2Generality(unittest.TestCase):
    """Pass 2 must generalize across coexisting frequencies and multiple string columns."""

    def setUp(self):
        self.worlds = [0, 1]
        self.env_ids = torch.tensor(self.worlds, dtype=torch.int32)
        self.mapping = torch.ones(1, len(self.worlds), dtype=torch.bool)

    def _register_synthetic_freq(self, builder, freq_name, world_attr_name, str_attr_names):
        """Register a ``syn:<freq_name>`` frequency with one world int column and N string columns."""
        freq = f"syn:{freq_name}"
        builder.add_custom_frequency(newton.ModelBuilder.CustomFrequency(name=freq_name, namespace="syn"))
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name=world_attr_name,
                frequency=freq,
                dtype=int,
                default=0,
                namespace="syn",
                references="world",
            )
        )
        for n in str_attr_names:
            builder.add_custom_attribute(
                newton.ModelBuilder.CustomAttribute(
                    name=n,
                    frequency=freq,
                    dtype=str,
                    default="",
                    namespace="syn",
                )
            )

    def _populate(self, builder, freq, world_attr_name, str_attr_names, worlds):
        wa = builder.custom_attributes[f"syn:{world_attr_name}"]
        if wa.values is None:
            wa.values = []
        for w in worlds:
            wa.values.append(w)
        for n in str_attr_names:
            sa = builder.custom_attributes[f"syn:{n}"]
            if sa.values is None:
                sa.values = []
            for w in worlds:
                sa.values.append(f"{_SRC}/{n}_{w}")
        builder._custom_frequency_counts[freq] = builder._custom_frequency_counts.get(freq, 0) + len(worlds)

    def test_two_coexisting_custom_frequencies(self):
        """Each registered ``references='world'`` companion must drive its own frequency's str columns."""
        b = newton.ModelBuilder()
        self._register_synthetic_freq(b, "freqA", "freqA_world", ["freqA_label"])
        self._register_synthetic_freq(b, "freqB", "freqB_world", ["freqB_label"])
        self._populate(b, "syn:freqA", "freqA_world", ["freqA_label"], self.worlds)
        self._populate(b, "syn:freqB", "freqB_world", ["freqB_label"], self.worlds)
        _rename_builder_labels(b, [_SRC], [_DST], self.env_ids, self.mapping)
        for n in ("freqA_label", "freqB_label"):
            wa = b.custom_attributes[f"syn:{n.split('_')[0]}_world"].values
            sa = b.custom_attributes[f"syn:{n}"].values
            for k, w in enumerate(wa):
                self.assertEqual(sa[k], f"/World/envs/env_{int(w)}/{n}_{int(w)}")

    def test_multiple_string_columns_at_one_frequency(self):
        """Two str columns sharing one frequency must both be rewritten using the shared world companion."""
        b = newton.ModelBuilder()
        self._register_synthetic_freq(b, "freqA", "freqA_world", ["freqA_label", "freqA_alt"])
        self._populate(b, "syn:freqA", "freqA_world", ["freqA_label", "freqA_alt"], self.worlds)
        _rename_builder_labels(b, [_SRC], [_DST], self.env_ids, self.mapping)
        wa = b.custom_attributes["syn:freqA_world"].values
        for n in ("freqA_label", "freqA_alt"):
            sa = b.custom_attributes[f"syn:{n}"].values
            for k, w in enumerate(wa):
                self.assertEqual(sa[k], f"/World/envs/env_{int(w)}/{n}_{int(w)}")

    def test_empty_values_pass_through(self):
        """A registered-but-empty string column must not crash the rename pass."""
        b = newton.ModelBuilder()
        self._register_synthetic_freq(b, "freqA", "freqA_world", ["freqA_label"])
        # values stay None (registered, never populated)
        _rename_builder_labels(b, [_SRC], [_DST], self.env_ids, self.mapping)
        # Fully populate after the no-op rename: ensures the early-return guard didn't corrupt state.
        self._populate(b, "syn:freqA", "freqA_world", ["freqA_label"], self.worlds)
        self.assertEqual(len(b.custom_attributes["syn:freqA_label"].values), len(self.worlds))


class TestRenameMultiSource(unittest.TestCase):
    """Multi-source handling must not cross-contaminate when source paths share a string prefix."""

    def test_prefix_overlap_does_not_cross_contaminate(self):
        """Sources whose paths share a string prefix and that both feed the same envs must not cross-rename.

        Common IL pattern: a robot proto and an object proto both feed every env. If the two source
        paths share a string prefix (``/Sources/protoA`` and ``/Sources/protoAB``), iter 0
        (``src=protoA``) sees the protoAB rows for the same world ids it owns and would over-match
        them under a non-boundary ``startswith``. The world-id guard alone does not catch this case
        because both sources contribute to the same set of worlds.
        """
        sources = ["/Sources/protoA", "/Sources/protoAB"]
        # 2 envs, both fed by both sources.
        env_ids = torch.tensor([0, 1], dtype=torch.int32)
        mapping = torch.tensor([[1, 1], [1, 1]], dtype=torch.bool)
        b = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(b)
        # One body row from each source per env: 4 rows total, world ids interleaved.
        b.body_label.extend(
            [
                f"{sources[0]}/body",  # row 0: protoA, world 0
                f"{sources[1]}/body",  # row 1: protoAB, world 0
                f"{sources[0]}/body",  # row 2: protoA, world 1
                f"{sources[1]}/body",  # row 3: protoAB, world 1
            ]
        )
        b.body_world.extend([0, 0, 1, 1])
        _rename_builder_labels(b, sources, ["/World/envs/env_{}", "/World/envs/env_{}"], env_ids, mapping)
        # Each row must end up under its own per-env root with the suffix preserved verbatim.
        # Without the "/" boundary on ``startswith``, iter 0 (src=protoA) would match rows 1 and 3
        # because ``/Sources/protoAB/body``.startswith(``/Sources/protoA``) is True, rewriting them
        # to ``/World/envs/env_<w>/B/body`` (wrong suffix).
        self.assertEqual(b.body_label[0], "/World/envs/env_0/body")
        self.assertEqual(b.body_label[1], "/World/envs/env_0/body")
        self.assertEqual(b.body_label[2], "/World/envs/env_1/body")
        self.assertEqual(b.body_label[3], "/World/envs/env_1/body")


if __name__ == "__main__":
    unittest.main()
