# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitless Newton-Warp backend of the ``VisualColorWriter`` contract.

Implements :class:`NewtonShapeColorWriter`, which applies per-environment diffuse
colors to a Newton model's visual shapes by writing rows of ``model.shape_color``.

This is the "live array, no notify" path: the Newton-Warp render context binds a
reference to ``model.shape_color`` at scene load, so writing the array between
frames recolors bodies on the very next render without any ``add_model_change`` /
``SolverNotifyFlags`` call.

It is dispatched from :class:`~isaaclab.envs.mdp.events.randomize_visual_color`,
mirroring :class:`~isaaclab.envs.mdp.events.randomize_rigid_body_material`'s
backend dispatch pattern.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.utils.visual_color import select_visual_color_targets

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


_DEFAULT_ENV_PATTERN = re.compile(r"/World/envs/env_(\d+)/")
_DEFAULT_COLLISION_TOKEN = "/collisions/"


class NewtonShapeColorWriter:
    """Apply per-environment diffuse colors to a Newton model's visual shapes.

    The writer is constructed **once** after the env is built. ``__init__`` walks
    ``model.shape_label`` to resolve, for each environment, the row indices into
    ``model.shape_color`` that correspond to the matched target prims. The
    cached map is used by :meth:`write_colors` to write one color per visual shape,
    honoring an ``env_ids`` subset (shapes whose env is not listed are left untouched).

    The default implementation uses the proven ``shape_color.numpy()`` ->
    ``shape_color.assign(...)`` round-trip. It is robust, deterministic, and
    adequate for typical Isaac Lab env counts; an optional Warp-scatter path can
    be added if profiling demands it.

    Args:
        env: The manager-based environment. The Newton model
            (``env.sim.physics_manager.get_model()``, which must expose ``shape_label`` and
            ``shape_color``) and the env count are read from it.
        mesh_prim_path: The (regex) prim-path pattern of the target visual meshes, resolved upstream in
            :class:`~isaaclab.envs.mdp.events.randomize_visual_color` (the same ``/visuals``-else-descendants
            pattern the OVRTX/Replicator backends consume). A shape is recolored when its label sits at or
            under a matched prim; collision shapes (``/collisions/``) are always skipped so the physical
            proxy geometry is not altered, only what the renderer actually rays into.
        env_pattern: Compiled regex with one capture group that yields the env
            index from a shape label. Defaults to
            ``r"/World/envs/env_(\\d+)/"`` (Isaac Lab's standard env prim
            layout).

    Raises:
        ValueError: If ``model`` lacks ``shape_label`` / ``shape_color``, or if
            no shapes can be mapped to any env.
    """

    @classmethod
    def pre_physics_ready_setup(cls, env, mesh_prim_path: str) -> None:
        """No-op: Newton reads ``model.shape_color`` live, so there is no pre-bake USD authoring to do.
        Present only to satisfy the common writer interface.
        """
        return

    def __init__(
        self,
        env: ManagerBasedEnv,
        mesh_prim_path: str,
        env_pattern: re.Pattern[str] | None = None,
    ) -> None:
        model = env.sim.physics_manager.get_model()
        num_envs = env.scene.num_envs
        if not hasattr(model, "shape_label") or not hasattr(model, "shape_color"):
            raise ValueError(
                "Model must expose `shape_label` (list of prim paths) and `shape_color` (wp.array of vec3f)."
            )
        shape_color = model.shape_color
        if shape_color is None:
            raise ValueError("Model.shape_color is None; the model must be finalized before constructing the writer.")
        if not isinstance(shape_color, wp.array):
            raise ValueError(f"Model.shape_color must be a warp.array (got {type(shape_color).__name__}).")

        self._model = model
        self._num_envs = int(num_envs)
        self._env_pattern = env_pattern or _DEFAULT_ENV_PATTERN

        # Resolve targets from the same pattern the term resolved (single source of truth shared with the
        # OVRTX/Replicator backends), so the ``/visuals``-else-descendants fallback is honored identically.
        # The pattern spans every env (``env_.*``), but Newton clones all envs into ``shape_label`` while the
        # USD stage may author only the source ``env_0`` -- so ``find_matching_prim_paths`` can return env_0
        # alone. We therefore match env-agnostically: strip each matched prim's ``env_<n>`` prefix to a suffix,
        # and a shape is a target when its label (prefix stripped) sits at or under one of those suffixes.
        # Collision proxies are never colored.
        target_suffixes: set[str] = set()
        for prim in sim_utils.find_matching_prim_paths(mesh_prim_path):
            prim_match = self._env_pattern.search(prim)
            if prim_match is not None:
                target_suffixes.add(prim[prim_match.end() :])

        self._per_env_rows: list[list[int]] = [[] for _ in range(self._num_envs)]
        self._matched_labels: list[tuple[int, int, str]] = []  # (env_id, shape_index, label) for diagnostics
        for shape_index, label in enumerate(model.shape_label):
            label_str = str(label)
            match = self._env_pattern.search(label_str)
            if match is None:
                continue
            env_id = int(match.group(1))
            if env_id < 0 or env_id >= self._num_envs:
                continue
            if _DEFAULT_COLLISION_TOKEN in label_str:
                continue
            suffix = label_str[match.end() :]
            if not any(suffix == target or suffix.startswith(target + "/") for target in target_suffixes):
                continue
            self._per_env_rows[env_id].append(shape_index)
            self._matched_labels.append((env_id, shape_index, label_str))

        total_rows = sum(len(rows) for rows in self._per_env_rows)
        if total_rows == 0:
            raise ValueError(
                "NewtonShapeColorWriter resolved zero shape rows. Check num_envs, env_pattern, and that "
                f"'{mesh_prim_path}' matches scene prims. First few labels: {list(model.shape_label[:6])}"
            )

        # Pre-build a flat (rows, env_idx) index pair so write_colors can vectorize.
        flat_rows: list[int] = []
        env_id_per_row: list[int] = []
        for env_id, rows in enumerate(self._per_env_rows):
            for r in rows:
                flat_rows.append(r)
                env_id_per_row.append(env_id)
        self._flat_rows = np.asarray(flat_rows, dtype=np.int64)
        self._env_id_per_row = np.asarray(env_id_per_row, dtype=np.int64)
        # Cached once so write_colors does not pay O(num_targets) Python-list alloc per reset.
        self._env_id_per_row_list = self._env_id_per_row.tolist()

    # ------------------------------------------------------------------ #
    # Introspection helpers
    # ------------------------------------------------------------------ #

    @property
    def matched_labels(self) -> list[tuple[int, int, str]]:
        """``(env_id, shape_index, label)`` triples for every shape this writer will recolor."""
        return list(self._matched_labels)

    # ------------------------------------------------------------------ #
    # Core write API
    # ------------------------------------------------------------------ #

    @property
    def num_targets(self) -> int:
        """Number of visual shapes this writer recolors (one color sample per target)."""
        return int(self._flat_rows.shape[0])

    def write_colors(self, env_ids: torch.Tensor, colors: torch.Tensor) -> None:
        """Apply one diffuse color per visual shape, for the shapes whose env is being reset.

        Args:
            env_ids: Tensor of int env indices to recolor, shape ``(E,)``. May be empty (no-op).
            colors: Tensor of per-target sRGB diffuse colors, shape ``(num_targets, 3)``, values in
                [0, 1], aligned to this writer's target order (see :attr:`num_targets`).

        Raises:
            ValueError: On shape mismatch.
        """
        targets = select_visual_color_targets(env_ids, colors, self._env_id_per_row_list, self.num_targets)
        if not targets:
            return  # empty subset -> no-op (matches `randomize_rigid_body_material` env_ids=[] semantics).

        colors_cpu = colors.detach().to(device="cpu", dtype=torch.float32).numpy()

        # Round-trip through host: read all rows, scatter per-target into the subset, upload back.
        # `assign` does an in-place wp.copy on the existing device buffer, so the render
        # context's bound reference to model.shape_color stays valid (no notify needed).
        buf = self._model.shape_color.numpy().copy()  # (n_total, 3) float32
        for g in targets:
            buf[int(self._flat_rows[g])] = colors_cpu[g]
        self._model.shape_color.assign(buf)


__all__ = ["NewtonShapeColorWriter"]
