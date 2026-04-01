# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-backed XformPrimView using sites (body + local offset) for GPU-native pose queries."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.sim.views.base_xform_prim_view import BaseXformPrimView
from isaaclab_newton.physics.newton_manager import NewtonManager


@wp.kernel
def _compute_site_world_transforms(
    body_q: wp.array(dtype=wp.transformf),
    site_body: wp.array(dtype=wp.int32),
    site_local: wp.array(dtype=wp.transformf),
    out_pos: wp.array(dtype=wp.vec3f),
    out_quat: wp.array(dtype=wp.vec4f),
):
    """Compute world transforms for sites: world = body_q[body] * local."""
    i = wp.tid()
    world = wp.transform_multiply(body_q[site_body[i]], site_local[i])
    out_pos[i] = wp.transform_get_translation(world)
    q = wp.transform_get_rotation(world)
    out_quat[i] = wp.vec4f(q[0], q[1], q[2], q[3])


class XformPrimView(BaseXformPrimView):
    """Batched prim view backed by Newton sites on GPU.

    Each matched prim is resolved to a Newton *site* -- a ``(body_index,
    local_transform)`` pair.  World poses are computed on GPU as
    ``body_q[body] * local_transform`` using a Warp kernel, matching Newton's
    native site mechanism.

    Resolution order for each prim path:

    1. Look up in ``model.shape_label`` (sites and collision shapes).  If found,
       use ``shape_body`` and ``shape_transform`` for the body index and local
       offset.
    2. Fall back to ``model.body_label`` (the body itself).  Local offset is
       identity.
    """

    def __init__(self, prim_path: str, device: str = "cpu", stage=None, **kwargs):
        self._prim_path = prim_path
        self._device = device

        stage = sim_utils.get_current_stage() if stage is None else stage
        self._prims = sim_utils.find_matching_prims(prim_path, stage=stage)

        model = NewtonManager.get_model()
        body_labels = list(model.body_label)
        shape_labels = list(model.shape_label)
        shape_body_np = wp.to_torch(model.shape_body)
        shape_xform_np = wp.to_torch(model.shape_transform)

        prim_paths = [str(p.GetPath()) for p in self._prims]
        site_bodies: list[int] = []
        site_locals: list[list[float]] = []

        identity_xform = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

        for pp in prim_paths:
            if pp in shape_labels:
                si = shape_labels.index(pp)
                site_bodies.append(int(shape_body_np[si]))
                site_locals.append(shape_xform_np[si].tolist())
            elif pp in body_labels:
                site_bodies.append(body_labels.index(pp))
                site_locals.append(identity_xform)
            else:
                raise ValueError(
                    f"XformPrimView (Newton): prim '{pp}' not found in model shape_label or body_label. "
                    f"Shape labels (first 10): {shape_labels[:10]}, "
                    f"Body labels (first 10): {body_labels[:10]}"
                )

        self._site_body = wp.array(site_bodies, dtype=wp.int32, device=device)
        self._site_local = wp.array(
            [wp.transform(*x) for x in site_locals],
            dtype=wp.transformf,
            device=device,
        )

        self._pos_buf = wp.zeros(self.count, dtype=wp.vec3f, device=device)
        self._quat_buf = wp.zeros(self.count, dtype=wp.vec4f, device=device)

    @property
    def count(self) -> int:
        return len(self._prims)

    # ------------------------------------------------------------------
    # World poses
    # ------------------------------------------------------------------

    def get_world_poses(
        self, indices: Sequence[int] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute site world poses on GPU: ``body_q[body] * local``."""
        state = NewtonManager.get_state_0()

        wp.launch(
            _compute_site_world_transforms,
            dim=self.count,
            inputs=[state.body_q, self._site_body, self._site_local],
            outputs=[self._pos_buf, self._quat_buf],
            device=self._device,
        )

        pos, quat = wp.to_torch(self._pos_buf), wp.to_torch(self._quat_buf)
        if indices is not None:
            return pos[indices], quat[indices]
        return pos, quat

    def set_world_poses(
        self,
        positions: torch.Tensor | None = None,
        orientations: torch.Tensor | None = None,
        indices: Sequence[int] | None = None,
    ) -> None:
        """Write world poses into Newton ``state.body_q``.

        Only works correctly for sites with identity local transforms (i.e.
        body-level prims).  For offset sites the inverse local transform would
        need to be applied first.
        """
        state = NewtonManager.get_state_0()
        body_q = wp.to_torch(state.body_q)

        if indices is not None:
            idx_t = torch.as_tensor(indices, dtype=torch.long, device=self._device)
            bodies = wp.to_torch(self._site_body)[idx_t]
        else:
            bodies = wp.to_torch(self._site_body)

        if positions is not None:
            body_q[bodies, :3] = positions
        if orientations is not None:
            body_q[bodies, 3:7] = orientations

    # ------------------------------------------------------------------
    # Local poses -- delegate to world (Newton bodies live in world space)
    # ------------------------------------------------------------------

    def get_local_poses(self, indices=None) -> tuple[torch.Tensor, torch.Tensor]:
        return self.get_world_poses(indices)

    def set_local_poses(self, translations=None, orientations=None, indices=None) -> None:
        self.set_world_poses(positions=translations, orientations=orientations, indices=indices)

    # ------------------------------------------------------------------
    # Scales
    # ------------------------------------------------------------------

    def get_scales(self, indices: Sequence[int] | None = None) -> torch.Tensor:
        model = NewtonManager.get_model()
        shape_scale = wp.to_torch(model.shape_scale)
        shape_body = wp.to_torch(model.shape_body)

        if indices is not None:
            bodies = wp.to_torch(self._site_body)[torch.as_tensor(indices, dtype=torch.long, device=self._device)]
        else:
            bodies = wp.to_torch(self._site_body)

        scales = []
        for body_idx in bodies:
            mask = shape_body == body_idx
            body_scales = shape_scale[mask]
            scales.append(body_scales[0] if len(body_scales) > 0 else torch.ones(3, device=self._device))
        return torch.stack(scales)

    def set_scales(self, scales: torch.Tensor, indices: Sequence[int] | None = None) -> None:
        model = NewtonManager.get_model()
        shape_scale = wp.to_torch(model.shape_scale)
        shape_body = wp.to_torch(model.shape_body)

        if indices is not None:
            bodies = wp.to_torch(self._site_body)[torch.as_tensor(indices, dtype=torch.long, device=self._device)]
        else:
            bodies = wp.to_torch(self._site_body)

        for i, body_idx in enumerate(bodies):
            mask = shape_body == body_idx
            shape_scale[mask] = scales[i]
