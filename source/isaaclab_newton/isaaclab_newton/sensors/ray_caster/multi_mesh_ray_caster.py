# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import warp as wp

from isaaclab.sensors.ray_caster import BaseMultiMeshRayCaster
from isaaclab.sensors.ray_caster._target_tracker_utils import split_env_path, walk_target_prototypes
from isaaclab.sim import SimulationContext

from isaaclab_newton.physics import NewtonManager

from .ray_caster import RayCaster

if TYPE_CHECKING:
    from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg

logger = logging.getLogger(__name__)


@wp.kernel(enable_backward=False)
def _copy_target_xform_kernel(
    transforms: wp.array(dtype=wp.transformf),
    slot_dst: wp.array(dtype=wp.int32),
    num_tracked_per_env: int,
    mesh_positions: wp.array2d(dtype=wp.vec3),
    mesh_rotations: wp.array2d(dtype=wp.quat),
):
    """Split flat ``SensorFrameTransform`` output into ``mesh_positions`` / ``mesh_rotations``.

    Sensor transforms are laid out as ``[env_0_s0, env_0_s1, ..., env_1_s0, ...]``;
    ``slot_dst[s]`` maps tracked-slot ``s`` to its mesh-pose slot.
    """
    e, s = wp.tid()
    xf = transforms[e * num_tracked_per_env + s]
    slot = slot_dst[s]
    mesh_positions[e, slot] = wp.transform_get_translation(xf)
    mesh_rotations[e, slot] = wp.transform_get_rotation(xf)


class MultiMeshRayCaster(BaseMultiMeshRayCaster, RayCaster):
    """Newton backend for the multi-mesh ray-cast sensor.

    Per-(prototype, slot) sites with proto-local body label + body→mesh offset, plus a
    shared world-origin. ``clone_mask`` from
    :func:`~isaaclab.sensors.ray_caster._target_tracker_utils.walk_target_prototypes`
    selects each env's prototype's site labels (heterogeneous-safe). A
    :class:`SensorFrameTransform` measures all per-env tracked sites against
    world-origin; a tiny copy kernel splits the flat output into
    ``_mesh_positions_w`` / ``_mesh_orientations_w``.
    """

    cfg: MultiMeshRayCasterCfg
    __backend_name__: str = "newton"

    def __init__(self, cfg: MultiMeshRayCasterCfg):
        super().__init__(cfg)
        # Sites must register before ``newton_replicate`` (fires from ``start_simulation``).
        self._register_target_sites()

    def _register_target_sites(self) -> None:
        """Walk per-prototype, register one site per ``(proto, slot)``, cache env→proto.
        STOP→reinit reuses the USD-walk cache; site registrations refire (Newton dedupes
        by ``(body_pattern, transform_key)`` so identical args return the same label)."""
        if not getattr(self, "_target_args_cached", False):
            self._gather_target_proto_data()
            self._target_args_cached = True

        if not self._target_slot_dst_cached:
            return  # no tracked targets

        self._tracked_site_labels_per_target_proto: list[list[list[str]]] = []
        for target_entries in self._target_proto_data:
            self._tracked_site_labels_per_target_proto.append(
                [
                    [NewtonManager.cl_register_site(body_label, xform) for body_label, xform in proto_entries]
                    for proto_entries in target_entries
                ]
            )
        self._target_world_origin_label = NewtonManager.cl_register_site(None, wp.transform())

    def _gather_target_proto_data(self) -> None:
        """Populate per-target proto entries + per-env proto assignment via
        :func:`walk_target_prototypes`. Newton-specific work: full path → proto-local
        body label (``cl_register_site`` matches proto builder body labels), offsets
        wrapped as :class:`wp.transform`."""
        sim = SimulationContext.instance()
        plans = list(sim.get_clone_plans().values()) if sim is not None else []

        self._target_proto_data: list[list[list[tuple[str, wp.transform]]]] = []
        self._target_env_proto_per_target: list[list[int]] = []
        self._target_slot_dst_cached: list[int] = []

        for target_cfg, (slot_start, slot_end) in zip(self._raycast_targets_cfg, self._target_slot_ranges):
            if not target_cfg.track_mesh_transforms:
                self._target_proto_data.append([])
                self._target_env_proto_per_target.append([])
                continue
            self._target_slot_dst_cached.extend(range(slot_start, slot_end))
            per_proto, env_proto = walk_target_prototypes(
                target_cfg.prim_expr, plans, self._num_envs, self.cfg.prim_path
            )
            self._target_proto_data.append(
                [
                    [
                        (split_env_path(body_path)[1], wp.transform(offset[:3], offset[3:]))
                        for body_path, offset in proto_entries
                    ]
                    for proto_entries in per_proto
                ]
            )
            self._target_env_proto_per_target.append(env_proto)

    def _initialize_impl(self) -> None:
        """Resolve per-env tracked-site indices and build the target ``SensorFrameTransform``."""
        super()._initialize_impl()
        if not self._target_slot_dst_cached:
            self._target_sensor_index = None
            self._target_newton_transforms = None
            return

        site_map = NewtonManager._cl_site_index_map
        if self._target_world_origin_label not in site_map:
            raise RuntimeError(
                f"MultiMeshRayCaster '{self.cfg.prim_path}': world-origin site label"
                f" '{self._target_world_origin_label}' missing from NewtonManager._cl_site_index_map."
            )
        world_origin_idx, _ = site_map[self._target_world_origin_label]

        # ``shapes_list`` ordering: (env-major, target-major, slot-major). Per env, per
        # tracked target_cfg, look up this env's prototype and emit its slot site indices.
        shapes_list: list[int] = []
        references_list: list[int] = []
        for env_idx in range(self._num_envs):
            for tgt_idx, target_cfg in enumerate(self._raycast_targets_cfg):
                if not target_cfg.track_mesh_transforms:
                    continue
                proto_idx = self._target_env_proto_per_target[tgt_idx][env_idx]
                proto_labels = self._tracked_site_labels_per_target_proto[tgt_idx][proto_idx]
                for label in proto_labels:
                    _, per_world = site_map[label]
                    if per_world is None or env_idx >= len(per_world) or len(per_world[env_idx]) != 1:
                        n = (
                            "None"
                            if per_world is None
                            else ("missing-env" if env_idx >= len(per_world) else len(per_world[env_idx]))
                        )
                        raise RuntimeError(
                            f"MultiMeshRayCaster '{self.cfg.prim_path}': site '{label}' has"
                            f" {n} matches in env {env_idx}, expected exactly 1."
                        )
                    shapes_list.append(per_world[env_idx][0])
                    references_list.append(world_origin_idx)

        self._target_sensor_index = NewtonManager.add_frame_transform_sensor(shapes_list, references_list)
        self._target_newton_transforms = NewtonManager._newton_frame_transform_sensors[
            self._target_sensor_index
        ].transforms

        self._target_slot_dst_wp = wp.array(self._target_slot_dst_cached, dtype=wp.int32, device=self._device)
        self._num_tracked_per_env = len(self._target_slot_dst_cached)

        logger.info(
            f"MultiMeshRayCaster '{self.cfg.prim_path}': tracking {self._num_tracked_per_env} target"
            f" mesh(es) per env, target sensor_index={self._target_sensor_index}."
        )

    def _update_target_mesh_transforms(self) -> None:
        """Split flat tracked-target sensor output into mesh-pose buffers."""
        if self._target_newton_transforms is None:
            return
        wp.launch(
            _copy_target_xform_kernel,
            dim=(self._num_envs, self._num_tracked_per_env),
            inputs=[
                self._target_newton_transforms,
                self._target_slot_dst_wp,
                self._num_tracked_per_env,
            ],
            outputs=[self._mesh_positions_w, self._mesh_orientations_w],
            device=self._device,
        )

    def _invalidate_initialize_callback(self, event) -> None:
        """Drop sensor refs on STOP and re-register sites (matches sensor-body path)."""
        super()._invalidate_initialize_callback(event)
        self._target_newton_transforms = None
        self._target_sensor_index = None
        if self._target_slot_dst_cached:
            self._register_target_sites()
