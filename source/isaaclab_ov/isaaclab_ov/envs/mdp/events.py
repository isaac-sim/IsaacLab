# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend implementations of MDP event terms for OVPhysX."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
import warp as wp

from pxr import UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab import assets
from isaaclab.envs.mdp.events import _GravityRandomization, _randomize_prop_by_op
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils import math as math_utils

from ... import tensor_types as ovphysx_tt
from ...sim.views.ovphysx_view import OvPhysxView

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv


class randomize_rigid_body_material(ManagerTermBase):
    """OVPhysX backend implementation for material randomization.

    OVPhysX runs the PhysX solver, so PhysX's 64000 unique-material limit applies and this
    mirrors the PhysX bucket approach: ``num_buckets`` materials are pre-sampled once and
    randomly assigned to shapes. Materials are written through the asset's
    :class:`~isaaclab_ov.sim.views.OvPhysxView` on the per-collision-shape
    ``shape_friction_and_restitution`` binding (shape ``[N, S, 3]`` = static friction,
    dynamic friction, restitution).

    Whole-articulation randomization uses the articulation material binding. For a
    body subset, individual articulation links are addressed through a rigid-body
    material binding, whose rows expose the exact link prim paths and per-link shape
    storage.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv) -> None:
        """Bind this term to the active simulation and capture term-local state.

        Args:
            cfg: Event configuration.
            env: Environment owning this term.
        """
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        asset: RigidObject | Articulation = env.scene[asset_cfg.name]

        # sample material buckets once (PhysX-style; the 64000 unique-material limit applies)
        static_friction_range = cfg.params.get("static_friction_range", (1.0, 1.0))
        dynamic_friction_range = cfg.params.get("dynamic_friction_range", (1.0, 1.0))
        restitution_range = cfg.params.get("restitution_range", (0.0, 0.0))
        num_buckets = int(cfg.params.get("num_buckets", 1))
        ranges = torch.tensor([static_friction_range, dynamic_friction_range, restitution_range], device="cpu")
        self.material_buckets = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_buckets, 3), device="cpu")
        if cfg.params.get("make_consistent", False):
            self.material_buckets[:, 1] = torch.min(self.material_buckets[:, 0], self.material_buckets[:, 1])

        self.asset = asset
        self.asset_cfg = asset_cfg
        self._material_view = asset.root_view
        self._material_rows_by_env = torch.arange(asset.num_instances, dtype=torch.long).unsqueeze(-1)

        if isinstance(asset, assets.BaseArticulation):
            self._material_type = ovphysx_tt.SHAPE_FRICTION_AND_RESTITUTION
            if not _is_all_body_selection(asset_cfg.body_ids, asset.num_bodies):
                body_ids = [int(body_id) for body_id in asset_cfg.body_ids]
                if len(body_ids) == 0:
                    self._material_view = None
                    self._material_rows_by_env = torch.empty((asset.num_instances, 0), dtype=torch.long)
                    return

                selected_body_names = [asset.body_names[body_id] for body_id in body_ids]
                asset_root_paths = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)
                articulation_root_paths = asset.root_view.prim_paths
                if len(articulation_root_paths) != asset.num_instances:
                    raise RuntimeError(
                        "Failed to map OVPhysX articulation material rows to asset instances: "
                        f"expected {asset.num_instances} articulation roots, got {len(articulation_root_paths)}."
                    )

                # With replicated physics, only the source asset may exist as a concrete
                # USD prim even though the tensor binding contains every environment. Find
                # the articulation-root suffix in that source asset, then strip the same
                # suffix from every binding row to recover its concrete asset root.
                source_pairs = [
                    (asset_root_path, articulation_root_path)
                    for asset_root_path in asset_root_paths
                    for articulation_root_path in articulation_root_paths
                    if articulation_root_path == asset_root_path
                    or articulation_root_path.startswith(f"{asset_root_path}/")
                ]
                if not source_pairs:
                    raise RuntimeError(
                        "Failed to find a source asset root containing an OVPhysX articulation root. "
                        f"Asset roots: {asset_root_paths}; articulation roots: {articulation_root_paths}."
                    )
                source_asset_root, source_articulation_root = max(source_pairs, key=lambda pair: len(pair[0]))
                articulation_root_suffix = source_articulation_root[len(source_asset_root) :]
                if articulation_root_suffix:
                    if not all(path.endswith(articulation_root_suffix) for path in articulation_root_paths):
                        raise RuntimeError(
                            "OVPhysX articulation roots do not share the source asset's relative root suffix "
                            f"'{articulation_root_suffix}': {articulation_root_paths}."
                        )
                    instance_root_paths = [path[: -len(articulation_root_suffix)] for path in articulation_root_paths]
                else:
                    instance_root_paths = articulation_root_paths

                selected_relative_paths = []
                for body_name in selected_body_names:

                    def is_selected_rigid_body(prim, expected_name=body_name):
                        return prim.GetName() == expected_name and prim.HasAPI(UsdPhysics.RigidBodyAPI)

                    source_matches = sim_utils.resolve_matching_prims_from_source(
                        asset.cfg.prim_path,
                        predicate=is_selected_rigid_body,
                        expected_num_matches=1,
                    )
                    source_body_path = source_matches[0][0].GetPath().pathString
                    if not (
                        source_body_path == source_asset_root or source_body_path.startswith(f"{source_asset_root}/")
                    ):
                        raise RuntimeError(
                            f"OVPhysX body '{body_name}' at '{source_body_path}' is not below source asset root "
                            f"'{source_asset_root}'."
                        )
                    selected_relative_paths.append(source_body_path[len(source_asset_root) :])

                selected_paths = []
                for instance_root_path in instance_root_paths:
                    selected_paths.extend(
                        f"{instance_root_path}{relative_path}" for relative_path in selected_relative_paths
                    )

                self._material_type = ovphysx_tt.RIGID_BODY_SHAPE_FRICTION_AND_RESTITUTION
                self._material_view = OvPhysxView(
                    asset._ovphysx,  # type: ignore[attr-defined]
                    prim_paths=selected_paths,
                    device=asset.device,
                )
                selected_binding = self._material_view.binding_for(self._material_type)
                resolved_paths = selected_binding.prim_paths
                if len(resolved_paths) != len(selected_paths) or set(resolved_paths) != set(selected_paths):
                    raise RuntimeError(
                        "OVPhysX rigid-body material binding did not resolve the requested articulation links. "
                        f"Requested {selected_paths}, resolved {resolved_paths}."
                    )
                row_by_path = {path: row for row, path in enumerate(resolved_paths)}
                self._material_rows_by_env = torch.tensor(
                    [row_by_path[path] for path in selected_paths], dtype=torch.long
                ).reshape(asset.num_instances, len(selected_body_names))
        else:
            self._material_type = ovphysx_tt.RIGID_BODY_SHAPE_FRICTION_AND_RESTITUTION
            if not _is_all_body_selection(asset_cfg.body_ids, asset.num_bodies):
                raise NotImplementedError(
                    "randomize_rigid_body_material on the OVPhysX backend cannot apply per-body selection to a "
                    "standalone rigid object. Use the default body selection."
                )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice | None,
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int,
        asset_cfg: SceneEntityCfg,
        make_consistent: bool = False,
    ) -> None:
        """Apply the configured randomization.

        Args:
            env: Environment owning this term.
            env_ids: Environment selection; None selects all environments.
            static_friction_range: Static friction bounds used to construct the material buckets.
            dynamic_friction_range: Dynamic friction bounds used to construct the material buckets.
            restitution_range: Restitution bounds; bucket-based backends use the construction-time range.
            num_buckets: Number of material buckets; must match the construction-time value.
            asset_cfg: Asset and body selection resolved at construction.
            make_consistent: Whether construction constrained dynamic friction to static friction.
        """
        if self._material_view is None:
            return

        view = self._material_view
        # read the current per-shape material [N, S, 3] on the binding's native CPU device
        materials = wp.to_torch(view.get_attribute(self._material_type))
        num_shapes = materials.shape[1]

        # Resolve environment ids to rows of the active material binding. A subset
        # articulation view contains one row per selected body and environment.
        if env_ids is None:
            material_rows = self._material_rows_by_env.flatten()
        else:
            material_rows = self._material_rows_by_env[
                env_ids if isinstance(env_ids, slice) else env_ids.to(device="cpu", dtype=torch.long)
            ].flatten()
        if material_rows.numel() == 0:
            return
        material_rows_device = material_rows.to(materials.device)

        # randomly assign pre-sampled bucket materials to every shape of the selected envs
        bucket_ids = torch.randint(0, num_buckets, (len(material_rows), num_shapes), device="cpu")
        material_samples = self.material_buckets[bucket_ids].to(materials.device)
        materials[material_rows_device] = material_samples

        # The wheel requires a full-shaped source buffer even for indexed writes.
        indices = wp.from_torch(material_rows_device.to(dtype=torch.int32))
        view.set_attribute(
            self._material_type,
            wp.from_torch(materials.contiguous(), dtype=wp.float32),
            indices=indices,
        )


class randomize_rigid_body_collider_offsets(ManagerTermBase):
    """OVPhysX backend implementation for collider offset randomization.

    OVPhysX runs the PhysX solver, so rest and contact offsets are written directly, per collision
    shape, through the asset's :class:`~isaaclab_ov.sim.views.OvPhysxView`. Articulations use the
    articulation offset bindings and rigid objects the rigid-body ones; both are CPU-resident
    ``[N, S]`` buffers, so the full tensor is read-modify-written on the host with the selected
    environments as write indices.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv) -> None:
        """Bind this term to the active simulation and capture term-local state.

        Args:
            cfg: Event configuration.
            env: Environment owning this term.
        """
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        asset: RigidObject | Articulation = env.scene[asset_cfg.name]

        self.asset = asset
        if isinstance(asset, assets.BaseArticulation):
            self._rest_offset_type = ovphysx_tt.REST_OFFSET
            self._contact_offset_type = ovphysx_tt.CONTACT_OFFSET
        else:
            self._rest_offset_type = ovphysx_tt.RIGID_BODY_REST_OFFSET
            self._contact_offset_type = ovphysx_tt.RIGID_BODY_CONTACT_OFFSET
        self.default_rest_offsets = wp.to_torch(asset.root_view.get_attribute(self._rest_offset_type)).clone()
        self.default_contact_offsets = wp.to_torch(asset.root_view.get_attribute(self._contact_offset_type)).clone()

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice | None,
        asset_cfg: SceneEntityCfg,
        rest_offset_distribution_params: tuple[float, float] | None = None,
        contact_offset_distribution_params: tuple[float, float] | None = None,
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ) -> None:
        """Apply the configured randomization.

        Args:
            env: Environment owning this term.
            env_ids: Environment selection; None selects all environments.
            asset_cfg: Asset selection; collider randomization operates on every body.
            rest_offset_distribution_params: Rest offset distribution parameters [m].
            contact_offset_distribution_params: Contact offset distribution parameters [m].
            distribution: Sampling distribution for the offsets.
        """
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu", dtype=torch.int32)
        elif isinstance(env_ids, slice):
            env_ids = torch.arange(env.scene.num_envs, device="cpu", dtype=torch.int32)[env_ids].contiguous()
        else:
            env_ids = env_ids.to(device="cpu", dtype=torch.int32)
        wp_env_ids = wp.from_torch(env_ids, dtype=wp.int32)

        if rest_offset_distribution_params is not None:
            rest_offset = self.default_rest_offsets.clone()
            rest_offset = _randomize_prop_by_op(
                rest_offset,
                rest_offset_distribution_params,
                None,
                slice(None),
                operation="abs",
                distribution=distribution,
            )
            # the wheel requires a full-shaped source buffer even for indexed writes
            self.asset.root_view.set_attribute(
                self._rest_offset_type, wp.from_torch(rest_offset.contiguous(), dtype=wp.float32), indices=wp_env_ids
            )

        if contact_offset_distribution_params is not None:
            contact_offset = self.default_contact_offsets.clone()
            contact_offset = _randomize_prop_by_op(
                contact_offset,
                contact_offset_distribution_params,
                None,
                slice(None),
                operation="abs",
                distribution=distribution,
            )
            self.asset.root_view.set_attribute(
                self._contact_offset_type,
                wp.from_torch(contact_offset.contiguous(), dtype=wp.float32),
                indices=wp_env_ids,
            )


class randomize_physics_scene_gravity(_GravityRandomization):
    """Randomize scene-wide gravity, shared by every environment.

    Environment IDs do not restrict this global operation. Add and scale start from
    configured gravity each call. Distribution is fixed at construction; distribution
    parameters [m/s^2] may change at runtime.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv) -> None:
        """Bind this term to the active simulation and capture term-local state.

        Args:
            cfg: Event configuration.
            env: Environment owning this term.
        """
        super().__init__(cfg, env, device="cpu")
        self._manager = env.sim.physics_manager

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice | None,
        gravity_distribution_params: tuple[list[float], list[float]],
        operation: Literal["add", "scale", "abs"],
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ) -> None:
        """Apply the configured randomization.

        Args:
            env: Environment owning this term.
            env_ids: Unused: scene gravity affects every environment.
            gravity_distribution_params: Distribution parameters [m/s^2] for add/abs, dimensionless for scale.
            operation: Apply absolute values, add to the baseline, or scale the baseline.
            distribution: Sampling distribution; gravity terms cache this at construction.
        """
        gravity = torch.tensor(env.sim.cfg.gravity, device="cpu").unsqueeze(0)
        gravity = self._sample_gravity(gravity, gravity_distribution_params, operation)[0].tolist()
        self._manager.set_gravity(tuple(gravity))


def _is_all_body_selection(body_ids: list[int] | slice, num_bodies: int) -> bool:
    """Return whether a body selector covers the entire asset."""
    if body_ids == slice(None):
        return True
    return sorted(body_ids) == list(range(num_bodies))
