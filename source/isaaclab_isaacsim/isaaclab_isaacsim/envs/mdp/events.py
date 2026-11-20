# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend implementations of MDP event terms for PhysX."""

from __future__ import annotations

import logging
import math
import re
from types import ModuleType
from typing import TYPE_CHECKING, Literal

import torch
import warp as wp

from isaaclab import assets
from isaaclab import sim as sim_utils
from isaaclab.envs.mdp.events import _GravityRandomization, _randomize_prop_by_op
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils.version import compare_versions, has_kit

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv


class randomize_rigid_body_material(ManagerTermBase):
    """PhysX backend implementation for material randomization.

    Uses the bucket-based approach required by PhysX's 64000 unique material limit.
    Materials are pre-sampled into buckets and randomly assigned to shapes.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv) -> None:
        """Sample buckets and resolve the asset's selected shapes."""
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        asset: RigidObject | Articulation = env.scene[asset_cfg.name]

        # obtain parameters for sampling friction and restitution values
        static_friction_range = cfg.params.get("static_friction_range", (1.0, 1.0))
        dynamic_friction_range = cfg.params.get("dynamic_friction_range", (1.0, 1.0))
        restitution_range = cfg.params.get("restitution_range", (0.0, 0.0))
        num_buckets = int(cfg.params.get("num_buckets", 1))

        # sample material properties from the given ranges
        # note: we only sample the materials once during initialization
        #   afterwards these are randomly assigned to the geometries of the asset
        self._contact_ranges = [cfg.params.get("contact_stiffness_range"), cfg.params.get("contact_damping_range")]
        range_list = [static_friction_range, dynamic_friction_range, restitution_range]
        if any(value is not None for value in self._contact_ranges):
            range_list.extend(value if value is not None else (0.0, 0.0) for value in self._contact_ranges)
        ranges = torch.tensor(range_list, device="cpu")
        self.material_buckets = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (num_buckets, len(range_list)), device="cpu"
        )

        # ensure dynamic friction is always less than static friction
        make_consistent = cfg.params.get("make_consistent", False)
        if make_consistent:
            self.material_buckets[:, 1] = torch.min(self.material_buckets[:, 0], self.material_buckets[:, 1])

        self.asset = asset
        self.asset_cfg = asset_cfg

        self._shape_indices = slice(None)
        if isinstance(asset, assets.BaseArticulation) and asset_cfg.body_ids != slice(None):
            num_shapes_per_body = [
                asset._physics_sim_view.create_rigid_body_view(path).max_shapes
                for path in asset.root_view.link_paths[0]
            ]
            num_shapes = sum(num_shapes_per_body)
            expected_shapes = asset.root_view.max_shapes
            if num_shapes != expected_shapes:
                raise ValueError(
                    "Randomization term 'randomize_rigid_body_material' failed to parse the number of shapes per body."
                    f" Expected total shapes: {expected_shapes}, but got: {num_shapes}."
                )
            self._shape_indices = []
            for body_id in asset.map_body_ids_to_backend(asset_cfg.body_ids):
                start_idx = sum(num_shapes_per_body[:body_id])
                self._shape_indices.extend(range(start_idx, start_idx + num_shapes_per_body[body_id]))

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
        """Assign bucketed material properties to the selected environments and shapes."""
        # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu", dtype=torch.int32)
        elif isinstance(env_ids, slice):
            env_ids = torch.arange(env.scene.num_envs, device="cpu", dtype=torch.int32)[env_ids].contiguous()
        else:
            env_ids = env_ids.to(device="cpu", dtype=torch.int32)

        # randomly assign material IDs to the geometries
        total_num_shapes = self.asset.root_view.max_shapes
        bucket_ids = torch.randint(0, num_buckets, (len(env_ids), total_num_shapes), device="cpu")
        material_samples = self.material_buckets[bucket_ids]

        view = self.asset.root_view
        materials = wp.to_torch(view.get_material_properties())
        # Negative restitution encodes compliant stiffness, not bounce.
        restitution = materials[env_ids, :, 2]
        material_samples[..., 2] = torch.where(restitution < 0.0, restitution, material_samples[..., 2])
        if self._contact_ranges[0] is not None:
            material_samples[..., 2] = -material_samples[..., 3]
        shape_ids = self._shape_indices
        if len(env_ids) == 0 or shape_ids == []:
            return
        rows = env_ids if isinstance(shape_ids, slice) else env_ids[:, None]
        materials[rows, shape_ids] = material_samples[:, shape_ids, :3]

        indices = wp.from_torch(env_ids, dtype=wp.int32)
        if self._contact_ranges[1] is not None:
            compliant, combine_modes = view.get_compliant_material_properties()
            compliant = wp.to_torch(compliant).clone()
            compliant[..., 0] = (-materials[..., 2]).clamp_min(0)
            compliant[rows, shape_ids, 1] = material_samples[:, shape_ids, 4]
            from pxr import UsdPhysics, UsdShade

            # The tensor getter omits friction combine modes; preserve the authored mode.
            friction_modes = set()
            for prim, _ in sim_utils.resolve_matching_prims_from_source(
                self.asset.cfg.prim_path, predicate=lambda prim: prim.HasAPI(UsdPhysics.CollisionAPI)
            ):
                material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose="physics")
                mode = material.GetPrim().GetAttribute("physxMaterial:frictionCombineMode").Get() if material else None
                friction_modes.add(mode or "average")
            if len(friction_modes) != 1:
                raise NotImplementedError("PhysX damping updates require one authored friction combine mode per asset.")
            friction_mode = {"average": 0, "min": 1, "multiply": 2, "max": 3}[friction_modes.pop()]
            combine_modes = wp.to_torch(combine_modes)
            modes = torch.cat((torch.full_like(combine_modes[..., :1], friction_mode), combine_modes), dim=-1)
            data = torch.cat((materials[..., :2], compliant), dim=-1)
            view.set_compliant_material_properties(
                wp.from_torch(data, dtype=wp.float32),
                wp.from_torch(modes, dtype=wp.uint8),
                indices,
            )
        else:
            view.set_material_properties(
                wp.from_torch(materials.contiguous(), dtype=wp.float32),
                indices,
            )


class randomize_rigid_body_collider_offsets(ManagerTermBase):
    """PhysX backend implementation for collider offset randomization.

    Uses rest offset and contact offset directly via the PhysX tensor API.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv) -> None:
        """Cache the asset's collider offsets.

        Args:
            cfg: Event configuration.
            env: Environment owning this term.
        """
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        asset: RigidObject | Articulation = env.scene[asset_cfg.name]
        self.asset = asset
        self.default_rest_offsets = wp.to_torch(asset.root_view.get_rest_offsets()).clone()
        self.default_contact_offsets = wp.to_torch(asset.root_view.get_contact_offsets()).clone()

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice | None,
        asset_cfg: SceneEntityCfg,
        rest_offset_distribution_params: tuple[float, float] | None = None,
        contact_offset_distribution_params: tuple[float, float] | None = None,
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ) -> None:
        """Set collider offsets for the selected environments.

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
            self.asset.root_view.set_rest_offsets(wp.from_torch(rest_offset), wp_env_ids)

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
            self.asset.root_view.set_contact_offsets(wp.from_torch(contact_offset), wp_env_ids)


class randomize_physics_scene_gravity(_GravityRandomization):
    """Randomize scene-wide gravity, shared by every environment.

    Environment IDs do not restrict this global operation. Add and scale start from
    configured gravity each call. Distribution is fixed at construction; distribution
    parameters [m/s^2] may change at runtime.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv) -> None:
        """Initialize gravity sampling for the active simulation.

        Args:
            cfg: Event configuration.
            env: Environment owning this term.
        """
        super().__init__(cfg, env, device="cpu")
        self._physics_sim_view = env.sim.physics_sim_view

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice | None,
        gravity_distribution_params: tuple[list[float], list[float]],
        operation: Literal["add", "scale", "abs"],
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ) -> None:
        """Sample and set gravity.

        Args:
            env: Environment owning this term.
            env_ids: Unused: scene gravity affects every environment.
            gravity_distribution_params: Distribution parameters [m/s^2] for add/abs, dimensionless for scale.
            operation: Apply absolute values, add to the baseline, or scale the baseline.
            distribution: Sampling distribution; gravity terms cache this at construction.
        """
        gravity = torch.tensor(env.sim.cfg.gravity, device="cpu").unsqueeze(0)
        gravity = self._sample_gravity(gravity, gravity_distribution_params, operation)[0].tolist()
        self._physics_sim_view.set_gravity(gravity)


class randomize_visual_color(ManagerTermBase):
    """Randomize USD mesh colors with Replicator."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv) -> None:
        """Initialize the randomization term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.
        """
        super().__init__(cfg, env)

        self._rep = rep = _get_replicator()
        # this term draws from Replicator's random streams, so they follow the environment seed
        if env.cfg.seed is not None:
            rep.set_global_seed(env.cfg.seed)

        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        mesh_name: str = cfg.params["mesh_name"]  # type: ignore

        asset = env.scene[asset_cfg.name]

        # Binding materials on the articulation root invalidates its PhysX view.
        if mesh_name:
            if not mesh_name.startswith("/"):
                mesh_name = "/" + mesh_name
            mesh_prim_path = f"{asset.cfg.prim_path}{mesh_name}"
        else:
            body_names = asset_cfg.body_names
            body_names_regex = "|".join(body_names) if isinstance(body_names, list) else body_names
            body_names_regex = f"(?:{body_names_regex})" if isinstance(body_names_regex, str) else ".*"
            pattern_with_visuals = f"{asset.cfg.prim_path}/{body_names_regex}/visuals"
            if sim_utils.resolve_matching_prims_from_source(pattern_with_visuals, raise_if_no_matches=False):
                mesh_prim_path = pattern_with_visuals
            else:
                mesh_prim_path = f"{asset.cfg.prim_path}/.*"
                logging.info(
                    f"Pattern '{pattern_with_visuals}' found no prims. Falling back to '{mesh_prim_path}'"
                    " for color randomization."
                )

        version = re.match(r"^(\d+\.\d+\.\d+)", rep.__file__.split("/")[-5][21:]).group(1)

        if compare_versions(version, "1.12.4") < 0:
            colors = cfg.params["colors"]
            event_name = cfg.params["event_name"]
            if isinstance(colors, dict):
                color_low = [colors[key][0] for key in ["r", "g", "b"]]
                color_high = [colors[key][1] for key in ["r", "g", "b"]]
                colors = rep.distribution.uniform(color_low, color_high)
            else:
                colors = list(colors)

            def rep_color_randomization():
                prims_group = rep.get.prims(path_pattern=mesh_prim_path)
                with prims_group:
                    rep.randomizer.color(colors=colors)

                return prims_group.node

            with rep.trigger.on_custom_event(event_name=event_name):
                rep_color_randomization()
        else:
            stage = env.sim.stage
            prims_group = rep.functional.get.prims(path_pattern=mesh_prim_path, stage=stage)

            num_prims = len(prims_group)
            self.color_rng = rep.rng.ReplicatorRNG()

            for i, prim in enumerate(prims_group):
                if prim.IsInstanceable():
                    prim.SetInstanceable(False)

            omni_pbr_mdl = _get_omni_pbr_mdl()

            self.material_prims = rep.functional.create_batch.material(
                mdl=omni_pbr_mdl, bind_prims=prims_group, count=num_prims, project_uvw=True
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice,
        event_name: str,
        asset_cfg: SceneEntityCfg,
        colors: list[tuple[float, float, float]] | dict[str, tuple[float, float]],
        mesh_name: str = "",
    ) -> None:
        # Replicator updates all matched prims; env_ids do not restrict the update.
        rep = self._rep

        version = re.match(r"^(\d+\.\d+\.\d+)", rep.__file__.split("/")[-5][21:]).group(1)

        if compare_versions(version, "1.12.4") < 0:
            rep.utils.send_og_event(event_name)
        else:
            colors = colors if colors else self._cfg.params["colors"]

            if isinstance(colors, dict):
                color_low = [colors[key][0] for key in ["r", "g", "b"]]
                color_high = [colors[key][1] for key in ["r", "g", "b"]]
                colors = [color_low, color_high]
            else:
                colors = list(colors)

            num_prims = len(self.material_prims)
            random_colors = self.color_rng.generator.uniform(colors[0], colors[1], size=(num_prims, 3))

            rep.functional.modify.attribute(self.material_prims, "diffuse_color_constant", random_colors)


class randomize_visual_texture_material(ManagerTermBase):
    """Randomize USD mesh textures with Replicator."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv) -> None:
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.
        """
        super().__init__(cfg, env)

        self._rep = rep = _get_replicator()
        # this term draws from Replicator's random streams, so they follow the environment seed
        if env.cfg.seed is not None:
            rep.set_global_seed(env.cfg.seed)

        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        asset = env.scene[asset_cfg.name]

        body_names = asset_cfg.body_names
        body_names_regex = "|".join(body_names) if isinstance(body_names, list) else body_names
        body_names_regex = f"(?:{body_names_regex})" if isinstance(body_names_regex, str) else ".*"

        asset_main_prim_path = asset.cfg.prim_path
        pattern_with_visuals = f"{asset_main_prim_path}/{body_names_regex}/visuals"
        matching_prims = sim_utils.resolve_matching_prims_from_source(pattern_with_visuals, raise_if_no_matches=False)
        if matching_prims:
            prim_path = pattern_with_visuals
        else:
            prim_path = f"{asset_main_prim_path}/.*"
            logging.info(
                f"Pattern '{pattern_with_visuals}' found no prims. Falling back to '{prim_path}' for texture"
                " randomization."
            )

        version = re.match(r"^(\d+\.\d+\.\d+)", rep.__file__.split("/")[-5][21:]).group(1)

        if compare_versions(version, "1.12.4") < 0:
            texture_paths = cfg.params["texture_paths"]
            event_name = cfg.params["event_name"]
            texture_rotation = cfg.params["texture_rotation"]

            texture_rotation = tuple(math.degrees(angle) for angle in texture_rotation)

            def rep_texture_randomization():
                prims_group = rep.get.prims(path_pattern=prim_path)

                with prims_group:
                    rep.randomizer.texture(
                        textures=texture_paths,
                        project_uvw=True,
                        texture_rotate=rep.distribution.uniform(*texture_rotation),
                    )
                return prims_group.node

            with rep.trigger.on_custom_event(event_name=event_name):
                rep_texture_randomization()
        else:
            stage = env.sim.stage
            prims_group = rep.functional.get.prims(path_pattern=prim_path, stage=stage)

            num_prims = len(prims_group)
            self.texture_rng = rep.rng.ReplicatorRNG()

            for i, prim in enumerate(prims_group):
                if prim.IsInstanceable():
                    prim.SetInstanceable(False)

            omni_pbr_mdl = _get_omni_pbr_mdl()

            self.material_prims = rep.functional.create_batch.material(
                mdl=omni_pbr_mdl, bind_prims=prims_group, count=num_prims, project_uvw=True
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice,
        event_name: str,
        asset_cfg: SceneEntityCfg,
        texture_paths: list[str],
        texture_rotation: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        # Replicator updates all matched prims; env_ids do not restrict the update.
        rep = self._rep

        version = re.match(r"^(\d+\.\d+\.\d+)", rep.__file__.split("/")[-5][21:]).group(1)

        if compare_versions(version, "1.12.4") < 0:
            rep.utils.send_og_event(event_name)
        else:
            texture_paths = texture_paths if texture_paths else self._cfg.params["texture_paths"]
            texture_rotation = texture_rotation if texture_rotation else self._cfg.params["texture_rotation"]

            texture_rotation = tuple(math.degrees(angle) for angle in texture_rotation)

            num_prims = len(self.material_prims)
            random_textures = self.texture_rng.generator.choice(texture_paths, size=num_prims)
            random_rotations = self.texture_rng.generator.uniform(
                texture_rotation[0], texture_rotation[1], size=num_prims
            )

            rep.functional.modify.attribute(self.material_prims, "diffuse_texture", random_textures)
            rep.functional.modify.attribute(self.material_prims, "texture_rotate", random_rotations)


def _get_replicator() -> ModuleType:
    """Import Replicator after enabling its Kit extension."""
    if not has_kit():
        raise NotImplementedError("Replicator visual events require Isaac Sim (Omniverse Kit).")
    sim_utils.enable_extension("omni.replicator.core")
    import omni.replicator.core as rep

    return rep


def _get_omni_pbr_mdl() -> str:
    """Resolve an absolute MDL path; Kit may bypass built-in MDL short names."""
    import carb.tokens

    return carb.tokens.get_tokens_interface().resolve("${kit}/mdl/core/Base/OmniPBR.mdl")
