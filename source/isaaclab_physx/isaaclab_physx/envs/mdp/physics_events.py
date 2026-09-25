# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX backend implementations for MDP event terms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
import warp as wp

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.physics_events import randomize_prop_by_op

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import EventTermCfg, SceneEntityCfg


class RandomizeRigidBodyMaterial:
    """PhysX backend implementation for material randomization.

    Uses the bucket-based approach required by PhysX's 64000 unique material limit.
    Materials are pre-sampled into buckets and randomly assigned to shapes.
    """

    def __init__(
        self, cfg: EventTermCfg, env: ManagerBasedEnv, asset: RigidObject | Articulation, asset_cfg: SceneEntityCfg
    ):
        from isaaclab.assets import BaseArticulation

        # obtain parameters for sampling friction and restitution values
        static_friction_range = cfg.params.get("static_friction_range", (1.0, 1.0))
        dynamic_friction_range = cfg.params.get("dynamic_friction_range", (1.0, 1.0))
        restitution_range = cfg.params.get("restitution_range", (0.0, 0.0))
        num_buckets = int(cfg.params.get("num_buckets", 1))

        # sample material properties from the given ranges
        # note: we only sample the materials once during initialization
        #   afterwards these are randomly assigned to the geometries of the asset
        range_list = [static_friction_range, dynamic_friction_range, restitution_range]
        ranges = torch.tensor(range_list, device="cpu")
        self.material_buckets = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_buckets, 3), device="cpu")

        # ensure dynamic friction is always less than static friction
        make_consistent = cfg.params.get("make_consistent", False)
        if make_consistent:
            self.material_buckets[:, 1] = torch.min(self.material_buckets[:, 0], self.material_buckets[:, 1])

        self.asset = asset
        self.asset_cfg = asset_cfg

        # obtain number of shapes per body (needed for indexing the material properties correctly)
        # note: this is a workaround since the Articulation does not provide a direct way to obtain the number of shapes
        #  per body. We use the physics simulation view to obtain the number of shapes per body.
        if isinstance(asset, BaseArticulation) and asset_cfg.body_ids != slice(None):
            self.num_shapes_per_body = []
            for link_path in asset.root_view.link_paths[0]:
                link_physx_view = asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
                self.num_shapes_per_body.append(link_physx_view.max_shapes)
            # ``body_ids`` are public IDs; convert once before deriving backend-ordered shape ranges.
            self._backend_body_ids = asset.map_body_ids_to_backend(asset_cfg.body_ids)
            num_shapes = sum(self.num_shapes_per_body)
            expected_shapes = asset.root_view.max_shapes
            if num_shapes != expected_shapes:
                raise ValueError(
                    "Randomization term 'randomize_rigid_body_material' failed to parse the number of shapes per body."
                    f" Expected total shapes: {expected_shapes}, but got: {num_shapes}."
                )
        else:
            # in this case, we don't need to do special indexing
            self.num_shapes_per_body = None
            self._backend_body_ids = None

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int,
        asset_cfg: SceneEntityCfg,
        make_consistent: bool = False,
    ):
        # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu", dtype=torch.int32)
        else:
            env_ids = env_ids.cpu()

        # randomly assign material IDs to the geometries
        total_num_shapes = self.asset.root_view.max_shapes
        bucket_ids = torch.randint(0, num_buckets, (len(env_ids), total_num_shapes), device="cpu")
        material_samples = self.material_buckets[bucket_ids]

        materials = wp.to_torch(self.asset.root_view.get_material_properties())
        if self.num_shapes_per_body is not None:
            for body_id in self._backend_body_ids:
                start_idx = sum(self.num_shapes_per_body[:body_id])
                end_idx = start_idx + self.num_shapes_per_body[body_id]
                # assign the new materials
                # material samples are of shape: num_env_ids x total_num_shapes x 3
                materials[env_ids, start_idx:end_idx] = material_samples[:, start_idx:end_idx]
        else:
            # assign all the materials
            materials[env_ids] = material_samples[:]

        self.asset.root_view.set_material_properties(
            wp.from_torch(materials, dtype=wp.float32), wp.from_torch(env_ids, dtype=wp.int32)
        )


class RandomizeRigidBodyColliderOffsets:
    """PhysX backend implementation for collider offset randomization.

    Uses rest offset and contact offset directly via the PhysX tensor API.
    """

    def __init__(self, asset: RigidObject | Articulation):
        self.asset = asset
        self.default_rest_offsets = wp.to_torch(asset.root_view.get_rest_offsets()).clone()
        self.default_contact_offsets = wp.to_torch(asset.root_view.get_contact_offsets()).clone()

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        rest_offset_distribution_params: tuple[float, float] | None = None,
        contact_offset_distribution_params: tuple[float, float] | None = None,
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ):
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu", dtype=torch.int32)
        else:
            env_ids = env_ids.to(device="cpu", dtype=torch.int32)
        wp_env_ids = wp.from_torch(env_ids, dtype=wp.int32)

        if rest_offset_distribution_params is not None:
            rest_offset = self.default_rest_offsets.clone()
            rest_offset = randomize_prop_by_op(
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
            contact_offset = randomize_prop_by_op(
                contact_offset,
                contact_offset_distribution_params,
                None,
                slice(None),
                operation="abs",
                distribution=distribution,
            )
            self.asset.root_view.set_contact_offsets(wp.from_torch(contact_offset), wp_env_ids)


class RandomizePhysicsSceneGravity:
    """PhysX backend implementation for scene gravity randomization.

    Samples a single gravity vector and sets it scene-wide via the PhysX simulation view;
    all environments share the same gravity.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        import carb  # noqa: PLC0415

        self._carb = carb
        self._physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view

        distribution = cfg.params.get("distribution", "uniform")
        if distribution not in ("uniform", "log_uniform", "gaussian"):
            raise NotImplementedError(
                f"Unknown distribution: '{distribution}' for gravity randomization."
                " Please use 'uniform', 'log_uniform', or 'gaussian'."
            )
        self._distribution = distribution

        operation = cfg.params["operation"]
        if operation not in ("add", "scale", "abs"):
            raise NotImplementedError(
                f"Unknown operation: '{operation}' for gravity randomization. Please use 'add', 'scale', or 'abs'."
            )

        gravity_distribution_params = cfg.params["gravity_distribution_params"]
        self._dist_param_0 = torch.tensor(gravity_distribution_params[0], device=env.device, dtype=torch.float32)
        self._dist_param_1 = torch.tensor(gravity_distribution_params[1], device=env.device, dtype=torch.float32)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        gravity_distribution_params: tuple[list[float], list[float]],
        operation: Literal["add", "scale", "abs"],
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ):
        self._dist_param_0[0] = gravity_distribution_params[0][0]
        self._dist_param_1[0] = gravity_distribution_params[1][0]
        self._dist_param_0[1] = gravity_distribution_params[0][1]
        self._dist_param_1[1] = gravity_distribution_params[1][1]
        self._dist_param_0[2] = gravity_distribution_params[0][2]
        self._dist_param_1[2] = gravity_distribution_params[1][2]

        # PhysX applies a single gravity vector scene-wide via the simulation view
        gravity = torch.tensor(env.sim.cfg.gravity, device="cpu").unsqueeze(0)
        gravity = randomize_prop_by_op(
            gravity,
            (self._dist_param_0.cpu(), self._dist_param_1.cpu()),
            None,
            slice(None),
            operation=operation,
            distribution=self._distribution,
        )
        gravity = gravity[0].tolist()
        self._physics_sim_view.set_gravity(self._carb.Float3(*gravity))
