# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Events for the deformable lift environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.assets import DeformableObject, RigidObject
    from isaaclab.envs import ManagerBasedEnv


@wp.kernel
def _write_tet_materials(
    tet_indices: wp.array2d(dtype=wp.int32),
    particle_offset0: wp.int32,
    particles_per_body: wp.int32,
    k_mu: wp.array(dtype=wp.float32),
    k_lambda: wp.array(dtype=wp.float32),
    tet_materials: wp.array2d(dtype=wp.float32),
):
    """Write per-env Lame parameters into the shared tet-material array (leaves k_damp untouched)."""
    t = wp.tid()
    # Map the tet to its env via its first particle index (contiguous under replicate_physics).
    e = (tet_indices[t, 0] - particle_offset0) // particles_per_body
    tet_materials[t, 0] = k_mu[e]
    tet_materials[t, 1] = k_lambda[e]


@wp.kernel
def _scale_particle_mass(
    offsets: wp.array(dtype=wp.int32),
    density_scale: wp.array(dtype=wp.float32),
    spawn_mass: wp.array(dtype=wp.float32),
    particle_mass: wp.array(dtype=wp.float32),
    particle_inv_mass: wp.array(dtype=wp.float32),
):
    """Scale free particle masses by the per-env density ratio; skip kinematic particles."""
    e, j = wp.tid()
    flat_idx = offsets[e] + j
    if particle_inv_mass[flat_idx] == 0.0:
        return
    m = spawn_mass[flat_idx] * density_scale[e]
    particle_mass[flat_idx] = m
    particle_inv_mass[flat_idx] = 1.0 / m


def randomize_deformable_material(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    youngs_modulus_range: tuple[float, float] = (7e4, 5e5),
    density_range: tuple[float, float] = (100.0, 1000.0),
    poissons_ratio: float = 0.25,
) -> None:
    """Randomize the deformable object's material stiffness and density per environment.

    Startup event that samples a Young's modulus and density independently for every
    environment instance and writes the corresponding Lame parameters into the Newton
    tetrahedral materials and the particle masses. The tetrahedral damping term
    ``k_damp`` is preserved. Kinematic particles keep their infinite mass.

    Args:
        env: The environment instance.
        env_ids: Unused; all instances are randomized at startup.
        asset_cfg: Scene entity of the deformable object to randomize.
        youngs_modulus_range: Sampling bounds for the Young's modulus [Pa].
        density_range: Sampling bounds for the mass density [kg/m^3].
        poissons_ratio: Poisson's ratio [dimensionless] used to convert to Lame parameters.
    """
    # Imported here, not at module scope: pulling ``newton`` imports ``pxr`` and a second USD
    # runtime, which must not happen before the Kit app has started.
    from isaaclab_newton.physics import NewtonManager
    from newton import ModelFlags

    asset = env.scene[asset_cfg.name]
    device = env.device
    num_instances = asset.num_instances

    model = NewtonManager.get_model()
    if model is None:
        return

    nu = poissons_ratio

    # Sample per-env Young's modulus and density, then convert E to Lame parameters.
    youngs = torch.empty(num_instances, device=device).uniform_(*youngs_modulus_range)
    density = torch.empty(num_instances, device=device).uniform_(*density_range)
    k_mu = youngs / (2.0 * (1.0 + nu))
    k_lambda = youngs * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    k_mu_wp = wp.from_torch(k_mu.contiguous(), dtype=wp.float32)
    k_lambda_wp = wp.from_torch(k_lambda.contiguous(), dtype=wp.float32)

    particle_offset0 = int(asset._recorded_particle_offsets[0])
    particles_per_body = asset._particles_per_body

    wp.launch(
        _write_tet_materials,
        dim=(model.tet_materials.shape[0],),
        inputs=[model.tet_indices, particle_offset0, particles_per_body, k_mu_wp, k_lambda_wp],
        outputs=[model.tet_materials],
        device=device,
    )

    # Scale masses by density relative to the spawn baseline (spawn mass already encodes it).
    spawn_density = asset.cfg.spawn.physics_material.density
    density_scale = (density / spawn_density).contiguous()
    density_scale_wp = wp.from_torch(density_scale, dtype=wp.float32)
    spawn_mass = wp.clone(model.particle_mass)

    wp.launch(
        _scale_particle_mass,
        dim=(num_instances, particles_per_body),
        inputs=[asset._particle_offsets, density_scale_wp, spawn_mass],
        outputs=[model.particle_mass, model.particle_inv_mass],
        device=device,
    )

    # Refresh the asset's cached inverse-mass snapshot used by the kinematic-target restore.
    asset._default_particle_inv_mass = wp.clone(model.particle_inv_mass)

    # notify the solver that model properties changed, else the randomization is ignored
    NewtonManager.add_model_change(ModelFlags.MODEL_PROPERTIES)


def reset_deformable_over_support(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: dict[str, tuple[float, float]],
    support_offset_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
    support_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> None:
    """Reset a deformable object and keep a support body underneath it.

    The deformable is displaced from its default nodal state by a sample from
    :paramref:`position_range`. The support receives the same planar displacement plus an
    independent sample from :paramref:`support_offset_range`, so it stays under the deformable
    while still varying between resets.

    Args:
        env: The environment instance.
        env_ids: The environment indices to reset.
        position_range: Deformable displacement bounds [m] keyed by ``x``, ``y``, ``z``.
        support_offset_range: Support jitter bounds [m] keyed by ``x``, ``y``, applied on top of
            the deformable's displacement.
        asset_cfg: Scene entity of the deformable object to reset.
        support_cfg: Scene entity of the rigid support body to keep underneath.
    """
    deformable: DeformableObject = env.scene[asset_cfg.name]
    support: RigidObject = env.scene[support_cfg.name]

    # shared planar displacement, so the support tracks the deformable
    ranges = torch.tensor([position_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z")], device=deformable.device)
    offset = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=deformable.device)

    nodal_state = deformable.data.default_nodal_state_w.torch[env_ids].clone()
    nodal_state[..., :3] += offset.unsqueeze(1)
    deformable.write_nodal_state_to_sim(nodal_state, env_ids=env_ids)

    ranges = torch.tensor([support_offset_range.get(key, (0.0, 0.0)) for key in ("x", "y")], device=support.device)
    jitter = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 2), device=support.device)

    root_pose = support.data.default_root_pose.torch[env_ids].clone()
    root_pose[:, :3] += env.scene.env_origins[env_ids]
    root_pose[:, :2] += offset[:, :2] + jitter
    support.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
    support.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros_like(support.data.default_root_vel.torch[env_ids]), env_ids=env_ids
    )
