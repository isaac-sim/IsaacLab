# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp device functions and kernels for moving terrain height queries and particles."""

import warp as wp


@wp.func
def surface_height(mesh: wp.uint64, x: float, y: float, ray_z: float, ray_length: float):
    """Cast downward at world XY [m] and return whether the mesh was hit and its height [m]."""
    query = wp.mesh_query_ray(mesh, wp.vec3(x, y, ray_z), wp.vec3(0.0, 0.0, -1.0), ray_length)
    height = ray_z - query.t
    return query.result, height


@wp.kernel
def sample_heights(mesh: wp.uint64, ray_z: float, ray_length: float, points: wp.array[wp.vec3], missed: wp.array[int]):
    """Replace query-point Z coordinates with terrain heights [m] and count missed rays."""
    i = wp.tid()
    p = points[i]
    hit, height = surface_height(mesh, p[0], p[1], ray_z, ray_length)
    if hit:
        points[i] = wp.vec3(p[0], p[1], height)
    else:
        wp.atomic_add(missed, 0, 1)


@wp.kernel
def mark_resets(selected: wp.array[wp.bool], pending: wp.array[int]):
    """Mark selected simulation worlds for particle reinitialization."""
    i = wp.tid()
    if selected[i]:
        pending[i] = 1


@wp.kernel
def update_patch_center(
    body_q: wp.array[wp.transform],
    body_indices: wp.array[int],
    env_origin: wp.array[wp.vec3],
    background_terrain_center: wp.vec2,
    background_terrain_size: wp.vec2,
    total_patch_size: wp.vec2,
    patch_discretization_step: float,
    patch_center: wp.array[wp.vec2],
):
    """Quantize robot-centered patch positions [m] and keep each full patch inside the background."""
    i = wp.tid()
    p = wp.transform_get_translation(body_q[body_indices[i]])
    o = env_origin[i]
    allowed_patch_center_half = (background_terrain_size - total_patch_size) * 0.5
    x = wp.floor((p[0] - o[0]) / patch_discretization_step + 0.5) * patch_discretization_step
    y = wp.floor((p[1] - o[1]) / patch_discretization_step + 0.5) * patch_discretization_step
    patch_center[i] = wp.vec2(
        wp.clamp(
            o[0] + x,
            background_terrain_center[0] - allowed_patch_center_half[0],
            background_terrain_center[0] + allowed_patch_center_half[0],
        ),
        wp.clamp(
            o[1] + y,
            background_terrain_center[1] - allowed_patch_center_half[1],
            background_terrain_center[1] + allowed_patch_center_half[1],
        ),
    )


@wp.kernel
def update_particles(
    particle_env_ids: wp.array[int],
    patch_centers: wp.array[wp.vec2],
    particle_template_positions: wp.array[wp.vec3],
    env_reset_requested: wp.array[int],
    particle_dynamic_mass: wp.array[float],
    initial_particle_plastic_volume_ratio: wp.array[float],
    simulated_terrain_size: wp.vec2,
    total_patch_size: wp.vec2,
    background_mesh: wp.uint64,
    ray_z: float,
    ray_length: float,
    particle_reset_positions: wp.array[wp.vec3],
    particle_is_dynamic: wp.array[int],
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    particle_inv_mass: wp.array[float],
    particle_qd_grad: wp.array[wp.mat33],
    particle_elastic_strain: wp.array[wp.mat33],
    particle_plastic_volume_ratio: wp.array[float],
    particle_stress: wp.array[wp.mat33],
    particle_transform: wp.array[wp.mat33],
    mass_change_count: wp.array[int],
    terrain_query_miss_count: wp.array[int],
):
    """Recycle particles into the patch, restore material state, and assign zero mass to its boundary.

    ``particle_template_positions`` holds fixed world XY coordinates [m] and particle-layer Z offsets [m].
    ``particle_reset_positions`` holds terrain-adjusted world XYZ positions [m]. The environment-indexed
    ``env_reset_requested`` flags request particle reinitialization, rather than recording completed resets.
    Each counter is a one-element array shared across all particles in this launch.
    Read-only inputs precede the output arrays, which are updated in place and may also be read.
    """
    particle_id = wp.tid()
    env_id = particle_env_ids[particle_id]
    patch_center = patch_centers[env_id]
    particle_template_position = particle_template_positions[particle_id]
    # Toroidal wrapping retains overlap and restores fresh material at the entering edge.
    x = (
        particle_template_position[0]
        + wp.floor((patch_center[0] + 0.5 * total_patch_size[0] - particle_template_position[0]) / total_patch_size[0])
        * total_patch_size[0]
    )
    y = (
        particle_template_position[1]
        + wp.floor((patch_center[1] + 0.5 * total_patch_size[1] - particle_template_position[1]) / total_patch_size[1])
        * total_patch_size[1]
    )
    # Float32 wrapping can round past the patch edge and miss the terrain at its total_patch_size boundary.
    x = wp.clamp(x, patch_center[0] - 0.5 * total_patch_size[0], patch_center[0] + 0.5 * total_patch_size[0])
    y = wp.clamp(y, patch_center[1] - 0.5 * total_patch_size[1], patch_center[1] + 0.5 * total_patch_size[1])
    particle_reset_position = particle_reset_positions[particle_id]
    if (
        wp.abs(particle_reset_position[0] - x) > 1.0e-5
        or wp.abs(particle_reset_position[1] - y) > 1.0e-5
        or env_reset_requested[env_id] != 0
    ):
        hit, height = surface_height(background_mesh, x, y, ray_z, ray_length)
        if not hit:
            wp.atomic_add(terrain_query_miss_count, 0, 1)
        particle_reset_position = wp.vec3(x, y, height + particle_template_position[2])
    is_dynamic = int(
        wp.abs(x - patch_center[0]) < 0.5 * simulated_terrain_size[0]
        and wp.abs(y - patch_center[1]) < 0.5 * simulated_terrain_size[1]
    )
    reset_position_changed = wp.length(particle_reset_position - particle_reset_positions[particle_id]) > 1.0e-5
    escaped_patch = (
        wp.abs(particle_q[particle_id][0] - patch_center[0]) >= 0.5 * total_patch_size[0]
        or wp.abs(particle_q[particle_id][1] - patch_center[1]) >= 0.5 * total_patch_size[1]
    )
    if (
        reset_position_changed
        or escaped_patch
        or is_dynamic != particle_is_dynamic[particle_id]
        or is_dynamic == 0
        or env_reset_requested[env_id] != 0
    ):
        particle_q[particle_id] = particle_reset_position
        particle_qd[particle_id] = wp.vec3(0.0)
        particle_qd_grad[particle_id] = wp.mat33(0.0)
        particle_elastic_strain[particle_id] = wp.identity(n=3, dtype=float)
        particle_plastic_volume_ratio[particle_id] = initial_particle_plastic_volume_ratio[particle_id]
        particle_stress[particle_id] = wp.mat33(0.0)
        particle_transform[particle_id] = wp.identity(n=3, dtype=float)
    target_particle_mass = float(is_dynamic) * particle_dynamic_mass[particle_id]
    if particle_mass[particle_id] != target_particle_mass:
        wp.atomic_add(mass_change_count, 0, 1)
    particle_mass[particle_id] = target_particle_mass
    particle_inv_mass[particle_id] = 0.0
    if is_dynamic != 0:
        particle_inv_mass[particle_id] = 1.0 / particle_dynamic_mass[particle_id]
    particle_reset_positions[particle_id] = particle_reset_position
    particle_is_dynamic[particle_id] = is_dynamic


@wp.kernel
def restore_boundary_particles(
    particle_is_dynamic: wp.array[int],
    particle_reset_positions: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
):
    """Restore boundary particles to their reset positions [m] and set velocities [m/s] to zero."""
    particle_id = wp.tid()
    if particle_is_dynamic[particle_id] == 0:
        particle_q[particle_id] = particle_reset_positions[particle_id]
        particle_qd[particle_id] = wp.vec3(0.0)


@wp.kernel
def refresh_density(
    particle_mass: wp.array[float], particle_volume: wp.array[float], particle_density: wp.array[float]
):
    """Update particle densities [kg/m^3] from masses [kg] and fixed reference volumes [m^3]."""
    particle_id = wp.tid()
    particle_density[particle_id] = particle_mass[particle_id] / particle_volume[particle_id]


@wp.kernel
def mark_visible_particles(
    particle_env_ids: wp.array[int],
    particle_is_dynamic: wp.array[int],
    env_visibility_mask: wp.array[int],
    filter_envs: bool,
    show_boundary_particles: bool,
    particle_visibility_mask: wp.array[int],
):
    """Select particles by environment and simulated/boundary membership without changing physics flags."""
    particle_id = wp.tid()
    is_visible = show_boundary_particles or particle_is_dynamic[particle_id] != 0
    if filter_envs:
        env_id = particle_env_ids[particle_id]
        if env_id >= 0 and env_id < env_visibility_mask.shape[0]:
            is_visible = is_visible and env_visibility_mask[env_id] != 0
        else:
            is_visible = False
    particle_visibility_mask[particle_id] = wp.where(is_visible, 1, 0)


@wp.kernel
def gather_visible_particles(
    particle_visibility_mask: wp.array[int],
    particle_visible_offsets: wp.array[int],
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    visible_particle_q: wp.array[wp.vec3],
    visible_particle_radius: wp.array[float],
):
    """Compact visible particle positions [m] and radii [m] using inclusive prefix-scan offsets."""
    particle_id = wp.tid()
    if particle_visibility_mask[particle_id] != 0:
        visible_particle_id = particle_visible_offsets[particle_id] - 1
        visible_particle_q[visible_particle_id] = particle_q[particle_id]
        visible_particle_radius[visible_particle_id] = particle_radius[particle_id]
