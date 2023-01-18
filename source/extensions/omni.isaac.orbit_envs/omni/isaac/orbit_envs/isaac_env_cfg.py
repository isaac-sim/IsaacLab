# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration of the environment.

This module defines the general configuration of the environment. It includes parameters for
configuring the environment instances, viewer settings, and simulation parameters.
"""

from dataclasses import MISSING
from typing import Tuple

from omni.isaac.orbit.utils import configclass

__all__ = ["IsaacEnvCfg", "EnvCfg", "ViewerCfg", "PhysxCfg", "SimCfg"]


##
# General environment configuration
##


@configclass
class EnvCfg:
    """Configuration of the common environment information."""

    num_envs: int = MISSING
    """Number of environment instances to create."""
    env_spacing: float = MISSING
    """Spacing between cloned environments."""
    episode_length_s: float = None
    """Duration of an episode (in seconds). Default is None (no limit)."""


@configclass
class ViewerCfg:
    """Configuration of the scene viewport camera."""

    debug_vis: bool = False
    """Whether to enable/disable debug visualization in the scene."""
    eye: Tuple[float, float, float] = (7.5, 7.5, 7.5)
    """Initial camera position (in m). Default is (7.5, 7.5, 7.5)."""
    lookat: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Initial camera target position (in m). Default is (0.0, 0.0, 0.0)."""


##
# Simulation settings
##


@configclass
class PhysxCfg:
    """PhysX solver parameters.

    These parameters are used to configure the PhysX solver. For more information, see the PhysX 5 SDK
    documentation.

    PhysX 5 supports GPU-accelerated physics simulation. This is enabled by default, but can be disabled
    through the flag `use_gpu`. Unlike CPU PhysX, the GPU simulation feature is not able to dynamically
    grow all the buffers. Therefore, it is necessary to provide a reasonable estimate of the buffer sizes
    for GPU features. If insufficient buffer sizes are provided, the simulation will fail with errors and
    lead to adverse behaviors. The buffer sizes can be adjusted through the `gpu_*` parameters.

    References:
        * PhysX 5 documentation: https://nvidia-omniverse.github.io/PhysX/
    """

    use_gpu: bool = True
    """Enable/disable GPU accelerated dynamics simulation. Default is True.

    This enables GPU-accelerated implementations for broad-phase collision checks, contact generation,
    shape and body management, and constrained solver.
    """

    solver_type: int = 1
    """The type of solver to use.Default is 1 (TGS).

    Available solvers:

    * :obj:`0`: PGS (Projective Gauss-Seidel)
    * :obj:`1`: TGS (Truncated Gauss-Seidel)
    """

    enable_stabilization: bool = True
    """Enable/disable additional stabilization pass in solver. Default is True."""

    bounce_threshold_velocity: float = 0.5
    """Relative velocity threshold for contacts to bounce (in m/s). Default is 0.5 m/s."""

    friction_offset_threshold: float = 0.04
    """Threshold for contact point to experience friction force (in m). Default is 0.04 m."""

    friction_correlation_distance: float = 0.025
    """Distance threshold for merging contacts into a single friction anchor point (in m). Default is 0.025 m."""

    gpu_max_rigid_contact_count: int = 1024 * 1024
    """Size of rigid contact stream buffer allocated in pinned host memory. Default is 2 ** 20."""

    gpu_max_rigid_patch_count: int = 80 * 1024 * 2
    """Size of the rigid contact patch stream buffer allocated in pinned host memory. Default is 80 * 2 ** 11."""

    gpu_found_lost_pairs_capacity: int = 1024 * 2
    """Capacity of found and lost buffers allocated in GPU global memory. Default is 2 ** 11.

    This is used for the found/lost pair reports in the BP.
    """

    gpu_found_lost_aggregate_pairs_capacity: int = 1024 * 1024 * 32
    """Capacity of found and lost buffers in aggregate system allocated in GPU global memory.
    Default is 2 ** 21.

    This is used for the found/lost pair reports in AABB manager.
    """

    gpu_total_aggregate_pairs_capacity: int = 1024 * 1024 * 2
    """Capacity of total number of aggregate pairs allocated in GPU global memory. Default is 2 ** 21."""

    gpu_heap_capacity: int = 64 * 1024 * 1024
    """Initial capacity of the GPU and pinned host memory heaps. Additional memory will be allocated
    if more memory is required. Default is 2 ** 26."""

    gpu_temp_buffer_capacity: int = 16 * 1024 * 1024
    """Capacity of temp buffer allocated in pinned host memory. Default is 2 ** 24."""

    gpu_max_num_partitions: int = 8
    """Limitation for the partitions in the GPU dynamics pipeline. Default is 8.

    This variable must be power of 2. A value greater than 32 is currently not supported. Range: (1, 32)
    """

    gpu_max_soft_body_contacts: int = 1024 * 1024
    """Size of soft body contacts stream buffer allocated in pinned host memory. Default is 2 ** 20."""

    gpu_max_particle_contacts: int = 1024 * 1024
    """Size of particle contacts stream buffer allocated in pinned host memory. Default is 2 ** 20."""


@configclass
class SimCfg:
    """Configuration for simulation physics."""

    dt: float = 1.0 / 60.0
    """The physics simulation time-step (in seconds). Default is 0.0167 seconds."""

    substeps: int = 1
    """The number of physics simulation steps per rendering step. Default is 1."""

    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)
    """The gravity vector (in m/s^2). Default is (0.0, 0.0, -9.81)."""

    enable_scene_query_support: bool = True
    """Enable/disable scene query support when using instanced assets. Default is True.

    If this is set to False, the geometries of instances assets will appear stationary. However, this
    can also provide some performance speed-up.
    """

    use_flatcache: bool = True  # output from simulation to flat cache
    """Enable/disable reading of physics buffers directly. Default is True.

    If this is set to False, the physics buffers will be read from USD, which leads to overhead with
    massive parallelization.
    """

    use_gpu_pipeline: bool = True
    """Enable/disable GPU pipeline. Default is True.

    If this is set to False, the physics data will be read as CPU buffers.
    """

    device: str = "cuda:0"
    """The device for running the simulation/environment. Default is "cuda:0"."""

    physx: PhysxCfg = PhysxCfg()
    """PhysX solver settings. Default is PhysxCfg()."""


##
# Environment configuration
##


@configclass
class IsaacEnvCfg:
    """Base configuration of the environment."""

    env: EnvCfg = MISSING
    """General environment configuration."""
    viewer: ViewerCfg = ViewerCfg()
    """Viewer configuration. Default is ViewerCfg()."""
    sim: SimCfg = SimCfg()
    """Physics simulation configuration. Default is SimCfg()."""
