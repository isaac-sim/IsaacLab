# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration of the environment.

This module defines the general configuration of the environment. It includes parameters for
configuring the environment instances, viewer settings, and simulation parameters.
"""

from typing import Tuple

from omni.isaac.orbit.utils import configclass

__all__ = ["PhysicsMaterialCfg", "PhysxCfg", "SimulationCfg"]


##
# Simulation settings
##


@configclass
class PhysicsMaterialCfg:
    """Physics material parameters."""

    static_friction: float = 1.0
    """The static friction coefficient. Defaults to 1.0."""

    dynamic_friction: float = 1.0
    """The dynamic friction coefficient. Defaults to 1.0."""

    restitution: float = 0.0
    """The restitution coefficient. Defaults to 0.0."""

    improve_patch_friction: bool = False
    """Whether to enable patch friction. Defaults to False."""

    combine_mode: str = "average"
    """Determines the way physics materials will be combined during collisions. Defaults to `average`.

    This includes for both friction and restitution combination.

    Available options are `average`, `min`, `multiply`, `multiply`, and `max`.
    """


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

    gpu_found_lost_pairs_capacity: int = 1024 * 1024 * 2
    """Capacity of found and lost buffers allocated in GPU global memory. Default is 2 ** 21.

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
class SimulationCfg:
    """Configuration for simulation physics."""

    physics_prim_path: str = "/physicsScene"
    """The prim path where the USD PhysicsScene is created. Default is "/physicsScene"."""

    dt: float = 1.0 / 60.0
    """The physics simulation time-step (in seconds). Default is 0.0167 seconds."""

    substeps: int = 1
    """The number of physics simulation steps per rendering step. Default is 1."""

    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)
    """The gravity vector (in m/s^2). Default is (0.0, 0.0, -9.81)."""

    enable_scene_query_support: bool = False
    """Enable/disable scene query support for collision shapes. Default is False.

    This flag allows performing collision queries (raycasts, sweeps, and overlaps) on actors and
    attached shapes in the scene. This is useful for implementing custom collision detection logic
    outside of the physics engine.

    If set to False, the physics engine does not create the scene query manager and the scene query
    functionality will not be available. However, this provides some performance speed-up.

    Note:
        This flag is overridden to True inside the :class:`IsaacEnv` class when running the simulation
        with the GUI enabled. This is to allow certain GUI features to work properly.
    """

    use_flatcache: bool = True
    """Enable/disable reading of physics buffers directly. Default is True.

    When running the simulation, updates in the states in the scene is normally synchronized with USD.
    This leads to an overhead in reading the data and does not scale well with massive parallelization.
    This flag allows disabling the synchronization and reading the data directly from the physics buffers.

    It is recommended to set this flag to :obj:`True` when running the simulation with a large number
    of primitives in the scene.

    Note:
        When enabled, the GUI will not update the physics parameters in real-time. To enable real-time
        updates, please set this flag to :obj:`False`.
    """

    disable_contact_processing: bool = False
    """Enable/disable contact processing. Default is False.

    By default, the physics engine processes all the contacts in the scene. However, reporting this contact
    information can be expensive due to its combinatorial complexity. This flag allows disabling the contact
    processing and querying the contacts manually by the user over a limited set of primitives in the scene.

    It is recommended to set this flag to :obj:`True` when using the TensorAPIs for contact reporting.
    """

    use_gpu_pipeline: bool = True
    """Enable/disable GPU pipeline. Default is True.

    If set to False, the physics data will be read as CPU buffers.
    """

    device: str = "cuda:0"
    """The device for running the simulation/environment. Default is "cuda:0"."""

    physx: PhysxCfg = PhysxCfg()
    """PhysX solver settings. Default is PhysxCfg()."""

    default_physics_material: PhysicsMaterialCfg = PhysicsMaterialCfg()
    """Default physics material settings. Default is PhysicsMaterialCfg().

    The physics engine defaults to this physics material for all the rigid body prims that do not have any
    physics material specified on them.

    The material is created at the path: ``{physics_prim_path}/defaultMaterial``.
    """
