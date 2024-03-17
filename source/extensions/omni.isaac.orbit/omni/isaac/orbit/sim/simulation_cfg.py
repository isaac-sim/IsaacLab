# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration of the environment.

This module defines the general configuration of the environment. It includes parameters for
configuring the environment instances, viewer settings, and simulation parameters.
"""

from __future__ import annotations

from typing import Literal

from omni.isaac.orbit.utils import configclass

from .spawners.materials import RigidBodyMaterialCfg


@configclass
class PhysxCfg:
    """Configuration for PhysX solver-related parameters.

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

    solver_type: Literal[0, 1] = 1
    """The type of solver to use.Default is 1 (TGS).

    Available solvers:

    * :obj:`0`: PGS (Projective Gauss-Seidel)
    * :obj:`1`: TGS (Truncated Gauss-Seidel)
    """

    min_position_iteration_count: int = 1
    """Minimum number of solver position iterations (rigid bodies, cloth, particles etc.). Default is 1.

    .. note::

        Each physics actor in Omniverse specifies its own solver iteration count. The solver takes
        the number of iterations specified by the actor with the highest iteration and clamps it to
        the range ``[min_position_iteration_count, max_position_iteration_count]``.
    """

    max_position_iteration_count: int = 255
    """Maximum number of solver position iterations (rigid bodies, cloth, particles etc.). Default is 255.

    .. note::

        Each physics actor in Omniverse specifies its own solver iteration count. The solver takes
        the number of iterations specified by the actor with the highest iteration and clamps it to
        the range ``[min_position_iteration_count, max_position_iteration_count]``.
    """

    min_velocity_iteration_count: int = 0
    """Minimum number of solver position iterations (rigid bodies, cloth, particles etc.). Default is 0.

    .. note::

        Each physics actor in Omniverse specifies its own solver iteration count. The solver takes
        the number of iterations specified by the actor with the highest iteration and clamps it to
        the range ``[min_velocity_iteration_count, max_velocity_iteration_count]``.
    """

    max_velocity_iteration_count: int = 255
    """Maximum number of solver position iterations (rigid bodies, cloth, particles etc.). Default is 255.

    .. note::

        Each physics actor in Omniverse specifies its own solver iteration count. The solver takes
        the number of iterations specified by the actor with the highest iteration and clamps it to
        the range ``[min_velocity_iteration_count, max_velocity_iteration_count]``.
    """

    enable_ccd: bool = False
    """Enable a second broad-phase pass that makes it possible to prevent objects from tunneling through each other.
    Default is False."""

    enable_stabilization: bool = True
    """Enable/disable additional stabilization pass in solver. Default is True."""

    enable_enhanced_determinism: bool = False
    """Enable/disable improved determinism at the expense of performance. Defaults to False.

    For more information on PhysX determinism, please check `here`_.

    .. _here: https://nvidia-omniverse.github.io/PhysX/physx/5.3.1/docs/RigidBodyDynamics.html#enhanced-determinism
    """

    bounce_threshold_velocity: float = 0.5
    """Relative velocity threshold for contacts to bounce (in m/s). Default is 0.5 m/s."""

    friction_offset_threshold: float = 0.04
    """Threshold for contact point to experience friction force (in m). Default is 0.04 m."""

    friction_correlation_distance: float = 0.025
    """Distance threshold for merging contacts into a single friction anchor point (in m). Default is 0.025 m."""

    gpu_max_rigid_contact_count: int = 2**23
    """Size of rigid contact stream buffer allocated in pinned host memory. Default is 2 ** 23."""

    gpu_max_rigid_patch_count: int = 5 * 2**15
    """Size of the rigid contact patch stream buffer allocated in pinned host memory. Default is 5 * 2 ** 15."""

    gpu_found_lost_pairs_capacity: int = 2**21
    """Capacity of found and lost buffers allocated in GPU global memory. Default is 2 ** 21.

    This is used for the found/lost pair reports in the BP.
    """

    gpu_found_lost_aggregate_pairs_capacity: int = 2**25
    """Capacity of found and lost buffers in aggregate system allocated in GPU global memory.
    Default is 2 ** 25.

    This is used for the found/lost pair reports in AABB manager.
    """

    gpu_total_aggregate_pairs_capacity: int = 2**21
    """Capacity of total number of aggregate pairs allocated in GPU global memory. Default is 2 ** 21."""

    gpu_collision_stack_size: int = 2**26
    """Size of the collision stack buffer allocated in pinned host memory. Default is 2 ** 26."""

    gpu_heap_capacity: int = 2**26
    """Initial capacity of the GPU and pinned host memory heaps. Additional memory will be allocated
    if more memory is required. Default is 2 ** 26."""

    gpu_temp_buffer_capacity: int = 2**24
    """Capacity of temp buffer allocated in pinned host memory. Default is 2 ** 24."""

    gpu_max_num_partitions: int = 8
    """Limitation for the partitions in the GPU dynamics pipeline. Default is 8.

    This variable must be power of 2. A value greater than 32 is currently not supported. Range: (1, 32)
    """

    gpu_max_soft_body_contacts: int = 2**20
    """Size of soft body contacts stream buffer allocated in pinned host memory. Default is 2 ** 20."""

    gpu_max_particle_contacts: int = 2**20
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

    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    """The gravity vector (in m/s^2). Default is (0.0, 0.0, -9.81).

    If set to (0.0, 0.0, 0.0), gravity is disabled.
    """

    enable_scene_query_support: bool = False
    """Enable/disable scene query support for collision shapes. Default is False.

    This flag allows performing collision queries (raycasts, sweeps, and overlaps) on actors and
    attached shapes in the scene. This is useful for implementing custom collision detection logic
    outside of the physics engine.

    If set to False, the physics engine does not create the scene query manager and the scene query
    functionality will not be available. However, this provides some performance speed-up.

    Note:
        This flag is overridden to True inside the :class:`SimulationContext` class when running the simulation
        with the GUI enabled. This is to allow certain GUI features to work properly.
    """

    use_fabric: bool = True
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

    .. note::

        It is required to set this flag to :obj:`True` when using the TensorAPIs for contact reporting.
    """

    use_gpu_pipeline: bool = True
    """Enable/disable GPU pipeline. Default is True.

    If set to False, the physics data will be read as CPU buffers.
    """

    device: str = "cuda:0"
    """The device for running the simulation/environment. Default is ``"cuda:0"``."""

    physx: PhysxCfg = PhysxCfg()
    """PhysX solver settings. Default is PhysxCfg()."""

    physics_material: RigidBodyMaterialCfg = RigidBodyMaterialCfg()
    """Default physics material settings for rigid bodies. Default is RigidBodyMaterialCfg().

    The physics engine defaults to this physics material for all the rigid body prims that do not have any
    physics material specified on them.

    The material is created at the path: ``{physics_prim_path}/defaultMaterial``.
    """
