# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for PhysX physics manager."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from isaaclab.physics import PhysicsCfg
from isaaclab.utils import configclass

from .physx_manager import PhysxManager

if TYPE_CHECKING:
    from isaaclab.physics import PhysicsManager


@configclass
class PhysxCfg(PhysicsCfg):
    """Configuration for PhysX physics manager.

    This configuration includes all PhysX-specific settings including solver
    parameters, scene configuration, and GPU buffer sizes. For more information,
    see the `PhysX 5 SDK documentation`_.

    PhysX 5 supports GPU-accelerated physics simulation. This is enabled by default,
    but can be disabled by setting :attr:`device` to ``cpu``. Unlike CPU PhysX, the
    GPU simulation feature is unable to dynamically grow all the buffers. Therefore,
    it is necessary to provide a reasonable estimate of the buffer sizes for GPU
    features. If insufficient buffer sizes are provided, the simulation will fail
    with errors and lead to adverse behaviors. The buffer sizes can be adjusted
    through the ``gpu_*`` parameters.

    .. _PhysX 5 SDK documentation: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/classPxSceneDesc.html
    """

    # ------------------------------------------------------------------
    # PhysX Scene Settings
    # ------------------------------------------------------------------

    class_type: type[PhysicsManager] = PhysxManager
    """The class type of the PhysxManager."""

    # ------------------------------------------------------------------
    # Solver Settings
    # ------------------------------------------------------------------

    solver_type: Literal[0, 1] = 1
    """The type of solver to use. Default is 1 (TGS).

    Available solvers:

    * :obj:`0`: PGS (Projective Gauss-Seidel)
    * :obj:`1`: TGS (Temporal Gauss-Seidel)
    """

    solve_articulation_contact_last: bool = False
    """Changes the ordering inside the articulation solver. Default is False.

    PhysX employs a strict ordering for handling constraints in an articulation. The outcome of
    each constraint resolution modifies the joint and associated link speeds. However, the default
    ordering may not be ideal for gripping scenarios because the solver favours the constraint
    types that are resolved last. This is particularly true of stiff constraint systems that are hard
    to resolve without resorting to vanishingly small simulation timesteps.

    With dynamic contact resolution being such an important part of gripping, it may make
    more sense to solve dynamic contact towards the end of the solver rather than at the
    beginning. This parameter modifies the default ordering to enable this change.

    For more information, please check `here <https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/articulation_stability_guide.html#articulation-solver-order>`__.

    .. versionadded:: v2.3
        This parameter is only available with Isaac Sim 5.1.
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
    """Minimum number of solver velocity iterations (rigid bodies, cloth, particles etc.). Default is 0.

    .. note::

        Each physics actor in Omniverse specifies its own solver iteration count. The solver takes
        the number of iterations specified by the actor with the highest iteration and clamps it to
        the range ``[min_velocity_iteration_count, max_velocity_iteration_count]``.
    """

    max_velocity_iteration_count: int = 255
    """Maximum number of solver velocity iterations (rigid bodies, cloth, particles etc.). Default is 255.

    .. note::

        Each physics actor in Omniverse specifies its own solver iteration count. The solver takes
        the number of iterations specified by the actor with the highest iteration and clamps it to
        the range ``[min_velocity_iteration_count, max_velocity_iteration_count]``.
    """

    enable_ccd: bool = False
    """Enable a second broad-phase pass that makes it possible to prevent objects from tunneling through each other.
    Default is False."""

    enable_stabilization: bool = False
    """Enable/disable additional stabilization pass in solver. Default is False.

    .. note::

        We recommend setting this flag to true only when the simulation step size is large
        (i.e., less than 30 Hz or more than 0.0333 seconds).

    .. warning::

        Enabling this flag may lead to incorrect contact forces report from the contact sensor.
    """

    enable_external_forces_every_iteration: bool = False
    """Enable/disable external forces every position iteration in the TGS solver. Default is False.

    When using the TGS solver (:attr:`solver_type` is 1), this flag allows enabling external forces
    every solver position iteration. This can help improve the accuracy of velocity updates.
    Consider enabling this flag if the velocities generated by the simulation are noisy.
    Increasing the number of velocity iterations, together with this flag, can help improve
    the accuracy of velocity updates.

    .. note::

        This flag is ignored when using the PGS solver (:attr:`solver_type` is 0).
    """

    enable_enhanced_determinism: bool = False
    """Enable/disable improved determinism at the expense of performance. Defaults to False.

    For more information on PhysX determinism, please check `here`_.

    .. _here: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/RigidBodyDynamics.html#enhanced-determinism
    """

    bounce_threshold_velocity: float = 0.5
    """Relative velocity threshold for contacts to bounce (in m/s). Default is 0.5 m/s."""

    friction_offset_threshold: float = 0.04
    """Threshold for contact point to experience friction force (in m). Default is 0.04 m."""

    friction_correlation_distance: float = 0.025
    """Distance threshold for merging contacts into a single friction anchor point (in m). Default is 0.025 m."""

    # ------------------------------------------------------------------
    # GPU Buffer Settings
    # ------------------------------------------------------------------

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
