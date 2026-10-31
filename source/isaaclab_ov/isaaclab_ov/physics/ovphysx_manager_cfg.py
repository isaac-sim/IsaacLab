# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the OvPhysX physics manager."""

from __future__ import annotations

import os
import tempfile
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.physics import PhysicsCfg
from isaaclab.sim import BackendCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .ovphysx_manager import OvPhysxBackend

# POSIX temp roots are shared between users; Windows already gives each user a private one.
_CACHE_DIR_NAME = f"ovphysx_derived_data_cache_{os.getuid()}" if hasattr(os, "getuid") else "ovphysx_derived_data_cache"

DEFAULT_COOKED_COLLIDER_CACHE_DIR: str = os.path.join(tempfile.gettempdir(), _CACHE_DIR_NAME)
"""Per-user cooked-collider cache shared by physics and native runtime configuration defaults."""


@configclass
class OvPhysxBackendCfg(BackendCfg):
    """Native runtime construction settings, separate from physics scene policy."""

    class_type: type[OvPhysxBackend] | str = "{DIR}.ovphysx_manager:OvPhysxBackend"

    device: str = MISSING
    """Normalized device: ``cpu`` or ``cuda:<index>``."""

    cooked_collider_cache_dir: str | None = DEFAULT_COOKED_COLLIDER_CACHE_DIR
    """Cooked-collider cache location, fixed by the first runtime created in the process."""


@configclass
class OvPhysxCfg(PhysicsCfg):
    """Configuration for the ovphysx physics manager.

    PhysX scene-level parameters that this config does not set, such as solver iterations,
    are read from the USD PhysicsScene prim. Before loading, the manager authors the scene
    settings held here onto that prim: determinism, external forces, GPU buffer sizes (GPU
    only) and the optional contact parameters :attr:`bounce_threshold_velocity`,
    :attr:`friction_correlation_distance` and :attr:`solve_articulation_contact_last`. A
    contact parameter left at ``None`` keeps the stage value, or the ``PhysxSceneAPI``
    schema default when the stage does not author it.
    """

    class_type = "{DIR}.ovphysx_manager:OvPhysxManager"

    legacy_joint_friction_replacements: dict[str, tuple[float, float, float]] = {}
    """Explicit OVPhysX 0.7 migration values for legacy USD joint friction.

    Keys are regular expressions matched against the complete joint name. Values are
    ``(static effort, dynamic effort, viscous coefficient)`` in SI units: N or N*m,
    and N*s/m or N*m*s/rad. These are calibrated replacement efforts, not converted
    legacy coefficients. Static effort must be at least dynamic effort.

    Only the private OVPhysX 0.7 export is changed. Older runtimes ignore this mapping.
    Authored modern friction takes precedence. Unmatched nonzero legacy friction still
    raises an error; this mapping does not provide a global fallback for unknown assets.
    """

    cooked_collider_cache_dir: str | None = DEFAULT_COOKED_COLLIDER_CACHE_DIR
    """Directory for the OVPhysX cooked-collider cache, defaulting to a per-user directory under the
    system temporary directory. Set to ``None`` to use the runtime default.

    The datastore is created once per process, so the first OVPhysX construction fixes the location and a
    later config does not relocate it. The runtime garbage collects the cache against a disk budget;
    deleting the directory reclaims the space immediately, and a system temporary directory may also be
    purged between boots.
    """

    enable_enhanced_determinism: bool = False
    """Enable/disable improved determinism at the expense of performance. Defaults to False.

    For more information on PhysX determinism, please check `here`_.

    .. _here: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/RigidBodyDynamics.html#enhanced-determinism
    """

    enable_external_forces_every_iteration: bool = True
    """Enable/disable external forces every position iteration in the TGS solver. Default is True.

    This matches the PhysX backend default and can improve the accuracy of velocity updates.
    Set to ``False`` explicitly to preserve the previous OvPhysX default behavior. Changing this
    setting can affect simulated velocities and contact forces.
    """

    bounce_threshold_velocity: float | None = None
    """Relative contact velocity below which contacts do not bounce [m/s].

    ``None`` leaves the stage value. The ``PhysxSceneAPI`` schema default is 0, which sets no speed
    threshold, so contact restitution alone decides whether a contact bounces.
    Authored as ``physxScene:bounceThreshold``.
    """

    friction_correlation_distance: float | None = None
    """Distance below which contact points merge into one friction anchor [m].

    ``None`` leaves the stage value. The ``PhysxSceneAPI`` schema default is ``0.025 * tolerancesLength``,
    which is 0.025 m on a meter-based stage. Authored as ``physxScene:frictionCorrelationDistance``.
    """

    solve_articulation_contact_last: bool | None = None
    """Whether the solver resolves articulation contacts after other constraints.

    Enable it so object contacts can stall position-driven grippers before they tunnel through a
    grasp. ``None`` leaves the stage value. The ``PhysxSceneAPI`` schema default is off. Authored as
    ``physxScene:solveArticulationContactLast``.
    """

    gpu_max_rigid_contact_count: int = 2**23
    """Size of the GPU rigid-body contact buffer."""

    gpu_max_rigid_patch_count: int = 5 * 2**15
    """Size of the GPU rigid-body patch buffer."""

    gpu_found_lost_pairs_capacity: int = 2**21
    """Capacity for GPU found/lost broadphase pairs."""

    gpu_found_lost_aggregate_pairs_capacity: int = 2**25
    """Capacity for GPU found/lost *aggregate* broadphase pairs."""

    gpu_total_aggregate_pairs_capacity: int = 2**21
    """Capacity for total GPU aggregate broadphase pairs."""

    gpu_collision_stack_size: int = 2**26
    """GPU collision stack size in bytes."""
