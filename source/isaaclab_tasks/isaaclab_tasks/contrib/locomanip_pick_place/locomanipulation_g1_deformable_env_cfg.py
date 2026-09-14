# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 locomanipulation pick-and-place with a deformable object, on Newton."""

from __future__ import annotations

from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
    NewtonSoftContactCfg,
    VBDSolverCfg,
)
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import NewtonDeformableBodyMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.deformable_object import DeformableObjectCfg
from isaaclab.utils.configclass import configclass

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg

from isaaclab_tasks.utils import PresetCfg

from .locomanipulation_g1_env_cfg import (
    _PACKING_TABLE_COLLIDER_POS,
    _PACKING_TABLE_COLLIDER_SIZE,
    LocomanipulationG1EnvCfg,
    LocomanipulationG1SceneCfg,
)

_OBJECT_SIZE = (0.06, 0.06, 0.06)
"""Edge lengths of the deformable object [m]."""

_OBJECT_POS = (-0.35, 0.45, 0.7261)
"""Start position of the deformable object [m].

The tabletop collider's top face is at z = 0.6941, so this clears it by 2 mm. A soft body
spawned intersecting the table is ejected rather than pushed out as a rigid body would be.
"""

_YOUNGS_MODULUS = 2.0e5
"""Young's modulus of the deformable object [Pa]."""

_POISSONS_RATIO = 0.3
"""Poisson's ratio of the deformable object."""

_HAND_PROXY_BODIES = [r"/World/envs/env_[^/]+/Robot/(left|right)_hand/.*_link"]
"""Robot bodies exposed to the deformable solver. Only the grasping links belong here."""

_GROUND_SHAPE = r"/World/GroundPlane.*"
"""Static shape owned by the rigid entry, so the robot keeps its ground contact."""

_TABLE_SHAPE_RIGID = r"/World/envs/env_[^/]+/PackingTableCollider.*"
"""Tabletop shape owned by the rigid entry, so the robot cannot reach through the table."""

_TABLE_SHAPE_SOFT = r"/World/envs/env_[^/]+/SoftPackingTableCollider.*"
"""Tabletop shape owned by the soft entry, so the deformable rests on the table.

A shape belongs to at most one entry, so the rigid and soft solvers cannot share one tabletop.
This is a second collider coincident with :data:`_TABLE_SHAPE_RIGID`.
"""


@configclass
class PhysicsCfg(PresetCfg):
    """Physics presets for the deformable variant.

    MJWarp has no deformable support, so the scene is partitioned between two solvers: a ``rigid``
    MJWarp entry owning the whole G1 articulation and a ``soft`` VBD entry owning the object's
    particles. Only the hand links are exposed as proxies, so VBD sees the grasping geometry
    without the rest of the robot. The rigid entry keeps the locomotion profile from the rigid
    task, with a larger constraint budget to cover the deformable contacts.
    """

    newton_mjwarp_vbd_proxy = NewtonCfg(
        solver_cfg=CouplerProxyCfg(
            entries=[
                CouplerEntryCfg(
                    name="rigid",
                    shape_label_patterns=[_GROUND_SHAPE, _TABLE_SHAPE_RIGID],
                    solver_cfg=MJWarpSolverCfg(
                        solver="newton",
                        integrator="implicitfast",
                        njmax=600,
                        nconmax=400,
                        impratio=1.0,
                        cone="pyramidal",
                        update_data_interval=2,
                        iterations=100,
                        ls_iterations=15,
                        ls_parallel=False,
                        use_mujoco_contacts=False,
                    ),
                    bodies=[r"/World/envs/env_[^/]+/Robot"],
                ),
                CouplerEntryCfg(
                    name="soft",
                    solver_cfg=VBDSolverCfg(iterations=10, rigid_body_particle_contact_buffer_size=256),
                    all_particles=True,
                    shape_label_patterns=[_TABLE_SHAPE_SOFT],
                ),
            ],
            proxies=[
                CouplerProxyMappingCfg(
                    source="rigid",
                    destination="soft",
                    bodies=_HAND_PROXY_BODIES,
                    collide_interval=1,
                    # Full-surface contact samples each rigid shape's SDF, and the G1's hand
                    # colliders are meshes with none, so contact falls back to per-vertex.
                    collision_pipeline=NewtonCollisionPipelineCfg(
                        enable_rigid_soft_full_surface_contact=False,
                    ),
                )
            ],
            iterations=1,
        ),
        # The Franka soft-lift preset carries its object with ``soft_contact_mu=10``, which the
        # docs call an unphysical value. A fixed-base arm absorbs the resulting tangential force
        # into its mount; a free-standing humanoid is torqued over by it, so friction stays
        # physical here. Raise it if the object slips, but expect balance to degrade.
        soft_contact_cfg=NewtonSoftContactCfg(
            soft_contact_ke=8.0e3,
            soft_contact_kd=1.0e-2,
            soft_contact_mu=1.0,
        ),
        # A humanoid's weight on two feet sinks into Newton's default ``ke=2.5e3``; see the rigid
        # task's preset.
        default_shape_cfg=NewtonShapeCfg(margin=0.0, ke=160000.0, kd=1100.0),
        num_substeps=2,
    )

    default = newton_mjwarp_vbd_proxy


@configclass
class LocomanipulationG1DeformableSceneCfg(LocomanipulationG1SceneCfg):
    """Scene with the rigid graspable object replaced by a deformable one."""

    object: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=_OBJECT_POS),
        spawn=sim_utils.MeshCuboidCfg(
            size=_OBJECT_SIZE,
            edge_refinement=3.0,
            deformable_props=NewtonDeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.6, 0.2)),
            physics_material=NewtonDeformableBodyMaterialCfg(
                density=1000.0,
                k_mu=_YOUNGS_MODULUS / (2.0 * (1.0 + _POISSONS_RATIO)),
                k_lambda=(
                    _YOUNGS_MODULUS * _POISSONS_RATIO / ((1.0 + _POISSONS_RATIO) * (1.0 - 2.0 * _POISSONS_RATIO))
                ),
                particle_radius=0.004,
            ),
        ),
    )

    # Coincident twin of ``packing_table_collider``, owned by the soft solver entry.
    soft_packing_table_collider: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SoftPackingTableCollider",
        init_state=AssetBaseCfg.InitialStateCfg(pos=list(_PACKING_TABLE_COLLIDER_POS)),
        spawn=sim_utils.CuboidCfg(
            size=_PACKING_TABLE_COLLIDER_SIZE,
            visible=False,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )

    # Coupled solvers hold contact forces in per-entry buffers and reject contact sensors outright.
    left_hand_contact = None
    right_hand_contact = None

    def __post_init__(self):
        super().__post_init__()
        # The rigid task gates this collider on the ``newton_mjwarp`` preset name, which this
        # task does not use. Without it Newton emits no tabletop shape and the object falls through.
        self.packing_table_collider.spawn.collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=True)


@configclass
class LocomanipulationG1DeformableEnvCfg(LocomanipulationG1EnvCfg):
    """G1 locomanipulation with a deformable object, Newton only."""

    scene: LocomanipulationG1DeformableSceneCfg = LocomanipulationG1DeformableSceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=True
    )

    def __post_init__(self):
        super().__post_init__()
        self.sim.physics = PhysicsCfg()

        # A deformable has no single rigid pose: its ``root_pos_w`` is the mean of the nodal
        # positions and it exposes no orientation, so the orientation-dependent terms go.
        self.observations.policy.object_rot = None
        self.observations.policy.object = None

        # Haptics read the per-hand contact sensors, which a coupled solver cannot provide.
        self.scene.robot.spawn.activate_contact_sensors = False
        self.haptic_feedback = None
