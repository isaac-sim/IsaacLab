.. _migrating-deformables:

Migration of Deformables
========================

.. currentmodule:: isaaclab

In the newer versions of Omni Physics (107.0 and later), the old deformable body functionality has become deprecated.
The following sections describe the changes to migrate to the new Omni Physics API, specifically moving away from
Soft Bodies and towards Surface and Volume Deformables. The deformable object asset classes remain in
``isaaclab.assets``. Schema define/modify functions remain unified in ``isaaclab.sim.schemas``, and deformable
material spawning remains unified in ``isaaclab.sim.spawners.materials``. Deformable property and material
configuration classes are backend-specific: PhysX configurations live in ``isaaclab_physx.sim`` and Newton
configurations live in ``isaaclab_newton.sim``.

.. note::

  The following changes are with respect to Isaac Lab v3.0.0 and Omni Physics v110.0. Please refer to the
  `release notes`_ for any changes in the future releases.


Surface and Volume Deformables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the new Omni Physics API, deformable bodies are split into two distinct types, as described in the
`Omni Physics documentation`_:

- **Volume deformables**: 3D objects simulated with a tetrahedral FEM mesh (e.g., soft cubes, spheres, capsules).
  These support kinematic targets on individual vertices. The simulation operates on a tetrahedral mesh internally,
  while a separate triangle surface mesh handles rendering.
- **Surface deformables**: 2D surfaces simulated directly on a triangle mesh (e.g., cloth, fabric, membranes).
  These have additional material properties for controlling stretch, shear, and bend stiffness, but do not support
  kinematic vertex targets.

The type of deformable is determined by the **physics material** assigned to the object:

- :class:`~isaaclab_physx.sim.PhysxDeformableBodyMaterialCfg` creates a PhysX **volume** deformable.
- :class:`~isaaclab_physx.sim.PhysxSurfaceDeformableBodyMaterialCfg` creates a PhysX **surface** deformable.
- :class:`~isaaclab_newton.sim.spawners.materials.NewtonDeformableBodyMaterialCfg` creates a Newton
  **volume** deformable.
- :class:`~isaaclab_newton.sim.spawners.materials.NewtonSurfaceDeformableBodyMaterialCfg` creates a Newton
  **surface** deformable.


Migration from the Old API
~~~~~~~~~~~~~~~~~~~~~~~~~~

Import Changes
^^^^^^^^^^^^^^

Deformable object cfgs remain in ``isaaclab.assets``. Deformable schema and material cfgs should be imported
from the physics backend package:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Old Import
     - New Import
   * - ``from isaaclab.sim import DeformableBodyPropertiesCfg``
     - ``from isaaclab_physx.sim import PhysxDeformableBodyPropertiesCfg``
   * - ``from isaaclab.sim import DeformableBodyMaterialCfg``
     - ``from isaaclab_physx.sim import PhysxDeformableBodyMaterialCfg``
   * - ``from isaaclab.sim import SurfaceDeformableBodyMaterialCfg``
     - ``from isaaclab_physx.sim import PhysxSurfaceDeformableBodyMaterialCfg``
   * - ``from isaaclab.sim import DeformableBodyPropertiesCfg``
     - ``from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg``
   * - ``from isaaclab.sim import DeformableBodyMaterialCfg``
     - ``from isaaclab_newton.sim.spawners.materials import NewtonDeformableBodyMaterialCfg``
   * - ``from isaaclab.sim import SurfaceDeformableBodyMaterialCfg``
     - ``from isaaclab_newton.sim.spawners.materials import NewtonSurfaceDeformableBodyMaterialCfg``
   * - ``from isaaclab_physx.assets import DeformableObjectCfg``
     - ``from isaaclab.assets import DeformableObjectCfg``


Removed Properties
^^^^^^^^^^^^^^^^^^

The following properties have been **removed** from
:class:`~isaaclab_physx.sim.PhysxDeformableBodyPropertiesCfg`:

- ``collision_simplification`` and related parameters (``collision_simplification_remeshing``,
  ``collision_simplification_target_triangle_count``, ``collision_simplification_force_conforming``,
  ``collision_simplification_remove_open_edges``) — collision mesh generation is now handled automatically by
  PhysX through ``deformableUtils.create_auto_volume_deformable_hierarchy()`` and
  ``deformableUtils.create_auto_surface_deformable_hierarchy()``.
- ``simulation_hexahedral_resolution`` — the simulation mesh resolution is no longer user-configurable;
  PhysX determines it automatically.
- ``vertex_velocity_damping`` — replaced by the more general ``linear_damping`` property from the
  `PhysX deformable schema`_.
- ``sleep_damping`` — replaced by ``settling_damping`` in the `PhysX deformable schema`_.

Added Properties
^^^^^^^^^^^^^^^^

The following properties have been **added** to
:class:`~isaaclab_physx.sim.PhysxDeformableBodyPropertiesCfg`:

- ``deformable_body_enabled``, ``kinematic_enabled``, and ``mass`` — OmniPhysics
  deformable body properties owned by the PhysX backend cfg.
- ``linear_damping`` — linear damping coefficient [1/s].
- ``max_linear_velocity`` — maximum allowable linear velocity [m/s]. A negative value lets the simulation choose
  a per-vertex value dynamically (currently only supported for surface deformables).
- ``settling_damping`` — additional damping applied when vertex velocity falls below ``settling_threshold`` [1/s].
- ``enable_speculative_c_c_d`` — enables speculative continuous collision detection.
- ``disable_gravity`` — per-deformable gravity control.
- ``collision_pair_update_frequency`` — how often surface-to-surface collision pairs are updated per time step
  (surface deformables only).
- ``collision_iteration_multiplier`` — collision subiterations per solver iteration (surface deformables only).

For a full description of all available properties, refer to the `PhysX deformable schema`_ and
`OmniPhysics deformable schema`_ documentation.

Material Changes
^^^^^^^^^^^^^^^^

The deformable material hierarchy is now split by backend:

- :class:`~isaaclab.sim.DeformableBodyMaterialBaseCfg` and
  :class:`~isaaclab.sim.SurfaceDeformableBodyMaterialBaseCfg` — empty base classes for backend-specific
  deformable material configs.
- :class:`~isaaclab_physx.sim.PhysxDeformableBodyMaterialCfg` — for PhysX volume deformables. Contains ``density``,
  ``static_friction``, ``dynamic_friction``, ``youngs_modulus``, ``poissons_ratio``, and ``elasticity_damping``.
- :class:`~isaaclab_physx.sim.PhysxSurfaceDeformableBodyMaterialCfg` — extends the PhysX volume material config with
  surface-specific properties: ``surface_thickness``, ``surface_stretch_stiffness``, ``surface_shear_stiffness``,
  ``surface_bend_stiffness``, and ``bend_damping``.
- :class:`~isaaclab_newton.sim.spawners.materials.NewtonDeformableBodyMaterialCfg` and
  :class:`~isaaclab_newton.sim.spawners.materials.NewtonSurfaceDeformableBodyMaterialCfg` contain Newton-specific
  fields such as density, particle radius, direct Lame parameters ``k_mu``/``k_lambda`` for volume deformables,
  and VBD stiffness parameters for surface deformables.

The old ``damping_scale`` property has been removed. Use ``elasticity_damping`` directly instead.

DeformableObject View Change
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The internal PhysX view type has changed from ``physx.SoftBodyView`` to ``physx.DeformableBodyView``.
The property ``root_physx_view`` has been deprecated in favor of ``root_view``.


Code Examples
~~~~~~~~~~~~~

Volume Deformable (Before and After)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Before**:

.. code-block:: python
   :emphasize-lines: 1,2

   import isaaclab.sim as sim_utils
   from isaaclab.assets import DeformableObject, DeformableObjectCfg

   cfg = DeformableObjectCfg(
       prim_path="/World/Origin.*/Cube",
       spawn=sim_utils.MeshCuboidCfg(
           size=(0.2, 0.2, 0.2),
           deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
           visual_material=sim_utils.PreviewSurfaceCfg(),
           physics_material=sim_utils.DeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=1e5),
       ),
   )
   cube_object = DeformableObject(cfg=cfg)

**After**:

.. code-block:: python
   :emphasize-lines: 1,2

   import isaaclab.sim as sim_utils
   from isaaclab.assets import DeformableObject, DeformableObjectCfg
   from isaaclab_physx.sim import PhysxDeformableBodyMaterialCfg, PhysxDeformableBodyPropertiesCfg

   cfg = DeformableObjectCfg(
       prim_path="/World/Origin.*/Cube",
       spawn=sim_utils.MeshCuboidCfg(
           size=(0.2, 0.2, 0.2),
           deformable_props=PhysxDeformableBodyPropertiesCfg(),
           visual_material=sim_utils.PreviewSurfaceCfg(),
           physics_material=PhysxDeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=1e5),
       ),
   )
   cube_object = DeformableObject(cfg=cfg)

Surface Deformable (New)
^^^^^^^^^^^^^^^^^^^^^^^^

Surface deformables use :class:`~isaaclab.sim.spawners.meshes.MeshRectangleCfg` for 2D meshes, combined with
:class:`~isaaclab_physx.sim.PhysxSurfaceDeformableBodyMaterialCfg`:

.. code-block:: python

   import isaaclab.sim as sim_utils
   from isaaclab.assets import DeformableObject, DeformableObjectCfg
   from isaaclab_physx.sim import PhysxDeformableBodyPropertiesCfg, PhysxSurfaceDeformableBodyMaterialCfg

   cfg = DeformableObjectCfg(
       prim_path="/World/Origin.*/Cloth",
       spawn=sim_utils.MeshRectangleCfg(
           size=(1.5, 1.5),
           resolution=(21, 21),
           deformable_props=PhysxDeformableBodyPropertiesCfg(),
           visual_material=sim_utils.PreviewSurfaceCfg(),
           physics_material=PhysxSurfaceDeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=1e5),
       ),
   )
   cloth_object = DeformableObject(cfg=cfg)

USD File Deformable
^^^^^^^^^^^^^^^^^^^

Deformable properties can also be applied to imported USD assets using
:class:`~isaaclab.sim.spawners.from_files.UsdFileCfg`:

.. code-block:: python

   import isaaclab.sim as sim_utils
   from isaaclab.assets import DeformableObject, DeformableObjectCfg
   from isaaclab_physx.sim import PhysxDeformableBodyMaterialCfg, PhysxDeformableBodyPropertiesCfg

   from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

   cfg = DeformableObjectCfg(
       prim_path="/World/Origin.*/Teddy",
       spawn=sim_utils.UsdFileCfg(
           usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Objects/Teddy_Bear/teddy_bear.usd",
           deformable_props=PhysxDeformableBodyPropertiesCfg(),
           physics_material=PhysxDeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=1e5),
           scale=[0.05, 0.05, 0.05],
       ),
   )
   teddy_object = DeformableObject(cfg=cfg)


Limitations
~~~~~~~~~~~

- **Kinematic targets are volume-only.** Calling
  :meth:`~isaaclab.assets.DeformableObject.write_nodal_kinematic_target_to_sim_index` on a surface
  deformable will raise a ``ValueError``.
- **Surface-specific solver properties** (``collision_pair_update_frequency``,
  ``collision_iteration_multiplier``) have no effect on volume deformables.
- **Newton deformables are experimental.** They are implemented in
  :mod:`isaaclab_contrib.deformable` and currently target VBD-based solvers and
  coupled rigid-deformable workflows.


.. _Omni Physics documentation: https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.0/dev_guide/deformables/deformable_bodies.html
.. _PhysX deformable schema: https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.0/dev_guide/deformables/physx_deformable_schema.html#physxbasedeformablebodyapi
.. _OmniPhysics deformable schema: https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.0/dev_guide/deformables/omniphysics_deformable_schema.html
.. _release notes: https://github.com/isaac-sim/IsaacLab/releases
