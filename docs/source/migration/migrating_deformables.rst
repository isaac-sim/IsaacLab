.. _migrating-deformables:

Migration of Deformables
========================

.. currentmodule:: isaaclab

In the newer versions of Omni Physics (107.0 and later), the old deformable body functionality has become deprecated.
The following sections describe the changes to migrate to the new Omni Physics API, specifically moving away from
Soft Bodies and towards Surface and Volume Deformables. We currently only support deformable bodies in the PhysX
backend, hence these features are implemented in ``isaaclab_physx``.

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

- :class:`~isaaclab_physx.sim.DeformableBodyMaterialCfg` creates a **volume** deformable.
- :class:`~isaaclab_physx.sim.SurfaceDeformableBodyMaterialCfg` creates a **surface** deformable.


Migration from the Old API
~~~~~~~~~~~~~~~~~~~~~~~~~~

Import Changes
^^^^^^^^^^^^^^

All deformable-related classes have moved from ``isaaclab`` to ``isaaclab_physx``. The table below summarizes the
import changes:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Old Import
     - New Import
   * - ``from isaaclab.sim import DeformableBodyPropertiesCfg``
     - ``from isaaclab_physx.sim import DeformableBodyPropertiesCfg``
   * - ``from isaaclab.sim import DeformableBodyMaterialCfg``
     - ``from isaaclab_physx.sim import DeformableBodyMaterialCfg``


Removed Properties
^^^^^^^^^^^^^^^^^^

The following properties have been **removed** from :class:`~isaaclab_physx.sim.DeformableBodyPropertiesCfg`:

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

The following properties have been **added** to :class:`~isaaclab_physx.sim.DeformableBodyPropertiesCfg`:

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

The old :class:`DeformableBodyMaterialCfg` (from ``isaaclab.sim``) has been replaced by a new hierarchy in
``isaaclab_physx``:

- :class:`~isaaclab_physx.sim.DeformableBodyMaterialCfg` — for volume deformables. Contains ``density``,
  ``static_friction``, ``dynamic_friction``, ``youngs_modulus``, ``poissons_ratio``, and ``elasticity_damping``.
- :class:`~isaaclab_physx.sim.SurfaceDeformableBodyMaterialCfg` — extends the volume material config with
  surface-specific properties: ``surface_thickness``, ``surface_stretch_stiffness``, ``surface_shear_stiffness``,
  ``surface_bend_stiffness``, and ``bend_damping``.

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
   :emphasize-lines: 1,2,3

   import isaaclab.sim as sim_utils
   from isaaclab_physx.assets import DeformableObject, DeformableObjectCfg
   from isaaclab_physx.sim import DeformableBodyPropertiesCfg, DeformableBodyMaterialCfg

   cfg = DeformableObjectCfg(
       prim_path="/World/Origin.*/Cube",
       spawn=sim_utils.MeshCuboidCfg(
           size=(0.2, 0.2, 0.2),
           deformable_props=DeformableBodyPropertiesCfg(),
           visual_material=sim_utils.PreviewSurfaceCfg(),
           physics_material=DeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=1e5),
       ),
   )
   cube_object = DeformableObject(cfg=cfg)

Surface Deformable (New)
^^^^^^^^^^^^^^^^^^^^^^^^

Surface deformables use :class:`~isaaclab.sim.spawners.meshes.MeshSquareCfg` for 2D meshes, combined with
:class:`~isaaclab_physx.sim.SurfaceDeformableBodyMaterialCfg`:

.. code-block:: python

   import isaaclab.sim as sim_utils
   from isaaclab_physx.assets import DeformableObject, DeformableObjectCfg
   from isaaclab_physx.sim import DeformableBodyPropertiesCfg, SurfaceDeformableBodyMaterialCfg

   cfg = DeformableObjectCfg(
       prim_path="/World/Origin.*/Cloth",
       spawn=sim_utils.MeshSquareCfg(
           size=1.5,
           resolution=(21, 21),
           deformable_props=DeformableBodyPropertiesCfg(),
           visual_material=sim_utils.PreviewSurfaceCfg(),
           physics_material=SurfaceDeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=1e5),
       ),
   )
   cloth_object = DeformableObject(cfg=cfg)

USD File Deformable
^^^^^^^^^^^^^^^^^^^

Deformable properties can also be applied to imported USD assets using
:class:`~isaaclab.sim.spawners.from_files.UsdFileCfg`:

.. code-block:: python

   import isaaclab.sim as sim_utils
   from isaaclab_physx.assets import DeformableObject, DeformableObjectCfg
   from isaaclab_physx.sim import DeformableBodyPropertiesCfg, DeformableBodyMaterialCfg

   from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

   cfg = DeformableObjectCfg(
       prim_path="/World/Origin.*/Teddy",
       spawn=sim_utils.UsdFileCfg(
           usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Objects/Teddy_Bear/teddy_bear.usd",
           deformable_props=DeformableBodyPropertiesCfg(),
           physics_material=DeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=1e5),
           scale=[0.05, 0.05, 0.05],
       ),
   )
   teddy_object = DeformableObject(cfg=cfg)


Limitations
~~~~~~~~~~~

- **Kinematic targets are volume-only.** Calling
  :meth:`~isaaclab_physx.assets.DeformableObject.write_nodal_kinematic_target_to_sim_index` on a surface
  deformable will raise a ``ValueError``.
- **Surface-specific solver properties** (``collision_pair_update_frequency``,
  ``collision_iteration_multiplier``) have no effect on volume deformables.
- **Deformables are PhysX-only.** The ``isaaclab_physx`` extension is required; other physics backends
  do not support deformable bodies through Isaac Lab yet.


.. _Omni Physics documentation: https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.0/dev_guide/deformables/deformable_bodies.html
.. _PhysX deformable schema: https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.0/dev_guide/deformables/physx_deformable_schema.html#physxbasedeformablebodyapi
.. _OmniPhysics deformable schema: https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.0/dev_guide/deformables/omniphysics_deformable_schema.html
.. _release notes: https://github.com/isaac-sim/IsaacLab/releases
