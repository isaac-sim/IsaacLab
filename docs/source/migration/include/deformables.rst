.. _migrating-deformables:

Migration of Deformables
------------------------

.. currentmodule:: isaaclab

Isaac Lab 3.0 updates the deformable body API to align with Omni Physics 110.0. The old soft body
API is deprecated and replaced by two distinct deformable types:

- **Volume deformables**: 3D objects simulated with a tetrahedral FEM mesh (soft cubes, teddy
  bears). They support kinematic targets on individual vertices.
- **Surface deformables**: 2D surfaces simulated directly on a triangle mesh (cloth, membranes).
  They add stretch, shear, and bend stiffness, but do not support kinematic vertex targets.

The type is determined by the physics material assigned to the object:

- :class:`~isaaclab_physx.sim.PhysxDeformableBodyMaterialCfg` for PhysX volume deformables.
- :class:`~isaaclab_physx.sim.PhysxSurfaceDeformableBodyMaterialCfg` for PhysX surface deformables.
- :class:`~isaaclab_newton.sim.NewtonDeformableBodyMaterialCfg` for Newton volume deformables.
- :class:`~isaaclab_newton.sim.NewtonSurfaceDeformableBodyMaterialCfg` for Newton surface deformables.

.. rubric:: Import Changes

Deformable object cfgs remain in ``isaaclab.assets``. Deformable schema and material cfgs are
backend-specific and move to the backend package:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Old Import (``isaaclab.sim``)
     - New Import
   * - ``DeformableBodyPropertiesCfg``
     - ``isaaclab_physx.sim.PhysxDeformableBodyPropertiesCfg`` or
       ``isaaclab_newton.sim.NewtonDeformableBodyPropertiesCfg``
   * - ``DeformableBodyMaterialCfg``
     - ``isaaclab_physx.sim.PhysxDeformableBodyMaterialCfg`` or
       ``isaaclab_newton.sim.NewtonDeformableBodyMaterialCfg``
   * - ``SurfaceDeformableBodyMaterialCfg``
     - ``isaaclab_physx.sim.PhysxSurfaceDeformableBodyMaterialCfg`` or
       ``isaaclab_newton.sim.NewtonSurfaceDeformableBodyMaterialCfg``

:class:`~isaaclab.sim.DeformableBodyPropertiesBaseCfg` is now empty; the OmniPhysics deformable
body fields are owned by :class:`~isaaclab_physx.sim.PhysxDeformableBodyPropertiesCfg`.

.. rubric:: Example: Volume Deformable

**Before**:

.. code-block:: python
   :emphasize-lines: 8,10

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
   :emphasize-lines: 3,9,11

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

.. rubric:: Removed Properties

The following fields no longer exist:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Removed from
     - Replacement
   * - ``PhysxDeformableBodyPropertiesCfg.collision_simplification`` and its
       ``collision_simplification_*`` parameters
     - None. PhysX generates the collision mesh automatically.
   * - ``PhysxDeformableBodyPropertiesCfg.simulation_hexahedral_resolution``
     - None. PhysX determines the simulation mesh resolution.
   * - ``PhysxDeformableBodyPropertiesCfg.vertex_velocity_damping``
     - ``linear_damping``
   * - ``PhysxDeformableBodyPropertiesCfg.sleep_damping``
     - ``settling_damping``
   * - ``PhysxDeformableBodyMaterialCfg.damping_scale``
     - ``elasticity_damping``
   * - ``contact_offset`` / ``rest_offset`` and ``PhysxDeformableCollisionPropertiesCfg``
     - Set them on the mesh spawner instead, using
       :class:`~isaaclab_physx.sim.schemas.PhysxCollisionCfg`:
       ``collision_props=[PhysxCollisionCfg(rest_offset=0.0005, contact_offset=0.005)]``.
       PhysX reads collision offsets off the collider, which for a deformable is its simulation
       mesh, so authoring them on the body prim never reached the solver.

:class:`~isaaclab_physx.sim.PhysxDeformableBodyPropertiesCfg` also gained fields from the new
schema. See the class reference and the `PhysX deformable schema`_ for the current list.

.. rubric:: Behavior Changes

- Kinematic targets are volume-only. Calling
  :meth:`~isaaclab.assets.DeformableObject.write_nodal_kinematic_target_to_sim_index` on a surface
  deformable raises a ``ValueError``.
- ``collision_pair_update_frequency`` and ``collision_iteration_multiplier`` have no effect on
  volume deformables.
- The PhysX view behind a deformable changed from ``physx.SoftBodyView`` to
  ``physx.DeformableBodyView``, and ``root_physx_view`` is deprecated in favor of ``root_view``.

For runnable volume, surface, and USD-asset examples, see the
:ref:`tutorial-interact-deformable-object` tutorial and ``scripts/demos/deformables.py``.


.. _PhysX deformable schema: https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.0/dev_guide/deformables/physx_deformable_schema.html#physxbasedeformablebodyapi
