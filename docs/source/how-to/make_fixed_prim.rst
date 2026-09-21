:orphan:

Making a physics prim fixed in the simulation
=============================================

.. currentmodule:: isaaclab

When a USD prim has physics schemas applied on it, it is affected by physics simulation.
This means that the prim can move, rotate, and collide with other prims in the simulation world.
However, there are cases where it is desirable to make certain prims static in the simulation world,
i.e. the prim should still participate in collisions but its position and orientation should not change.

The following sections describe how to spawn a prim with physics schemas and make it static in the simulation world.

Static colliders and fixed articulation roots use shared USD topology on PhysX and Newton.
Use the shared configurations below for these operations. Backend-specific solver settings are
covered in :ref:`asset-config-backends`.

Static colliders
----------------

Static colliders are prims that are not affected by physics but can collide with other prims in the simulation world.
These don't have any rigid body properties applied on them. However, this also means that they can't be accessed
using the physics tensor API (i.e., through the :class:`assets.RigidObject` class).

For instance, to spawn a sphere static in the simulation world, the following code can be used:

.. code-block:: python

    import isaaclab.sim as sim_utils

    sphere_spawn_cfg = sim_utils.SphereCfg(
        radius=0.15,
        collision_props=sim_utils.CollisionBaseCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
    )
    sphere_spawn_cfg.func(
        "/World/Sphere", sphere_spawn_cfg, translation=(0.0, 0.0, 2.0)
    )


Rigid object
------------

Rigid objects (i.e. object only has a single body) can be made static by setting the parameter
:attr:`sim.schemas.RigidBodyBaseCfg.kinematic_enabled` as True. This will make the object
kinematic: its motion is prescribed by user code rather than integrated from gravity and
forces. Without prescribed motion, it stays fixed. It can still participate in collisions.

For instance, to spawn a sphere static in the simulation world but with rigid body schema on it,
the following code can be used:

.. code-block:: python

    import isaaclab.sim as sim_utils

    sphere_spawn_cfg = sim_utils.SphereCfg(
        radius=0.15,
        rigid_props=sim_utils.RigidBodyBaseCfg(kinematic_enabled=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionBaseCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
    )
    sphere_spawn_cfg.func(
        "/World/Sphere", sphere_spawn_cfg, translation=(0.0, 0.0, 2.0)
    )


Articulation
------------

Fixing the root of an articulation requires having a fixed joint to the root rigid body link of the articulation.
This can be achieved by setting the parameter :attr:`sim.schemas.ArticulationRootBaseCfg.fix_root_link`
as True. Based on the value of this parameter, the following cases are possible:

* If set to :obj:`None`, the root link is not modified.
* If the articulation already has a fixed root link, this flag will enable or disable the fixed joint.
* If the articulation does not have a fixed root link, this flag will create a fixed joint between the world
  frame and the root link, under a writable prim in the asset hierarchy.

For instance, to spawn an ANYmal robot and fix its base while leaving its joints movable, the following code can be used:

.. code-block:: python

    import isaaclab.sim as sim_utils
    from isaaclab.sim.schemas import ArticulationRootBaseCfg
    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

    anymal_spawn_cfg = sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd",
        articulation_props=ArticulationRootBaseCfg(fix_root_link=True),
    )
    anymal_spawn_cfg.func(
        "/World/ANYmal", anymal_spawn_cfg, translation=(0.0, 0.0, 0.8), orientation=(0.0, 0.0, 0.0, 1.0)
    )


This creates a fixed joint between the world frame and the root link of the ANYmal robot.
The joint is authored under a writable prim; its exact path depends on the asset's root
and instancing layout.

For a spawn configuration using schema fragments, or no ``articulation_props``, set
``UsdFileCfg(fix_root_link=True, ...)`` instead. When using a legacy properties configuration
as above, keep ``fix_root_link`` on that configuration. Both paths author the fixed joint
before the backend imports the articulation.

.. _further-notes:

.. dropdown:: PhysX articulation-root placement details

   The following parser details explain the PhysX-specific root-placement adjustment. Newton
   imports the resulting USD joint topology; do not apply PhysX parser heuristics as Newton
   configuration rules.

   Given the flexibility of USD asset designing the following possible scenarios are usually encountered:

   1. **Articulation root schema on the rigid body prim without a fixed joint**:

      This is the most common and recommended scenario for floating-base articulations. The root prim
      has both the rigid body and the articulation root properties. In this case, the articulation root
      is parsed as a floating-base with the root prim of the articulation ``Link0Xform``.

      .. code-block:: text

          ArticulationXform
              └── Link0Xform  (RigidBody and ArticulationRoot schema)

   2. **Articulation root schema on the parent prim with a fixed joint**:

      This is the expected arrangement for fixed-base articulations. The root prim has only the rigid body
      properties and the articulation root properties are applied to its parent prim. In this case, the
      articulation root is parsed as a fixed-base with the root prim of the articulation ``Link0Xform``.

      .. code-block:: text

          ArticulationXform (ArticulationRoot schema)
              └── Link0Xform  (RigidBody schema)
              └── FixedJoint (connecting the world frame and Link0Xform)

   3. **Articulation root schema on the parent prim without a fixed joint**:

      This is a scenario where the root prim has only the rigid body properties and the articulation root properties
      are applied to its parent prim. However, the fixed joint is not created between the world frame and the root link.
      In this case, the articulation is parsed as a floating-base system. However, the PhysX parser uses its own
      heuristic (such as alphabetical order) to determine the root prim of the articulation. It may select the root prim
      at ``Link0Xform`` or choose another prim as the root prim.

      .. code-block:: text

          ArticulationXform (ArticulationRoot schema)
              └── Link0Xform  (RigidBody schema)

   4. **Articulation root schema on the rigid body prim with a fixed joint**:

      While this is a valid scenario, it is not recommended as it may lead to unexpected behavior. In this case,
      the articulation is still parsed as a floating-base system. However, the fixed joint, created between the
      world frame and the root link, is considered as a part of the maximal coordinate tree. This is different from
      PhysX considering the articulation as a fixed-base system. Hence, the simulation may not behave as expected.

      .. code-block:: text

          ArticulationXform
              └── Link0Xform  (RigidBody and ArticulationRoot schema)
              └── FixedJoint (connecting the world frame and Link0Xform)

   For floating base articulations, the root prim usually has both the rigid body and the articulation
   root properties. However, directly connecting this prim to the world frame will cause the simulation
   to consider the fixed joint as a part of the maximal coordinate tree. This is different from PhysX
   considering the articulation as a fixed-base system.

   Internally, when the parameter :attr:`sim.schemas.ArticulationRootBaseCfg.fix_root_link` is set to True
   and the articulation is detected as a floating-base system, the fixed joint is created between the world frame
   the root rigid body link of the articulation. However, to make the PhysX parser consider the articulation as a
   fixed-base system, the articulation root properties are removed from the root rigid body prim and applied to
   its parent prim instead.
