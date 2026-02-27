.. _tutorial-spawn-prims:


Spawning prims into the scene
=============================

.. currentmodule:: isaaclab

This tutorial explores how to spawn various objects (or prims) into the scene in Isaac Lab from Python.
It builds on the previous tutorial on running the simulator from a standalone script and
demonstrates how to spawn a ground plane, lights, primitive shapes, and meshes from USD files.


The Code
~~~~~~~~

The tutorial corresponds to the ``spawn_prims.py`` script in the ``scripts/tutorials/00_sim`` directory.
Let's take a look at the Python script:

.. dropdown:: Code for spawn_prims.py
   :icon: code

   .. literalinclude:: ../../../../scripts/tutorials/00_sim/spawn_prims.py
      :language: python
      :emphasize-lines: 40-88, 100-101
      :linenos:


The Code Explained
~~~~~~~~~~~~~~~~~~

Scene designing in Omniverse is built around a software system and file format called USD (Universal Scene Description).
It allows describing 3D scenes in a hierarchical manner, similar to a file system. Since USD is a comprehensive framework,
we recommend reading the `USD documentation`_ to learn more about it.

For completeness, we introduce the must know concepts of USD in this tutorial.

* **Primitives (Prims)**: These are the basic building blocks of a USD scene. They can be thought of as nodes in a scene
  graph. Each node can be a mesh, a light, a camera, or a transform. It can also be a group of other prims under it.
* **Attributes**: These are the properties of a prim. They can be thought of as key-value pairs. For example, a prim can
  have an attribute called ``color`` with a value of ``red``.
* **Relationships**: These are the connections between prims. They can be thought of as pointers to other prims. For
  example, a mesh prim can have a relationship to a material prim for shading.

A collection of these prims, with their attributes and relationships, is called a **USD stage**. It can be thought of
as a container for all prims in a scene. When we say we are designing a scene, we are actually designing a USD stage.

While working with direct USD APIs provides a lot of flexibility, it can be cumbersome to learn and use. To make it
easier to design scenes, Isaac Lab builds on top of the USD APIs to provide a configuration-driven interface to spawn prims
into a scene. These are included in the :mod:`sim.spawners` module.

When spawning prims into the scene, each prim requires a configuration class instance that defines the prim's attributes
and relationships (through material and shading information). The configuration class is then passed to its respective
function where the prim name and transformation are specified. The function then spawns the prim into the scene.

At a high-level, this is how it works:

.. code-block:: python

   # Create a configuration class instance
   cfg = MyPrimCfg()
   prim_path = "/path/to/prim"

   # Spawn the prim into the scene using the corresponding spawner function
   spawn_my_prim(prim_path, cfg, translation=[0, 0, 0], orientation=[1, 0, 0, 0], scale=[1, 1, 1])
   # OR
   # Use the spawner function directly from the configuration class
   cfg.func(prim_path, cfg, translation=[0, 0, 0], orientation=[1, 0, 0, 0], scale=[1, 1, 1])


In this tutorial, we demonstrate the spawning of various different prims into the scene. For more
information on the available spawners, please refer to the :mod:`sim.spawners` module in Isaac Lab.

.. attention::

   All the scene designing must happen before the simulation starts. Once the simulation starts, we recommend keeping
   the scene frozen and only altering the properties of the prim. This is particularly important for GPU simulation
   as adding new prims during simulation may alter the physics simulation buffers on GPU and lead to unexpected
   behaviors.


Spawning a ground plane
-----------------------

The :class:`~sim.spawners.from_files.GroundPlaneCfg` configures a grid-like ground plane with
modifiable properties such as its appearance and size.

.. literalinclude:: ../../../../scripts/tutorials/00_sim/spawn_prims.py
   :language: python
   :start-at: # Ground-plane
   :end-at: cfg_ground.func("/World/defaultGroundPlane", cfg_ground)


Spawning lights
---------------

It is possible to spawn `different light prims`_ into the stage. These include distant lights, sphere lights, disk
lights, and cylinder lights. In this tutorial, we spawn a distant light which is a light that is infinitely far away
from the scene and shines in a single direction.

.. literalinclude:: ../../../../scripts/tutorials/00_sim/spawn_prims.py
   :language: python
   :start-at: # spawn distant light
   :end-at: cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))


Spawning primitive shapes
-------------------------

Before spawning primitive shapes, we introduce the concept of a transform prim or Xform. A transform prim is a prim that
contains only transformation properties. It is used to group other prims under it and to transform them as a group.
Here we make an Xform prim to group all the primitive shapes under it.

.. literalinclude:: ../../../../scripts/tutorials/00_sim/spawn_prims.py
   :language: python
   :start-at: # create a new xform prim for all objects to be spawned under
   :end-at: sim_utils.create_prim("/World/Objects", "Xform")

Next, we spawn a cone using the :class:`~sim.spawners.shapes.ConeCfg` class. It is possible to specify
the radius, height, physics properties, and material properties of the cone. By default, the physics and material
properties are disabled.

The first two cones we spawn ``Cone1`` and ``Cone2`` are visual elements and do not have physics enabled.

.. literalinclude:: ../../../../scripts/tutorials/00_sim/spawn_prims.py
   :language: python
   :start-at: # spawn a red cone
   :end-at: cfg_cone.func("/World/Objects/Cone2", cfg_cone, translation=(-1.0, -1.0, 1.0))

For the third cone ``ConeRigid``, we add rigid body physics to it by setting the attributes for that in the configuration
class. Through these attributes, we can specify the mass, friction, and restitution of the cone. If unspecified, they
default to the default values set by USD Physics.

.. literalinclude:: ../../../../scripts/tutorials/00_sim/spawn_prims.py
   :language: python
   :start-at: # spawn a green cone with colliders and rigid body
   :end-before: # spawn a blue cuboid with deformable body

Lastly, we spawn a cuboid ``CuboidDeformable`` which contains deformable body physics properties. Unlike the
rigid body simulation, a deformable body can have relative motion between its vertices. This is useful for simulating
soft bodies like cloth, rubber, or jello. It is important to note that deformable bodies are only supported in
GPU simulation and require a mesh object to be spawned with the deformable body physics properties.

.. literalinclude:: ../../../../scripts/tutorials/00_sim/spawn_prims.py
   :language: python
   :start-at: # spawn a blue cuboid with deformable body
   :end-before: # spawn a usd file of a table into the scene

Spawning from another file
--------------------------

Lastly, it is possible to spawn prims from other file formats such as other USD, URDF, or OBJ files. In this tutorial,
we spawn a USD file of a table into the scene. The table is a mesh prim and has a material prim associated with it.
All of this information is stored in its USD file.

.. literalinclude:: ../../../../scripts/tutorials/00_sim/spawn_prims.py
   :language: python
   :start-at: # spawn a usd file of a table into the scene
   :end-at: cfg.func("/World/Objects/Table", cfg, translation=(0.0, 0.0, 1.05))

The table above is added as a reference to the scene. In layman terms, this means that the table is not actually added
to the scene, but a ``pointer`` to the table asset is added. This allows us to modify the table asset and have the changes
reflected in the scene in a non-destructive manner. For example, we can change the material of the table without
actually modifying the underlying file for the table asset directly. Only the changes are stored in the USD stage.


Executing the Script
~~~~~~~~~~~~~~~~~~~~

Similar to the tutorial before, to run the script, execute the following command:

.. code-block:: bash

  ./isaaclab.sh -p scripts/tutorials/00_sim/spawn_prims.py

Once the simulation starts, you should see a window with a ground plane, a light, some cones, and a table.
The green cone, which has rigid body physics enabled, should fall and collide with the table and the ground
plane. The other cones are visual elements and should not move. To stop the simulation, you can close the window,
or press ``Ctrl+C`` in the terminal.

.. figure:: ../../_static/tutorials/tutorial_spawn_prims.jpg
    :align: center
    :figwidth: 100%
    :alt: result of spawn_prims.py

This tutorial provided a foundation for spawning various prims into the scene in Isaac Lab. Although simple, it
demonstrates the basic concepts of scene designing in Isaac Lab and how to use the spawners. In the coming tutorials,
we will now look at how to interact with the scene and the simulation.


.. _`USD documentation`: https://graphics.pixar.com/usd/docs/index.html
.. _`different light prims`: https://youtu.be/c7qyI8pZvF4?feature=shared
