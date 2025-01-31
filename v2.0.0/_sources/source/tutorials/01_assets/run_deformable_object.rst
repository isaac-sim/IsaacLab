.. _tutorial-interact-deformable-object:


Interacting with a deformable object
====================================

.. currentmodule:: isaaclab

While deformable objects sometimes refer to a broader class of objects, such as cloths, fluids and soft bodies,
in PhysX, deformable objects syntactically correspond to soft bodies. Unlike rigid objects, soft bodies can deform
under external forces and collisions.

Soft bodies are simulated using Finite Element Method (FEM) in PhysX. The soft body comprises of two tetrahedral
meshes -- a simulation mesh and a collision mesh. The simulation mesh is used to simulate the deformations of
the soft body, while the collision mesh is used to detect collisions with other objects in the scene.
For more details, please check the `PhysX documentation`_.

This tutorial shows how to interact with a deformable object in the simulation. We will spawn a
set of soft cubes and see how to set their nodal positions and velocities, along with apply kinematic
commands to the mesh nodes to move the soft body.


The Code
~~~~~~~~

The tutorial corresponds to the ``run_deformable_object.py`` script in the ``scripts/tutorials/01_assets`` directory.

.. dropdown:: Code for run_deformable_object.py
   :icon: code

   .. literalinclude:: ../../../../scripts/tutorials/01_assets/run_deformable_object.py
      :language: python
      :emphasize-lines: 61-73, 75-77, 102-110, 112-115, 117-118, 123-130, 132-133, 139-140
      :linenos:


The Code Explained
~~~~~~~~~~~~~~~~~~

Designing the scene
-------------------

Similar to the :ref:`tutorial-interact-rigid-object` tutorial, we populate the scene with a ground plane
and a light source. In addition, we add a deformable object to the scene using the :class:`assets.DeformableObject`
class. This class is responsible for spawning the prims at the input path and initializes their corresponding
deformable body physics handles.

In this tutorial, we create a cubical soft object using the spawn configuration similar to the deformable cube
in the :ref:`Spawn Objects <tutorial-spawn-prims>` tutorial. The only difference is that now we wrap
the spawning configuration into the :class:`assets.DeformableObjectCfg` class. This class contains information about
the asset's spawning strategy and default initial state. When this class is passed to
the :class:`assets.DeformableObject` class, it spawns the object and initializes the corresponding physics handles
when the simulation is played.

.. note::
    The deformable object is only supported in GPU simulation and requires a mesh object to be spawned with the
    deformable body physics properties on it.


As seen in the rigid body tutorial, we can spawn the deformable object into the scene in a similar fashion by creating
an instance of the :class:`assets.DeformableObject` class by passing the configuration object to its constructor.

.. literalinclude:: ../../../../scripts/tutorials/01_assets/run_deformable_object.py
   :language: python
   :start-at: # Create separate groups called "Origin1", "Origin2", "Origin3"
   :end-at: cube_object = DeformableObject(cfg=cfg)

Running the simulation loop
---------------------------

Continuing from the rigid body tutorial, we reset the simulation at regular intervals, apply kinematic commands
to the deformable body, step the simulation, and update the deformable object's internal buffers.

Resetting the simulation state
""""""""""""""""""""""""""""""

Unlike rigid bodies and articulations, deformable objects have a different state representation. The state of a
deformable object is defined by the nodal positions and velocities of the mesh. The nodal positions and velocities
are defined in the **simulation world frame** and are stored in the :attr:`assets.DeformableObject.data` attribute.

We use the :attr:`assets.DeformableObject.data.default_nodal_state_w` attribute to get the default nodal state of the
spawned object prims. This default state can be configured from the :attr:`assets.DeformableObjectCfg.init_state`
attribute, which we left as identity in this tutorial.

.. attention::
   The initial state in the configuration :attr:`assets.DeformableObjectCfg` specifies the pose
   of the deformable object at the time of spawning. Based on this initial state, the default nodal state is
   obtained when the simulation is played for the first time.

We apply transformations to the nodal positions to randomize the initial state of the deformable object.

.. literalinclude:: ../../../../scripts/tutorials/01_assets/run_deformable_object.py
   :language: python
   :start-at: # reset the nodal state of the object
   :end-at: nodal_state[..., :3] = cube_object.transform_nodal_pos(nodal_state[..., :3], pos_w, quat_w)

To reset the deformable object, we first set the nodal state by calling the :meth:`assets.DeformableObject.write_nodal_state_to_sim`
method. This method writes the nodal state of the deformable object prim into the simulation buffer.
Additionally, we free all the kinematic targets set for the nodes in the previous simulation step by calling
the :meth:`assets.DeformableObject.write_nodal_kinematic_target_to_sim` method. We explain the
kinematic targets in the next section.

Finally, we call the :meth:`assets.DeformableObject.reset` method to reset any internal buffers and caches.

.. literalinclude:: ../../../../scripts/tutorials/01_assets/run_deformable_object.py
   :language: python
   :start-at: # write nodal state to simulation
   :end-at: cube_object.reset()

Stepping the simulation
"""""""""""""""""""""""

Deformable bodies support user-driven kinematic control where a user can specify position targets for some of
the mesh nodes while the rest of the nodes are simulated using the FEM solver. This `partial kinematic`_ control
is useful for applications where the user wants to interact with the deformable object in a controlled manner.

In this tutorial, we apply kinematic commands to two out of the four cubes in the scene. We set the position
targets for the node at index 0 (bottom-left corner) to move the cube along the z-axis.

At every step, we increment the kinematic position target for the node by a small value. Additionally,
we set the flag to indicate that the target is a kinematic target for that node in the simulation buffer.
These are set into the simulation buffer by calling the :meth:`assets.DeformableObject.write_nodal_kinematic_target_to_sim`
method.

.. literalinclude:: ../../../../scripts/tutorials/01_assets/run_deformable_object.py
   :language: python
   :start-at: # update the kinematic target for cubes at index 0 and 3
   :end-at: cube_object.write_nodal_kinematic_target_to_sim(nodal_kinematic_target)

Similar to the rigid object and articulation, we perform the :meth:`assets.DeformableObject.write_data_to_sim` method
before stepping the simulation. For deformable objects, this method does not apply any external forces to the object.
However, we keep this method for completeness and future extensions.

.. literalinclude:: ../../../../scripts/tutorials/01_assets/run_deformable_object.py
   :language: python
   :start-at: # write internal data to simulation
   :end-at: cube_object.write_data_to_sim()

Updating the state
""""""""""""""""""

After stepping the simulation, we update the internal buffers of the deformable object prims to reflect their new state
inside the :class:`assets.DeformableObject.data` attribute. This is done using the :meth:`assets.DeformableObject.update` method.

At a fixed interval, we print the root position of the deformable object to the terminal. As mentioned
earlier, there is no concept of a root state for deformable objects. However, we compute the root position as
the average position of all the nodes in the mesh.

.. literalinclude:: ../../../../scripts/tutorials/01_assets/run_deformable_object.py
   :language: python
   :start-at: # update buffers
   :end-at: print(f"Root position (in world): {cube_object.data.root_pos_w[:, :3]}")


The Code Execution
~~~~~~~~~~~~~~~~~~

Now that we have gone through the code, let's run the script and see the result:

.. code-block:: bash

   ./isaaclab.sh -p scripts/tutorials/01_assets/run_deformable_object.py


This should open a stage with a ground plane, lights, and several green cubes. Two of the four cubes must be dropping
from a height and settling on to the ground. Meanwhile the other two cubes must be moving along the z-axis. You
should see a marker showing the kinematic target position for the nodes at the bottom-left corner of the cubes.
To stop the simulation, you can either close the window, or press ``Ctrl+C`` in the terminal

.. figure:: ../../_static/tutorials/tutorial_run_deformable_object.jpg
    :align: center
    :figwidth: 100%
    :alt: result of run_deformable_object.py

This tutorial showed how to spawn deformable objects and wrap them in a :class:`DeformableObject` class to initialize their
physics handles which allows setting and obtaining their state. We also saw how to apply kinematic commands to the
deformable object to move the mesh nodes in a controlled manner. In the next tutorial, we will see how to create
a scene using the :class:`InteractiveScene` class.

.. _PhysX documentation: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/SoftBodies.html
.. _partial kinematic: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/SoftBodies.html#kinematic-soft-bodies
