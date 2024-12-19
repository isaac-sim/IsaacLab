.. _tutorial-interact-rigid-object:


Interacting with a rigid object
===============================

.. currentmodule:: isaaclab

In the previous tutorials, we learned the essential workings of the standalone script and how to
spawn different objects (or *prims*) into the simulation. This tutorial shows how to create and interact
with a rigid object. For this, we will use the :class:`assets.RigidObject` class provided in Isaac Lab.

The Code
~~~~~~~~

The tutorial corresponds to the ``run_rigid_object.py`` script in the ``scripts/tutorials/01_assets`` directory.

.. dropdown:: Code for run_rigid_object.py
   :icon: code

   .. literalinclude:: ../../../../scripts/tutorials/01_assets/run_rigid_object.py
      :language: python
      :emphasize-lines: 55-74, 76-78, 98-108, 111-112, 118-119, 132-134, 139-140
      :linenos:


The Code Explained
~~~~~~~~~~~~~~~~~~

In this script, we split the ``main`` function into two separate functions, which highlight the two main
steps of setting up any simulation in the simulator:

1. **Design scene**: As the name suggests, this part is responsible for adding all the prims to the scene.
2. **Run simulation**: This part is responsible for stepping the simulator, interacting with the prims
   in the scene, e.g., changing their poses, and applying any commands to them.

A distinction between these two steps is necessary because the second step only happens after the first step
is complete and the simulator is reset. Once the simulator is reset (which automatically plays the simulation),
no new (physics-enabled) prims should be added to the scene as it may lead to unexpected behaviors. However,
the prims can be interacted with through their respective handles.


Designing the scene
-------------------

Similar to the previous tutorial, we populate the scene with a ground plane and a light source. In addition,
we add a rigid object to the scene using the :class:`assets.RigidObject` class. This class is responsible for
spawning the prims at the input path and initializes their corresponding rigid body physics handles.

In this tutorial, we create a conical rigid object using the spawn configuration similar to the rigid cone
in the :ref:`Spawn Objects <tutorial-spawn-prims>` tutorial. The only difference is that now we wrap
the spawning configuration into the :class:`assets.RigidObjectCfg` class. This class contains information about
the asset's spawning strategy, default initial state, and other meta-information. When this class is passed to
the :class:`assets.RigidObject` class, it spawns the object and initializes the corresponding physics handles
when the simulation is played.

As an example on spawning the rigid object prim multiple times, we create its parent Xform prims,
``/World/Origin{i}``, that correspond to different spawn locations. When the regex expression
``/World/Origin*/Cone`` is passed to the :class:`assets.RigidObject` class, it spawns the rigid object prim at
each of the ``/World/Origin{i}`` locations. For instance, if ``/World/Origin1`` and ``/World/Origin2`` are
present in the scene, the rigid object prims are spawned at the locations ``/World/Origin1/Cone`` and
``/World/Origin2/Cone`` respectively.

.. literalinclude:: ../../../../scripts/tutorials/01_assets/run_rigid_object.py
   :language: python
   :start-at: # Create separate groups called "Origin1", "Origin2", "Origin3"
   :end-at: cone_object = RigidObject(cfg=cone_cfg)

Since we want to interact with the rigid object, we pass this entity back to the main function. This entity
is then used to interact with the rigid object in the simulation loop. In later tutorials, we will see a more
convenient way to handle multiple scene entities using the :class:`scene.InteractiveScene` class.

.. literalinclude:: ../../../../scripts/tutorials/01_assets/run_rigid_object.py
   :language: python
   :start-at: # return the scene information
   :end-at: return scene_entities, origins


Running the simulation loop
---------------------------

We modify the simulation loop to interact with the rigid object to include three steps -- resetting the
simulation state at fixed intervals, stepping the simulation, and updating the internal buffers of the
rigid object. For the convenience of this tutorial, we extract the rigid object's entity from the scene
dictionary and store it in a variable.

Resetting the simulation state
""""""""""""""""""""""""""""""

To reset the simulation state of the spawned rigid object prims, we need to set their pose and velocity.
Together they define the root state of the spawned rigid objects. It is important to note that this state
is defined in the **simulation world frame**, and not of their parent Xform prim. This is because the physics
engine only understands the world frame and not the parent Xform prim's frame. Thus, we need to transform
desired state of the rigid object prim into the world frame before setting it.

We use the :attr:`assets.RigidObject.data.default_root_state` attribute to get the default root state of the
spawned rigid object prims. This default state can be configured from the :attr:`assets.RigidObjectCfg.init_state`
attribute, which we left as identity in this tutorial. We then randomize the translation of the root state and
set the desired state of the rigid object prim using the :meth:`assets.RigidObject.write_root_pose_to_sim` and :meth:`assets.RigidObject.write_root_velocity_to_sim` methods.
As the name suggests, this method writes the root state of the rigid object prim into the simulation buffer.

.. literalinclude:: ../../../../scripts/tutorials/01_assets/run_rigid_object.py
   :language: python
   :start-at: # reset root state
   :end-at: cone_object.reset()

Stepping the simulation
"""""""""""""""""""""""

Before stepping the simulation, we perform the :meth:`assets.RigidObject.write_data_to_sim` method. This method
writes other data, such as external forces, into the simulation buffer. In this tutorial, we do not apply any
external forces to the rigid object, so this method is not necessary. However, it is included for completeness.

.. literalinclude:: ../../../../scripts/tutorials/01_assets/run_rigid_object.py
   :language: python
   :start-at: # apply sim data
   :end-at: cone_object.write_data_to_sim()

Updating the state
""""""""""""""""""

After stepping the simulation, we update the internal buffers of the rigid object prims to reflect their new state
inside the :class:`assets.RigidObject.data` attribute. This is done using the :meth:`assets.RigidObject.update` method.

.. literalinclude:: ../../../../scripts/tutorials/01_assets/run_rigid_object.py
   :language: python
   :start-at: # update buffers
   :end-at: cone_object.update(sim_dt)


The Code Execution
~~~~~~~~~~~~~~~~~~

Now that we have gone through the code, let's run the script and see the result:

.. code-block:: bash

   ./isaaclab.sh -p scripts/tutorials/01_assets/run_rigid_object.py


This should open a stage with a ground plane, lights, and several green cones. The cones must be dropping from
a random height and settling on to the ground. To stop the simulation, you can either close the window, or press
the ``STOP`` button in the UI, or press ``Ctrl+C`` in the terminal

.. figure:: ../../_static/tutorials/tutorial_run_rigid_object.jpg
    :align: center
    :figwidth: 100%
    :alt: result of run_rigid_object.py


This tutorial showed how to spawn rigid objects and wrap them in a :class:`RigidObject` class to initialize their
physics handles which allows setting and obtaining their state. In the next tutorial, we will see how to interact
with an articulated object which is a collection of rigid objects connected by joints.
