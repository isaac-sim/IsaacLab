.. _how_to_rigid_objects_label:

Interacting with a rigid object
===============================

In the previous tutorial, we explained the essential workings of the standalone script, how to
play the simulator and add the first robot to the scene.

This tutorial demonstrates how to wrap objects in the :class:`RigidObject` class and how to control their position and velocity by directly interacting with the physx buffers.


The Code
~~~~~~~~

The tutorial corresponds to the ``rigid_object.py`` script in the ``orbit/source/standalone/tutorials/01_assets`` directory.

.. dropdown:: :fa:`eye,mr-1` Code for `rigid_object.py`

   .. literalinclude:: ../../../../source/standalone/tutorials/01_assets/rigid_object.py
      :language: python
      :emphasize-lines: 64-74, 76-79, 130-131, 92-110, 111-119
      :linenos:


The Code Explained
~~~~~~~~~~~~~~~~~~

The scene is designed similarly to the previous tutorial. In the following, we will discuss in detail the ``add_rigid_objects`` function, which is responsible for adding the objects to the scene, and the ``run_simulator`` function, which steps the simulator and changes their position and orientation.

Adding the rigid_objects
------------------------

The objects are included in the Nucleus Server of Omniverse. At first, we define their path and then spawn them into the scene. This procedure is discussed in the :ref:`Spawn Objects <_how_to_spawn_objects_label>` how-to guide.

.. literalinclude:: ../../../../source/standalone/tutorials/01_assets/rigid_object.py
   :language: python
   :lines: 64-74
   :linenos:
   :lineno-start: 64

In addition, we now wrap the spawned objects as objects of the :class:`RigidObject` class. This class is a wrapper around the PhysX rigid-body view and allows us to access the physics handles of the object directly. The class also provides methods to set the object's pose and velocity and retrieve the current state.

.. literalinclude:: ../../../../source/standalone/tutorials/01_assets/rigid_object.py
   :language: python
   :lines: 76-79
   :linenos:
   :lineno-start: 76

Please note that the physics handles are initialized when the simulation is played. Thus, we must call ``sim.reset()`` before accessing the physics handles.

.. literalinclude:: ../../../../source/standalone/tutorials/01_assets/rigid_object.py
   :language: python
   :lines: 130-131
   :linenos:
   :lineno-start: 130


Running the simulation loop
---------------------------

In this tutorial, we step the simulation and change the translation and orientation of the objects at regular intervals. The translation is sampled uniformly from a cylinder surface while the orientation is random. By calling the :meth:`RigidObject.write_root_state_to_sim` method, we directly write these values into the PhysX buffer. Then, we call :meth:`RigidObject.reset` the internal buffers of the objects are reset.

.. literalinclude:: ../../../../source/standalone/tutorials/01_assets/rigid_object.py
   :language: python
   :lines: 92-110
   :linenos:
   :lineno-start: 92

The method :meth:`RigidObject.write_data_to_sim` is included for completeness. It is only necessary when external torques or wrenches are applied to the objects and have to be written into the PhysX buffer. Afterward, we step the simulator and update the articulation object's internal buffers to reflect the robot's new state.

.. literalinclude:: ../../../../source/standalone/tutorials/01_assets/rigid_object.py
   :language: python
   :lines: 111-119
   :linenos:
   :lineno-start: 111


The Code Execution
~~~~~~~~~~~~~~~~~~

Now that we have gone through the code let's run the script and see the result:

.. code-block:: bash

   ./orbit.sh -p source/standalone/tutorials/01_assets/rigid_object.py


This should open a stage with a ground plane, lights, and objects. To stop the simulation,
you can either close the window, or press the ``STOP`` button in the UI, or press ``Ctrl+C``
in the terminal

In this how-to guide, we saw how to spawn rigid objects and wrap them in a :class:`RigidObject` class to initialize all physics handles and that lets us control their position and velocity.
