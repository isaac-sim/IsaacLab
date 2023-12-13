.. _how_to_ray_caster_label:


Using the Ray-Caster
====================

This tutorial demonstrates using the :class:`RayCaster` sensor from the Orbit framework.
Here, we present its usability as a height-scanner, but it can also be used as LiDAR or for any other purpose achieved by ray-casting.


The Code
~~~~~~~~

The tutorial corresponds to the ``run_ray_caster.py`` script in the
``orbit/source/standalone/tutorials/04_sensors`` directory.

.. dropdown:: Code for run_ray_caster.py
   :icon: code

   .. literalinclude:: ../../../../source/standalone/tutorials/04_sensors/run_ray_caster.py
      :language: python
      :emphasize-lines: 84-94, 100-117, 137-138, 118-123
      :linenos:


The Code Explained
~~~~~~~~~~~~~~~~~~

As usual, we design a minimal scene for this example. In addition to the GroundPlane and a Distant Light, for this how-to guide, we add some spheres to the scene that are wrapped as a RigidObject to access their position, orientation, and velocity.

In the following, we will detail the ``add_sensor`` function, responsible for adding the raycaster sensor to the scene.

Adding the ray-caster sensor
----------------------------

As usual, the ray-caster is defined over its config class, :class:`RayCasterCfg`.
Unlike the Camera sensor, the ray-caster operates as a virtual sensor and does not require instantiation within the scene. Instead, it is solely attached to a prim, employed to specify its location, along with a potential offset. This attachment allows the ray-caster to cast a predetermined pattern of rays against a mesh.

.. literalinclude:: ../../../../source/standalone/tutorials/04_sensors/run_ray_caster.py
   :language: python
   :lines: 84-94
   :linenos:
   :lineno-start: 84

Please note that the buffers, physics handles for the RigidObject, and other aspects are initialized when the simulation is played. Thus, we must call ``sim.reset()``.

.. literalinclude:: ../../../../source/standalone/tutorials/04_sensors/run_ray_caster.py
   :language: python
   :lines: 137-138
   :linenos:
   :lineno-start: 137

.. attention::

    Currently, ray-casting is only supported against a single mesh. We are working on extending this functionality to multiple meshes.


Running the simulation loop
---------------------------

In this tutorial, we step the simulation and reset the position of the spheres to initial random positions. For a detailed explanation of the simulation loop, please refer to the :ref:`RigidObject How-to-Guide <_how_to_rigid_objects_label>`.

.. literalinclude:: ../../../../source/standalone/tutorials/04_sensors/run_ray_caster.py
   :language: python
   :lines: 100-117
   :linenos:
   :lineno-start: 100

For the ray-caster sensor, we execute the ray-casting operation. Here, we also time this operation to give an impression of how much time such operations require.

.. literalinclude:: ../../../../source/standalone/tutorials/04_sensors/run_ray_caster.py
   :language: python
   :lines: 118-123
   :linenos:
   :lineno-start: 118

The Code Execution
~~~~~~~~~~~~~~~~~~

Now that we have gone through the code let's run the script and see the result:

.. code-block:: bash

   ./orbit.sh -p source/standalone/tutorials/04_sensors/run_ray_caster.py


This command should open a stage with a ground plane, lights, and some spheres first falling and then rolling on rough terrain with a raycaster pattern next to them.
To stop the simulation, you can either close the window, press the ``STOP`` button in the UI, or press ``Ctrl+C`` in the terminal.

In this guide, we saw how to use a ray-caster sensor. In the following how-to guide, we will see how to use the ray-caster sensor as a faster camera when only geometric information is required.
