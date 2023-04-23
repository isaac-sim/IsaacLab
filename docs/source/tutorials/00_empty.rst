Creating an empty scene
=======================

This tutorial introduces how to create a standalone python script to set up a simple empty scene in Orbit.
It introduces the two main classes used in the simulator, :class:`SimulationApp` and :class:`SimulationContext`,
that help launch and control the simulation timeline respectively.

Additionally, it introduces the :meth:`create_prim()` function that allows users to create objects in the scene.
We will use this function to create two lights in the scene.

Please review `Isaac Sim Interface <https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/tutorial_intro_interface.html#isaac-sim-app-tutorial-intro-interface>`_
and `Isaac Sim Workflows <https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/tutorial_intro_workflows.html#standalone-application>`_
prior to beginning this tutorial.


The Code
~~~~~~~~

The tutorial corresponds to the ``play_empty.py`` script in the ``orbit/source/standalone/demo`` directory.


.. literalinclude:: ../../../source/standalone/demo/play_empty.py
   :language: python
   :emphasize-lines: 20-22,42-43,65-66,70-80,86-87
   :linenos:


The Code Explained
~~~~~~~~~~~~~~~~~~

Launching the simulator
-----------------------

The first step when working with standalone python scripts is to import the ``SimulationApp`` class. This
class is used to launch the simulation app from the python script. It takes in a dictionary of configuration parameters
that can be used to configure the launched application. For more information on the configuration parameters, please
check the `SimulationApp documentation <https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.kit/docs/index.html#simulation-application-omni-isaac-kit>`_.

Here, we launch the simulation app with the default configuration parameters but with the ``headless`` flag
read from the parsed command line arguments.

.. literalinclude:: ../../../source/standalone/demo/play_empty.py
   :language: python
   :lines: 20-22
   :linenos:
   :lineno-start: 20

Importing python modules
------------------------

It is important that the simulation app is launched at the start of the script since it loads various python modules
that are required for the rest of the script to run. Once that is done, we can import the various python modules that
we will be using in the script.

.. literalinclude:: ../../../source/standalone/demo/play_empty.py
   :language: python
   :lines: 28-32
   :linenos:
   :lineno-start: 28


Designing the simulation scene
------------------------------

The next step is to design the simulation scene. This includes creating the stage, setting up the physics scene,
and adding objects to the stage.

Isaac Sim core provides the `SimulationContext <https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.core/docs/index.html#module-omni.isaac.core.simulation_context>`_ that handles various timeline related events (pausing, playing,
stepping, or stopping the simulator), configures the stage (such as stage units or up-axis), and creates the
`physicsScene <https://docs.omniverse.nvidia.com/prod_extensions/prod_extensions/ext_physics.html#physics-scene>`_
prim (which provides physics simulation parameters such as gravity direction and magnitude, the simulation
time-step size, and advanced solver algorithm settings).

.. attention::

    The :class:`SimulationContext` class also takes in the ``backend`` parameter which specifies the tensor library
    (such as ``"numpy"`` and ``"torch"``) in which the returned physics tensors are casted in. Currently, ``orbit``
    only supports ``"torch"`` backend.

For this tutorial, we set the physics time step to 0.01 seconds and the rendering time step to 0.01 seconds. Rendering,
here, refers to updating a frame of the current simulation state, which includes the viewport, cameras, and other
existing UI elements.

.. literalinclude:: ../../../source/standalone/demo/play_empty.py
   :language: python
   :lines: 42-45
   :linenos:
   :lineno-start: 42

Next, we add a ground plane and some lights into the scene. These objects are referred to as primitives or prims in
the USD definition. More concretely, prims are the basic building blocks of a USD scene. They can be considered
similar to an "element" in HTML. For example, a polygonal mesh is a prim, a light is a prim, a material is a prim.
An “xform” prim stores a transform that applies to it's child prims.

Each prim has a unique name, zero or more *properties*, and zero or more *children*. Prims can be nested to create
a hierarchy of objects that define a simulation *stage*. Using this powerful concept, it is possible to create
complex scenes with a single USD file. For more information on USD,
please refer to the `USD documentation <https://graphics.pixar.com/usd/release/index.html>`_.


In this tutorial, we create prims, we use the :meth:`create_prim()` function which takes as inputs the USD prim path,
the type of prim, prim's location and the prim's properties. The prim path is a string that specifies the prim's
unique name in the USD stage. The prim type is the type of prim (such as ``"Xform"``, ``"Sphere"``, or ``"SphereLight"``).
The prim's properties are passed as key-value pairs in a dictionary. For example, in this tutorial, the ``"radius"``
property of a ``"SphereLight"`` prim is set to ``2.5``.


.. literalinclude:: ../../../source/standalone/demo/play_empty.py
   :language: python
   :lines: 50-63
   :linenos:
   :lineno-start: 50


Running the simulation loop
---------------------------

As mentioned earlier, the :class`SimulationContext` class provides methods to control timeline events such as resetting,
playing, pausing and stepping the simulator. An important thing to note is that the simulator is not running until
the :meth:`sim.reset()` method is called. Once the method is called, it plays the timeline that initializes all the
physics handles and tensors.

After playing the timeline, we can start the main loop. The main loop is where we step the physics simulation, rendering
and other timeline events. The :meth:`sim.step()` method takes in a ``render`` parameter that specifies whether the
current simulation state should be rendered or not. This parameter is set to ``False`` when the ``headless`` flag is
set to ``True``.

To ensure a safe execution, we wrap the execution loop with checks to ensure that the simulation app is running and
that the simulator is playing. If the simulator is not playing, we simply step the simulator and continue to the next
iteration of the loop.

.. literalinclude:: ../../../source/standalone/demo/play_empty.py
   :language: python
   :lines: 65-80
   :linenos:
   :lineno-start: 65

Lastly, we call the :meth:`simulation_app.stop()` method to stop the simulation application and close the window.

.. literalinclude:: ../../../source/standalone/demo/play_empty.py
   :language: python
   :lines: 86-87
   :linenos:
   :lineno-start: 86

The Code Execution
~~~~~~~~~~~~~~~~~~

Now that we have gone through the code, let's run the script and see the result:

.. code-block:: bash

   ./orbit.sh -p source/standalone/demo/play_empty.py


This should open a stage with a ground plane and lights spawned at the specified locations.
The simulation should be playing and the stage should be rendering. To stop the simulation,
you can either close the window, or press the ``STOP`` button in the UI, or press ``Ctrl+C``
in the terminal.

Now that we have a basic understanding of how to run a simulation, let's move on to the next tutorial
where we will learn how to add a robot to the stage.
