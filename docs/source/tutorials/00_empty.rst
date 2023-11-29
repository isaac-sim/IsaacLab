Creating an empty scene
=======================

This tutorial introduces how to create a standalone python script to set up a simple empty scene in Orbit.
It introduces the two main classes used in the simulator, :class:`AppLauncher` and :class:`SimulationContext`,
that help launch and control the simulation timeline respectively.

Please review `Isaac Sim Interface <https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/tutorial_intro_interface.html#isaac-sim-app-tutorial-intro-interface>`_
and `Isaac Sim Workflows <https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/tutorial_intro_workflows.html#standalone-application>`_
prior to beginning this tutorial.


The Code
~~~~~~~~

The tutorial corresponds to the ``play_empty.py`` script in the ``orbit/source/standalone/demo`` directory.


.. literalinclude:: ../../../source/standalone/demo/play_empty.py
   :language: python
   :emphasize-lines: 11-23,27-31,37-41,44,49-51,64
   :linenos:


The Code Explained
~~~~~~~~~~~~~~~~~~

Launching the simulator
-----------------------

The first step when working with standalone python scripts is to import the :class:`AppLauncher` class. This
class is used to launch the simulation app from the python script. It takes in a :class:`argparse.Namespace` object of configuration parameters
that can be used to configure the launched application. There is a set of standard parameters that can be
added automatically to the parser with the :meth:`AppLauncher.add_app_launcher_args()` method. These parameters include
`headless` (launch app in no-gui mode), `livestream` (determine the streamining option of the app), `ros`
(enables the ROS1 or ROS2 bridge) and `offscreen_render`(enables offscreen-render mode).
For more information on the configuration parameters, please
check the `SimulationApp documentation <https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.kit/docs/index.html#simulation-application-omni-isaac-kit>`_.

Here, we only use the arguments defined by :meth:`AppLauncher.add_app_launcher_args()` which can be set from the
command line. If not explicitly set, the default values are used.

.. literalinclude:: ../../../source/standalone/demo/play_empty.py
   :language: python
   :lines: 11-23
   :linenos:
   :lineno-start: 11

Importing python modules
------------------------

It is important that the simulation app is launched at the start of the script since it loads various python modules
that are required for the rest of the script to run. Once that is done, we can import the various python modules that
we will be using in the script.

.. literalinclude:: ../../../source/standalone/demo/play_empty.py
   :language: python
   :lines: 27-31
   :linenos:
   :lineno-start: 31


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

For this tutorial, we set the physics and rendering time step to 0.01 seconds. Rendering,
here, refers to updating a frame of the current simulation state, which includes the viewport, cameras, and other
existing UI elements.

.. literalinclude:: ../../../source/standalone/demo/play_empty.py
   :language: python
   :lines: 37-39
   :linenos:
   :lineno-start: 37

Next, we set the initial view shown the GUI. The view is defined as the position where the viewpoint is placed (here `[2.5, 2.5, 2.5]`)
and the target to look at which defines the orientation of the viewpoint (here `[0, 0, 0]`).


.. literalinclude:: ../../../source/standalone/demo/play_empty.py
   :language: python
   :lines: 40-41
   :linenos:
   :lineno-start: 40


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

.. literalinclude:: ../../../source/standalone/demo/play_empty.py
   :language: python
   :lines: 43-51
   :linenos:
   :lineno-start: 43

To ensure a safe execution, we wrap the execution loop with checks to ensure that the simulation app is running and
that the simulator is playing. If the simulator is not playing, we simply step the simulator and continue to the next
iteration of the loop.
Lastly, we call the :meth:`simulation_app.close()` method to stop the simulation application and close the window.

.. literalinclude:: ../../../source/standalone/demo/play_empty.py
   :language: python
   :lines: 55-64
   :linenos:
   :lineno-start: 55


The Code Execution
~~~~~~~~~~~~~~~~~~

Now that we have gone through the code, let's run the script and see the result:

.. code-block:: bash

   ./orbit.sh -p source/standalone/demo/play_empty.py


The simulation should be playing and the stage should be rendering. To stop the simulation,
you can either close the window, or press the ``STOP`` button in the UI, or press ``Ctrl+C``
in the terminal.

Now that we have a basic understanding of how to run a simulation, let's move on to the next tutorial
where we will learn how to add a assets to the stage.
