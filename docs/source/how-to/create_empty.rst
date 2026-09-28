:orphan:

Creating an empty scene
=======================

.. currentmodule:: isaaclab

This tutorial shows how to launch and control Isaac Sim simulator from a standalone Python script. It sets up an
empty scene in Isaac Lab and introduces the two main entry points used in the framework, the
:func:`app.launch_simulation` context manager and the :class:`sim.SimulationContext` class.

Please review `Isaac Sim Workflows`_ prior to beginning this tutorial to get
an initial understanding of working with the simulator.


The Code
~~~~~~~~

The tutorial corresponds to the ``create_empty.py`` script in the ``scripts/tutorials/00_sim`` directory.

.. dropdown:: Code for create_empty.py
   :icon: code

   .. literalinclude:: ../../../scripts/tutorials/00_sim/create_empty.py
      :language: python
      :emphasize-lines: 18-27,31,37-44,46-47,51-54
      :linenos:


The Code Explained
~~~~~~~~~~~~~~~~~~

Parsing the command-line arguments
----------------------------------

A standalone script starts by parsing its command-line arguments. The simulator itself is not started yet:
it is launched later, once we know which runtime the simulation configuration needs.

The launch-related command-line options are added to a user-defined :class:`argparse.ArgumentParser`
by passing the parser instance to the :func:`app.add_launcher_args` function. It appends options such as
``--device`` to choose the simulation device, ``--visualizer`` (or ``--viz``) to open a visualizer window,
and ``--livestream`` to stream the simulator. These options are later handed to
:func:`app.launch_simulation`.

.. literalinclude:: ../../../scripts/tutorials/00_sim/create_empty.py
   :language: python
   :start-at: import argparse
   :end-at: args_cli = parser.parse_args()

Importing python modules
------------------------

Isaac Lab configuration classes and utilities can be imported before the simulator is launched. Modules that
need the running simulator, such as Isaac Sim's ``omni.*`` and ``isaacsim.*`` modules or Isaac Lab classes that
work on the USD stage, are imported later, after the simulator has been launched, as shown in the following
tutorials. Here we import the following module:

* :mod:`isaaclab.sim`: A sub-package in Isaac Lab for all the core simulator-related operations.

.. literalinclude:: ../../../scripts/tutorials/00_sim/create_empty.py
   :language: python
   :start-at: from isaaclab.sim import SimulationCfg, SimulationContext
   :end-at: from isaaclab.sim import SimulationCfg, SimulationContext


Configuring the simulation context
----------------------------------

When launching the simulator from a standalone script, the user has complete control over playing,
pausing and stepping the simulator. All these operations are handled through the **simulation
context**. It takes care of various timeline events and also configures the physics scene for
simulation.

In Isaac Lab, the :class:`sim.SimulationContext` class wraps Isaac Sim's simulation stack
to allow configuring the simulation
through Python's ``dataclass`` object and handle certain intricacies of the simulation stepping.

For this tutorial, we set the physics and rendering time step to 0.01 seconds. This is done
by passing these quantities to the :class:`sim.SimulationCfg`, which is then used to create an
instance of the simulation context.

Before the simulation context can be created, the simulator runtime must be launched. This is done with the
:func:`app.launch_simulation` context manager. It inspects the given configuration together with the parsed
command-line arguments and starts only the runtime that they need: the default PhysX physics backend runs
inside Isaac Sim (Kit), so Isaac Sim is started here, while a kitless backend such as Newton needs no Isaac Sim
at all. The simulation runs headless unless a visualizer is requested, for example with ``--viz kit``.
Everything that uses the simulator runs inside the ``with`` block, and the runtime is closed automatically
when the block exits.

.. literalinclude:: ../../../scripts/tutorials/00_sim/create_empty.py
   :language: python
   :start-at: # Configure the simulation
   :end-at: sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])


Following the creation of the simulation context, we have only configured the physics acting on the
simulated scene. This includes the device to use for simulation, the gravity vector, and other advanced
solver parameters. There are now two main steps remaining to run the simulation:

1. Designing the simulation scene: Adding sensors, robots and other simulated objects
2. Running the simulation loop: Stepping the simulator, and setting and getting data from the simulator

In this tutorial, we look at Step 2 first for an empty scene to focus on the simulation control first.
In the following tutorials, we will look into Step 1 and working with simulation handles for interacting
with the simulator.

Running the simulation
----------------------

The first thing, after setting up the simulation scene, is to call the :meth:`sim.SimulationContext.reset`
method. This method plays the timeline and initializes the physics handles in the simulator. It must always
be called the first time before stepping the simulator. Otherwise, the simulation handles are not initialized
properly.

.. note::

   :meth:`sim.SimulationContext.reset` is different from :meth:`sim.SimulationContext.play` method as the latter
   only plays the timeline and does not initializes the physics handles.

After playing the simulation timeline, we set up a simple simulation loop where the simulator is stepped repeatedly.
The loop condition :meth:`sim.SimulationContext.is_headless_or_exist_active_visualizer` keeps it running
forever when no visualizer is open, and until the last visualizer window is closed otherwise.
The method :meth:`sim.SimulationContext.step` takes in as argument :attr:`render`, which dictates whether
the step includes updating the rendering-related events or not. By default, this flag is set to True.

.. literalinclude:: ../../../scripts/tutorials/00_sim/create_empty.py
   :language: python
   :start-at: # Play the simulator
   :end-at: sim.step()

Exiting the simulation
----------------------

There is no explicit shutdown call. When the simulation loop ends and the ``with launch_simulation(...)``
block exits, the simulator runtime is stopped and its window is closed.


The Code Execution
~~~~~~~~~~~~~~~~~~

Now that we have gone through the code, let's run the script and see the result:

.. code-block:: bash

   python scripts/tutorials/00_sim/create_empty.py --viz kit


The simulation should be playing, and the stage should be rendering. To stop the simulation,
you can either close the window, or press ``Ctrl+C`` in the terminal.

.. figure:: ../_static/tutorials/tutorial_create_empty.jpg
    :align: center
    :figwidth: 100%
    :alt: result of create_empty.py

Passing ``--help`` to the above script will show the different command-line arguments added
earlier by the :func:`app.add_launcher_args` function. To run the script headless, omit the visualizer
selection:

.. code-block:: bash

   python scripts/tutorials/00_sim/create_empty.py

If a config or command selects a visualizer, force-disable all visualizers with
``--visualizer none`` or ``--viz none``.

Now that we have a basic understanding of how to run a simulation, let's move on to the
following tutorial where we will learn how to add assets to the stage.

.. _`Isaac Sim Workflows`: https://docs.isaacsim.omniverse.nvidia.com/latest/introduction/workflows.html
.. _carb: https://docs.omniverse.nvidia.com/kit/docs/carbonite/latest/index.html
