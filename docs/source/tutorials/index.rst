Tutorials
=========

Welcome to the Orbit tutorials! These tutorials provide a step-by-step guide to help you understand
and use various features of the framework. We recommend that you go through the tutorials in the
order they are listed here.

All the tutorials are written as Python scripts. You can find the source code for each tutorial in
the ``source/standalone/tutorials`` directory of the Orbit repository.

.. note::

    We would love to extend the tutorials to cover more topics and use cases, so please let us know if
    you have any suggestions.

.. toctree::
   :maxdepth: 1
   :titlesonly:

   00_sim/index


These tutorials show you how to launch the simulation with different settings and spawn objects in the
simulated scene. They cover the following APIs: :class:`~omni.isaac.orbit.app.AppLauncher`,
:class:`~omni.isaac.orbit.sim.SimulationContext`, and :class:`~omni.isaac.orbit.sim.spawners`.

.. toctree::
    :maxdepth: 1
    :titlesonly:

    01_assets/index


Having spawned objects in the scene, these tutorials show you how to create physics handles for these
objects and interact with them. These revolve around the :class:`~omni.isaac.orbit.assets.AssetBase`
class and its derivatives such as :class:`~omni.isaac.orbit.assets.RigidObject` and
:class:`~omni.isaac.orbit.assets.Articulation`.

.. toctree::
    :maxdepth: 1
    :titlesonly:

    02_scene/index


With the basic concepts of the framework covered, the tutorials move to a more intuitive scene
interface that uses the :class:`~omni.isaac.orbit.scene.InteractiveScene` class. This class
provides a higher level abstraction for creating scenes easily.

.. toctree::
    :maxdepth: 1
    :titlesonly:

    03_envs/index


The following tutorials introduce the concept of environments: :class:`~omni.isaac.orbit.envs.BaseEnv`
and its derivative :class:`~omni.isaac.orbit.envs.RLTaskEnv`. These environments bring-in together
different aspects of the framework to create a simulation environment for agent interaction.

.. toctree::
    :maxdepth: 1
    :titlesonly:

    04_sensors/index


The following tutorials show you how to integrate sensors into the simulation environment. These
tutorials introduce the :class:`~omni.isaac.orbit.sensors.SensorBase` class and its derivatives
such as :class:`~omni.isaac.orbit.sensors.FrameTransformer` and :class:`~omni.isaac.orbit.sensors.RayCaster`.


.. toctree::
    :maxdepth: 1
    :titlesonly:

    05_controllers/index

While the robots in the simulation environment can be controlled at the joint-level, the following
tutorials show you how to use motion generators to control the robots at the task-level.
