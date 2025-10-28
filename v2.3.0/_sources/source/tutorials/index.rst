.. _tutorials:

Tutorials
=========

Welcome to the Isaac Lab tutorials! These tutorials provide a step-by-step guide to help you understand
and use various features of the framework. All the tutorials are written as Python scripts. You can
find the source code for each tutorial in the ``scripts/tutorials`` directory of the Isaac Lab
repository.

.. note::

    We would love to extend the tutorials to cover more topics and use cases, so please let us know if
    you have any suggestions.

We recommend that you go through the tutorials in the order they are listed here.


Setting up a Simple Simulation
-------------------------------

These tutorials show you how to launch the simulation with different settings and spawn objects in the
simulated scene. They cover the following APIs: :class:`~isaaclab.app.AppLauncher`,
:class:`~isaaclab.sim.SimulationContext`, and :class:`~isaaclab.sim.spawners`.

.. toctree::
    :maxdepth: 1
    :titlesonly:

    00_sim/create_empty
    00_sim/spawn_prims
    00_sim/launch_app

Interacting with Assets
-----------------------

Having spawned objects in the scene, these tutorials show you how to create physics handles for these
objects and interact with them. These revolve around the :class:`~isaaclab.assets.AssetBase`
class and its derivatives such as :class:`~isaaclab.assets.RigidObject`,
:class:`~isaaclab.assets.Articulation` and :class:`~isaaclab.assets.DeformableObject`.

.. toctree::
    :maxdepth: 1
    :titlesonly:

    01_assets/add_new_robot
    01_assets/run_rigid_object
    01_assets/run_articulation
    01_assets/run_deformable_object
    01_assets/run_surface_gripper

Creating a Scene
----------------

With the basic concepts of the framework covered, the tutorials move to a more intuitive scene
interface that uses the :class:`~isaaclab.scene.InteractiveScene` class. This class
provides a higher level abstraction for creating scenes easily.

.. toctree::
    :maxdepth: 1
    :titlesonly:

    02_scene/create_scene

Designing an Environment
------------------------

The following tutorials introduce the concept of manager-based environments: :class:`~isaaclab.envs.ManagerBasedEnv`
and its derivative :class:`~isaaclab.envs.ManagerBasedRLEnv`, as well as the direct workflow base class
:class:`~isaaclab.envs.DirectRLEnv`. These environments bring-in together
different aspects of the framework to create a simulation environment for agent interaction.

.. toctree::
    :maxdepth: 1
    :titlesonly:

    03_envs/create_manager_base_env
    03_envs/create_manager_rl_env
    03_envs/create_direct_rl_env
    03_envs/register_rl_env_gym
    03_envs/run_rl_training
    03_envs/configuring_rl_training
    03_envs/modify_direct_rl_env
    03_envs/policy_inference_in_usd

Integrating Sensors
-------------------

The following tutorial shows you how to integrate sensors into the simulation environment. The
tutorials introduce the :class:`~isaaclab.sensors.SensorBase` class and its derivatives
such as :class:`~isaaclab.sensors.Camera` and :class:`~isaaclab.sensors.RayCaster`.

.. toctree::
    :maxdepth: 1
    :titlesonly:

    04_sensors/add_sensors_on_robot

Using motion generators
-----------------------

While the robots in the simulation environment can be controlled at the joint-level, the following
tutorials show you how to use motion generators to control the robots at the task-level.

.. toctree::
    :maxdepth: 1
    :titlesonly:

    05_controllers/run_diff_ik
    05_controllers/run_osc
