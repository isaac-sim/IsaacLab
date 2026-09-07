.. _isaac-lab-ecosystem:

Isaac Lab Ecosystem
===================

Isaac Lab is an open-source framework for building, training, and evaluating robot-learning
systems at scale. It provides the reusable layer between your task code and the simulation,
rendering, data-generation, and learning tools around it.

With Isaac Lab, you can describe a robot, scene, sensors, and task once, then combine them with
the backend and learning workflow that fit your project. Common uses include reinforcement
learning, imitation learning, motion planning, teleoperation, and synthetic data generation.

.. important::

   Isaac Lab is **not a simulator**. It is a robot-learning framework that runs on top of a
   supported physics backend. This separation keeps task code reusable while allowing the
   underlying runtime to evolve.


How the pieces fit together
---------------------------

Most projects use Isaac Lab as a set of layers:

1. **Describe the world** with robots, objects, terrains, sensors, and vectorized scenes.
2. **Define the task** with either reusable manager terms or a direct environment class.
3. **Choose a backend** for physics and rendering based on the features and runtime you need.
4. **Connect a workflow** for training, demonstration collection, planning, teleoperation, or
   visualization.
5. **Scale and iterate** using GPU-parallel environments, shared configurations, and standard
   `gymnasium`_ interfaces.

Backend-specific implementations are selected at runtime through Isaac Lab's factory system.
As long as the requested features are supported by the selected backend, the environment code
does not need to import backend-specific modules. See
:doc:`/source/overview/core-concepts/multi_backend_architecture` for the architecture details.

.. image:: ../_static/setup/ecosystem.gif
   :align: center
   :alt: Isaac Lab connecting robots, environments, physics backends, sensors, learning workflows, and deployment


Choose the runtime that fits your project
-----------------------------------------

Isaac Lab supports two physics engines through several backend packages. You do not need to
learn every option before getting started.

.. list-table::
   :header-rows: 1
   :widths: 22 32 46

   * - Runtime
     - Best fit
     - What it provides
   * - **Isaac Sim + PhysX**
     - The full simulation and visualization experience
     - GPU-accelerated rigid-body simulation, deformable objects, USD workflows, Fabric views,
       tiled RTX rendering, ROS/ROS 2, asset importers, and the Omniverse toolchain.
   * - **Standalone PhysX and RTX**
     - Kit-less workflows that still need PhysX or RTX rendering
     - The optional ``ovphysx`` and ``ovrtx`` runtimes, without launching the full Isaac Sim
       application. Feature support depends on the selected runtime.
   * - **Newton**
     - Lightweight, Warp-native, GPU-parallel simulation
     - A kit-less backend for articulations and rigid bodies, with Warp-based rendering and no
       Isaac Sim installation required.

If you are new to Isaac Lab, start with Isaac Sim and PhysX for the broadest feature coverage.
Choose Newton when you specifically want a lightweight, kit-less workflow or a Warp-native
simulation stack.

.. note::

   Isaac Lab 3.0 can be installed without Isaac Sim and used with the Newton backend. See
   :ref:`isaaclab-installation-root` for installation options.


Build tasks your way
--------------------

Isaac Lab provides two environment-authoring styles. Both expose vectorized `gymnasium`_
environments and use the same scene, asset, and sensor interfaces.

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - Workflow
     - Choose it when
     - How it is organized
   * - **Manager-based**
     - You want reusable, configurable task components and a clean separation of concerns.
     - Observations, actions, rewards, commands, events, curricula, and terminations are composed
       from small MDP terms and manager configurations.
   * - **Direct**
     - You want a compact environment or need highly custom step and reset logic.
     - A single environment class implements observations, rewards, terminations, and resets
       directly, similar to the Isaac Gym style.

Manager-based environments are usually the better fit for shared research infrastructure and
families of related tasks. Direct environments are often easier for small experiments and rapid
prototypes. You can choose per task; neither workflow limits which supported backend or RL library
you can use.


Package guide
-------------

The repository is split into focused packages. Most users begin with ``isaaclab`` and add only
the packages required by their workflow.

Core and backends
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Package
     - Role
   * - ``isaaclab``
     - Backend-independent APIs for scenes, assets, sensors, environments, managers, MDP terms,
       actuators, controllers, terrains, devices, configuration, and simulation orchestration.
   * - ``isaaclab_physx``
     - PhysX-backed assets, physics views, sensors, USD spawners, and Isaac RTX rendering. Requires
       Isaac Sim.
   * - ``isaaclab_ov``
     - Standalone Omniverse integrations, including ``ovphysx`` physics and ``ovrtx`` tiled RTX
       rendering for compatible kit-less workflows.
   * - ``isaaclab_newton``
     - Newton-backed assets, physics views, sensors, spawners, and Warp rendering for kit-less
       workflows.

Tasks and workflows
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Package
     - Use it for
   * - ``isaaclab_assets``
     - Ready-to-use configurations for robots and sensors, including manipulators, quadrupeds,
       humanoids, aerial robots, lidar, and tactile sensors.
   * - ``isaaclab_tasks``
     - Registered manager-based and direct `gymnasium`_ environments for training and evaluation.
   * - ``isaaclab_rl``
     - Adapters for `RSL-RL`_, `skrl`_, `Stable Baselines 3`_, and `RL Games`_.
   * - ``isaaclab_mimic``
     - Demonstration generation, imitation-learning datasets, and motion-planning-assisted data
       collection.
   * - ``isaaclab_teleop``
     - Teleoperation sessions, XR integration, and retargeting for manipulators and humanoids.
   * - ``isaaclab_visualizers``
     - Additional visualization through Isaac Kit, Rerun, and Viser across supported backends.
   * - ``isaaclab_contrib``
     - Community-maintained assets, sensors, controllers, and other integrations.
   * - ``isaaclab_experimental``
     - Early-stage APIs and accelerated implementations that are still being evaluated.

Packages such as ``isaaclab_contrib`` and ``isaaclab_experimental`` may evolve faster than the
core API. Check their documentation and changelogs before depending on them in long-lived projects.


What Isaac Lab gives you
------------------------

Isaac Lab brings the common parts of robot-learning projects into one shared framework:

* **Reusable task components** for observations, actions, rewards, events, commands, curricula,
  and terminations.
* **Vectorized GPU simulation** for training many environments in parallel.
* **Multiple physics and rendering backends** behind a common frontend API.
* **A broad sensor and asset library** that can be extended with your own implementations.
* **Integrated learning workflows** for reinforcement learning, imitation learning, planning,
  demonstration collection, and teleoperation.
* **Reproducible configuration** through Hydra presets and command-line overrides.
* **A shared benchmark base** that makes tasks, components, and results easier to compare and
  reuse across projects.

The goal is simple: spend less time rebuilding simulation infrastructure and more time working on
the robot-learning problem itself.


How Isaac Lab relates to earlier Isaac projects
-----------------------------------------------

Isaac Lab builds on lessons from several earlier NVIDIA robotics projects:

* `Isaac Gym`_ :cite:`makoviychuk2021isaac` introduced an end-to-end GPU pipeline for massively
  parallel robot learning with PhysX. It enabled influential work in locomotion
  :cite:`rudin2022learning` :cite:`rudin2022advanced`, dexterous manipulation
  :cite:`handa2022dextreme` :cite:`allshire2022transferring`, and industrial assembly
  :cite:`narang2022factory`.
* `IsaacGymEnvs`_ and `OmniIsaacGymEnvs`_ provided example environment collections for Isaac Gym
  and Isaac Sim. Both are now deprecated in favor of Isaac Lab.
* `Orbit`_ was the research framework that preceded Isaac Lab and was incorporated into the
  current project.
* `Isaac Sim`_ remains the full-featured simulator and application platform. Isaac Lab adds the
  task, environment, learning, and workflow abstractions used to build robot-learning systems on
  top of it.

Today, Isaac Lab is the primary robot-learning framework in the Isaac ecosystem. It combines the
Isaac Sim and PhysX stack with multi-backend support, reusable environments, learning-library
integrations, imitation-learning tools, and teleoperation workflows.


Where to go next
----------------

* Start with :ref:`isaaclab-installation-root` to choose an installation and backend.
* Follow the :ref:`isaac-lab-quickstart` to train and inspect your first task.
* Read :doc:`/source/overview/core-concepts/multi_backend_architecture` before adapting an
  environment to multiple backends.

Isaac Lab is developed in the open with contributions from robotics labs, researchers, and
developers. See the contribution guidelines if you want to report an issue, add a feature, or
share a task with the community.


.. _PhysX: https://developer.nvidia.com/physx-sdk
.. _Newton: https://github.com/newton-physics/newton
.. _Warp: https://github.com/NVIDIA/warp
.. _Isaac Sim: https://developer.nvidia.com/isaac-sim
.. _Isaac Gym: https://developer.nvidia.com/isaac-gym
.. _IsaacGymEnvs: https://github.com/isaac-sim/IsaacGymEnvs
.. _OmniIsaacGymEnvs: https://github.com/isaac-sim/OmniIsaacGymEnvs
.. _Orbit: https://isaac-orbit.github.io/
.. _gymnasium: https://gymnasium.farama.org/
.. _Hydra: https://hydra.cc/
.. _RSL-RL: https://github.com/leggedrobotics/rsl_rl
.. _skrl: https://skrl.readthedocs.io/
.. _Stable Baselines 3: https://stable-baselines3.readthedocs.io/
.. _RL Games: https://github.com/Denys88/rl_games
