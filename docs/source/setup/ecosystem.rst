.. _isaac-lab-ecosystem:

Ecosystem
=========

Isaac Lab is the robot-learning framework between accelerated simulation and the applications that
train, generate data for, and evaluate robot policies. It provides reusable environments, assets,
sensors, controllers, and learning integrations for reinforcement learning, imitation learning,
teleoperation, and motion planning.

Isaac Lab is not itself a simulator. Its common scene, asset, and sensor interfaces can run on
multiple physics and rendering backends. Backend support is task-specific, but the shared interface
lets a supported environment keep the same structure while its preset selects the runtime. See
:doc:`/source/concepts/backend_architecture` for the implementation model.


Capabilities in motion
----------------------

These examples span policy learning, synthetic demonstration generation, humanoid
loco-manipulation, and contact-rich material simulation. Explore the packaged
:doc:`demos </source/setup/demos>` and :doc:`environment catalog </source/setup/environments>`
to run them yourself.

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Humanoid loco-manipulation

      .. image:: https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/locomanipulation_sdg_disjoint_nav_groot_policy_4x.gif
         :width: 100%
         :alt: A humanoid policy navigates around obstacles and places a steering wheel.

      A vision-conditioned policy combines navigation and whole-body manipulation.
      See :doc:`humanoid imitation learning
      </source/features/imitation-learning/humanoids_imitation>`.

   .. grid-item-card:: Synthetic demonstration generation

      .. image:: https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/cube_stack_data_gen_skillgen.gif
         :width: 100%
         :alt: A robot arm generates varied demonstrations for a cube-stacking task.

      SkillGen turns a small set of task demonstrations into diverse training trajectories.
      See :doc:`SkillGen </source/features/imitation-learning/skillgen>`.

   .. grid-item-card:: Newton MPM fluid simulation

      .. raw:: html

         <video autoplay loop muted playsinline preload="metadata" aria-label="A teapot pours MPM water into a bowl" style="width:100%;">
           <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/videos/mpm_teapot_particles.mp4" type="video/mp4">
         </video>

      Newton's MPM solver couples fluid particles, rigid bodies, and interactive controls in the
      packaged **Teapot Fill** demo.

   .. grid-item-card:: Contact-rich material coupling

      .. raw:: html

         <video autoplay loop muted playsinline preload="metadata" aria-label="MPM snowballs strike a stack of rigid crates" style="width:100%;">
           <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/videos/mpm_snowball.mp4" type="video/mp4">
         </video>

      The **Snowball Smash** demo combines deformable MPM material with rigid-body contact.


Applications built on Isaac Lab
-------------------------------

The framework also serves as the simulation and environment layer for higher-level applications.
These projects have their own releases and installation requirements:

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Isaac Lab-Arena
      :link: https://github.com/isaac-sim/IsaacLab-Arena

      Composes scenes, robot embodiments, and tasks into scalable benchmarks, then evaluates
      generalist robot policies across controlled environment variations.

   .. grid-item-card:: NVIDIA Isaac GR00T workflows
      :link: https://github.com/NVIDIA/Isaac-GR00T

      Uses Isaac Lab environments for demonstration collection, synthetic trajectory generation,
      policy training, and closed-loop evaluation of generalist robot models.


Choose a physics backend
------------------------

Isaac Lab supports three physics backends. Start with the backend exposed by the target task's
preset; changing solver families can require task and controller retuning.

.. list-table::
   :header-rows: 1
   :widths: 13 25 31 31

   * - Backend
     - Best fit
     - Benefits
     - Trade-offs
   * - **PhysX**
     - Established Isaac Sim workflows and the broadest feature coverage
     - Mature reference behavior, Isaac RTX rendering, deformables, Fabric, ROS/ROS 2, USD tools,
       and importers
     - Requires the larger Isaac Sim and Kit runtime; large GPU scenes can require explicit
       capacity tuning
   * - **Newton**
     - Kit-less training and workflows that need multiple solver families
     - Warp-native GPU execution without Isaac Sim; provides MJWarp, VBD, MPM, and other solver
       paths
     - Beta integration; feature and task coverage varies by solver, and switching solvers usually
       requires retuning
   * - **OvPhysX**
     - Experimental kit-less PhysX workflows
     - Runs PhysX through a smaller standalone runtime and can pair with kit-less OVRTX rendering
     - Experimental and still gaining feature coverage; configuration is primarily USD-based and
       the runtime cannot share a process with Kit

Physics, camera rendering, and interactive visualization are separate choices. For example, Newton
can use the lightweight Newton Warp renderer or OVRTX for RTX image quality without Isaac Sim.
See :doc:`/source/concepts/physics_backends`, :doc:`/source/concepts/renderers`, and
:doc:`/source/concepts/visualization` for current support matrices and preset commands.


Where does Isaac Lab fit in the Isaac ecosystem?
------------------------------------------------

Over the years, NVIDIA has developed a number of tools for robotics and AI. These tools leverage
the power of GPUs to accelerate simulation both in terms of speed and realism.

`Isaac Gym`_ :cite:`makoviychuk2021isaac` provided a high-performance GPU-based physics simulation
for robot learning built on top of `PhysX`_. Its end-to-end GPU pipeline enabled frame rates
far beyond what CPU-based physics engines could achieve. The tool proved successful across a
number of research projects, including legged locomotion :cite:`rudin2022learning`
:cite:`rudin2022advanced`, in-hand manipulation :cite:`handa2022dextreme`
:cite:`allshire2022transferring`, and industrial assembly :cite:`narang2022factory`.

`Isaac Sim`_ is a general-purpose robot simulation toolkit built on top of `Omniverse`_. It
integrates the capabilities of Isaac Gym while adding high-fidelity rendering, ROS/ROS2,
deformable-object simulation, synthetic data generation, domain randomization, tiled rendering
for vectorized observations, and cloud support via `Isaac Automator`_. With the Isaac Gym legacy
API absorbed into Isaac Sim, NVIDIA also released open-sourced environment collections
`IsaacGymEnvs`_ and `OmniIsaacGymEnvs`_ to showcase the capabilities of these simulators.
Those environment collections are now deprecated in favor of Isaac Lab.

Isaac Lab supersedes `IsaacGymEnvs`_, `OmniIsaacGymEnvs`_, and `Orbit`_ as the single robot
learning framework for Isaac Sim. It retains full access to the PhysX/Isaac Sim stack while
adding the Newton physics backend for kit-less deployments, an expanded sensor suite, imitation
learning tooling, XR teleoperation, and a rich set of pre-built tasks.


Is Isaac Lab a simulator?
-------------------------

At its core, Isaac Lab is **not** a robotics simulator; it is a framework for building robot
learning applications on top of a simulator. An analogous example is `RoboSuite`_, which is
built on top of `MuJoCo`_ for fixed-base manipulation. Other examples include
`MuJoCo Playground`_ (built on `MJX`_) and Isaac Gym (built on `PhysX`_).

Isaac Lab's shared interfaces cover the PhysX, Newton, and experimental OvPhysX backends without
requiring environment code to import backend-specific modules directly. Actual portability depends
on the presets and features supported by each task.

The framework addresses a recurring problem with standalone task implementations: because each
task reimplements the observation, reward, termination, and randomization logic from scratch,
large projects accumulate significant code duplication. Isaac Lab solves this with two
complementary patterns:

* **Manager-based** environments defer every behavioral concern to typed, composable
  *manager* objects (``ObservationManager``, ``RewardManager``, ``TerminationManager``,
  ``EventManager``, ``CurriculumManager``, ``CommandManager``, ``ActionManager``,
  ``RecorderManager``). Each manager is driven by small, reusable MDP term functions that live
  in :mod:`isaaclab.envs.mdp`. This makes it easy to mix and match terms across tasks and to
  test individual components in isolation.

* **Direct** environments implement ``_get_observations``, ``_get_rewards``,
  ``_get_dones``, and ``_reset_idx`` directly in a subclass, similar to the Isaac Gym style.
  They sacrifice some modularity for simplicity and are a natural starting point for rapid
  prototyping.

Both patterns expose a standard `gymnasium`_ ``Env`` interface with vectorized semantics,
so the same environment works unmodified with any of the supported RL libraries.
Configuration management uses `Hydra`_ with a preset system that allows selecting physics
backends and hyperparameter sweeps from the command line.


Why should I use Isaac Lab?
---------------------------

Isaac Lab provides an open-sourced platform for the community to build benchmarks and robot
learning systems together. Sharing a common infrastructure lets teams reuse existing components,
compare results on the same tasks, and focus on the research problems that matter rather than
rebuilding simulation scaffolding from scratch.

Concretely, Isaac Lab offers:

* **Two authoring patterns** — manager-based for modular research and direct for rapid
  prototyping — with a shared :class:`~isaaclab.scene.InteractiveScene` and sensor stack.
* **Multi-backend simulation** — select PhysX, Newton, or experimental OvPhysX from the command
  line when the task provides the corresponding preset.
* **Rich sensor suite** — cameras (tiled and standard), ray-casters, contact sensors, IMU,
  frame transformers, joint-wrench sensors, and visuo-tactile sensors.
* **Imitation learning tooling** — ``isaaclab_mimic`` provides cuRobo-based planners and a
  full dataset-generation pipeline for human demonstration collection.
* **Teleoperation and XR** — ``isaaclab_teleop`` supports OpenXR, CloudXR, gamepads,
  spacemouses, and Haply devices with retargeters for manipulators and humanoids.
* **Hydra configuration management** — hierarchical configs with command-line overrides and a
  preset system for multi-backend environment variants.
* **RL library integrations** — wrappers for RSL-RL, skrl, Stable Baselines 3, and RL Games
  ship in ``isaaclab_rl``.
* **Kit-less deployment** — run policies and simulations using the Newton backend without a
  full Isaac Sim installation. URDF and MJCF command-line conversion can also run kit-less
  when the standalone ``isaacsim-asset-isolated`` importer wheel is installed.

We are working with labs in universities and research institutions to integrate their work into
Isaac Lab and hope that others in the community will join us. If you are interested in
contributing, please reach out to us.


.. _PhysX: https://developer.nvidia.com/physx-sdk
.. _Newton: https://github.com/newton-physics/newton
.. _Isaac Sim: https://developer.nvidia.com/isaac-sim
.. _Omniverse: https://www.nvidia.com/en-us/omniverse/
.. _Isaac Gym: https://developer.nvidia.com/isaac-gym
.. _IsaacGymEnvs: https://github.com/isaac-sim/IsaacGymEnvs
.. _OmniIsaacGymEnvs: https://github.com/isaac-sim/OmniIsaacGymEnvs
.. _Orbit: https://isaac-orbit.github.io/
.. _Isaac Automator: https://github.com/isaac-sim/IsaacAutomator
.. _gymnasium: https://gymnasium.farama.org/
.. _Hydra: https://hydra.cc/
.. _RoboSuite: https://github.com/ARISE-Initiative/robosuite
.. _MuJoCo: https://mujoco.org/
.. _MuJoCo Playground: https://playground.mujoco.org/
.. _MJX: https://mujoco.readthedocs.io/en/stable/mjx.html
