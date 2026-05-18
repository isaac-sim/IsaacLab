Repository organization
-----------------------

.. code-block:: bash

   IsaacLab
   ├── .vscode
   ├── CONTRIBUTING.md
   ├── CONTRIBUTORS.md
   ├── LICENSE
   ├── isaaclab.bat
   ├── isaaclab.sh
   ├── pyproject.toml
   ├── README.md
   ├── docs
   ├── docker
   ├── source
   │   ├── isaaclab                   # core framework
   │   ├── isaaclab_physx             # PhysX backend (requires Isaac Sim)
   │   ├── isaaclab_ovphysx           # standalone PhysX backend (requires ovphysx)
   │   ├── isaaclab_ov                # OVRTX renderer backend (requires ovrtx)
   │   ├── isaaclab_newton            # Newton backend (kit-less)
   │   ├── isaaclab_assets            # pre-configured robot & sensor assets
   │   ├── isaaclab_tasks             # pre-built RL/IL environments
   │   ├── isaaclab_tasks_experimental # Warp-accelerated environments
   │   ├── isaaclab_rl                # RL library wrappers
   │   ├── isaaclab_mimic             # imitation learning & data generation
   │   ├── isaaclab_teleop            # teleoperation & XR
   │   ├── isaaclab_visualizers       # external visualizer backends
   │   ├── isaaclab_contrib           # community-contributed extensions
   │   ├── isaaclab_experimental      # Warp-accelerated manager and environment variants
   │   ├── extensions                 # legacy Omniverse extension wrappers
   │   └── standalone                 # standalone tutorials & workflows
   ├── scripts
   │   ├── benchmarks
   │   ├── demos
   │   ├── environments
   │   ├── imitation_learning
   │   ├── reinforcement_learning
   │   ├── sim2sim_transfer
   │   ├── tools
   │   └── tutorials
   ├── tools
   └── VERSION

Isaac Lab supports the **PhysX** and **Newton** physics engines through backend packages. The
default PhysX path runs through Isaac Sim, while ``ovphysx`` supports standalone PhysX workflows
without launching Isaac Sim and Newton provides a Warp-native kit-less backend. The ``source``
directory contains all packages that compose Isaac Lab, while ``scripts`` contains standalone
Python applications for training, evaluation, and tooling.
See :doc:`/source/overview/core-concepts/multi_backend_architecture` for details on the backend
system, and :doc:`/source/setup/ecosystem` for a full package-layer overview.

Submodules
~~~~~~~~~~

The packages under ``source/`` are installed as Python packages using
`setuptools <https://setuptools.readthedocs.io/en/latest/>`__. They are organized into three
groups:

**Core, physics backends, and renderers**

* **isaaclab**: The core framework. Provides :mod:`~isaaclab.sim` (simulation context and
  configuration), :class:`~isaaclab.scene.InteractiveScene`, asset and sensor base classes and
  factory interfaces (:mod:`~isaaclab.assets`, :mod:`~isaaclab.sensors`), environment base
  classes (:mod:`~isaaclab.envs`), the manager system (:mod:`~isaaclab.managers`), composable
  MDP term library (:mod:`~isaaclab.envs.mdp`), actuator models (:mod:`~isaaclab.actuators`),
  low-level controllers (:mod:`~isaaclab.controllers`), procedural terrain generation
  (:mod:`~isaaclab.terrains`), and human-input device support (:mod:`~isaaclab.devices`).
* **isaaclab_physx**: PhysX-backed implementations of articulations, rigid bodies, deformable
  objects, Fabric views, the Isaac RTX renderer, and USD spawners. Requires Isaac Sim.
* **isaaclab_ovphysx**: Standalone PhysX backend variant using ``ovphysx`` and the
  TensorBindingsAPI. Requires the ``ovphysx`` package and can run without launching Isaac Sim.
* **isaaclab_ov**: OVRTX renderer backend for RTX-based tiled camera rendering. Requires the
  ``ovrtx`` package and can run without Isaac Sim.
* **isaaclab_newton**: Newton-backed implementations of articulations, rigid bodies, rigid
  object collections, cameras, USD spawners, and the Warp renderer. Supports
  :ref:`kit-less installation <isaaclab-installation-root>` without Isaac Sim.

**Tasks and assets**

* **isaaclab_assets**: Pre-configured :class:`~isaaclab.utils.configclass` dataclasses for a
  wide range of robots (Franka, Unitree, ANYmal, Spot, Allegro, humanoids, quadcopters, and
  more) and sensors (Velodyne, GelSight).
* **isaaclab_tasks**: Registered `gymnasium <https://gymnasium.farama.org/>`__ environments for
  reinforcement and imitation learning, organized as *manager-based* and *direct* tasks across
  locomotion, manipulation, navigation, and classic control domains.
* **isaaclab_tasks_experimental**: Experimental task implementations under active development,
  not yet part of the stable task suite.

**Optional Additions**

* **isaaclab_rl**: Thin wrappers that adapt Isaac Lab environments to the interfaces expected
  by `RSL-RL <https://github.com/leggedrobotics/rsl_rl>`__,
  `skrl <https://skrl.readthedocs.io/>`__,
  `Stable Baselines 3 <https://stable-baselines3.readthedocs.io/>`__, and
  `RL Games <https://github.com/Denys88/rl_games>`__.
* **isaaclab_mimic**: APIs and pre-configured environments for data generation and imitation
  learning, including cuRobo-based motion planners and a dataset-generation pipeline.
* **isaaclab_teleop**: Teleoperation session orchestration with OpenXR / CloudXR support,
  device retargeters for manipulators and humanoids, and gamepad / spacemouse / keyboard input.
* **isaaclab_visualizers**: Supplementary visualizer backends (Isaac Sim Kit, Newton, Rerun, Viser) that
  work with any physics backend.
* **isaaclab_contrib**: Community-contributed features: multirotor assets, TacSL
  visuo-tactile sensors, drone thrust controllers, and more.
* **isaaclab_experimental**: Pre-production core experiments including Warp-accelerated manager
  and environment variants.


Standalone
~~~~~~~~~~

The ``scripts`` directory contains standalone Python applications.
They are structured as follows:

* **benchmarks**: Scripts for benchmarking different framework components.
* **demos**: Demo applications that showcase the core framework :mod:`isaaclab`.
* **environments**: Scripts for running environments defined in :mod:`isaaclab_tasks` with
  different agents (random policy, zero-action policy, teleoperation, scripted state machines).
* **imitation_learning**: Applications for training and evaluating policies with imitation
  learning libraries (e.g. robomimic).
* **reinforcement_learning**: Applications for training and evaluating policies with RL
  libraries (e.g. rsl_rl, rl_games, sb3, skrl).
* **sim2sim_transfer**: Scripts for transferring policies trained in one simulator to another.
* **tools**: Applications for using framework tools such as converting assets and generating
  datasets.
* **tutorials**: Step-by-step tutorials for using the APIs provided by the framework.
