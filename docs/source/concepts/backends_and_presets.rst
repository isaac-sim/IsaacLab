.. _backends-and-presets:

Backends and Presets
====================

An Isaac Lab environment describes the robot, scene, sensors, and task. A
**backend** supplies the physics or rendering implementation that brings that
description to life. A **preset** is a named, tested configuration choice that
lets you switch implementations or task modes without editing Python.

In practice, the same environment can run with a different physics engine,
renderer, or observation mode by adding a short selector:

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl \
      --task Isaac-Cartpole-Direct physics=newton_mjwarp

The task stays ``Isaac-Cartpole-Direct``. The preset changes the configuration
used to launch it.


The mental model
----------------

Think of an environment as the experiment and presets as the knobs that select
tested versions of its larger building blocks:

.. code-block:: text

   Environment
   ├── physics=...    Which physics configuration runs the simulation?
   ├── renderer=...   Which renderer produces camera data?
   └── presets=...    Which task-specific mode or config bundle is used?

After configuration is resolved, Isaac Lab's common asset, sensor, and scene
APIs dispatch to the selected backend implementation. Environment code can use
the same public API across PhysX, Newton, and OvPhysX instead of branching on
the active engine throughout the task.

Physics, rendering, and visualization are separate choices. For example, a
camera environment can use Newton physics with the Newton Warp renderer, while
the visualizer is selected independently with ``--viz``.


Find what a task supports
-------------------------

Preset support is task-specific. Before choosing a name, ask the task what it
offers:

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl \
      --task Isaac-Cartpole-Camera-Direct --help

The help output groups names by ``physics=``, ``renderer=``, and ``presets=``.
To browse all registered environments and their presets at once, run:

.. code-block:: bash

   uv run python scripts/environments/list_envs.py --show_presets

An empty preset list is not an error. It means that the environment uses its
registered default configuration and does not expose alternatives. Passing a
name that a task does not list is unsupported and fails during configuration
validation.


Choose a selector
-----------------

.. list-table::
   :widths: 23 32 45
   :header-rows: 1

   * - Selector
     - Example
     - What it changes
   * - ``physics=NAME``
     - ``physics=newton_mjwarp``
     - Selects a physics configuration, including its backend and solver.
   * - ``renderer=NAME``
     - ``renderer=newton_renderer``
     - Selects a renderer configuration for tasks that produce camera data.
   * - ``presets=NAME[,NAME,...]``
     - ``presets=rgb``
     - Applies task-specific choices such as observation modes, camera layouts,
       or compatible configuration bundles.

These are Hydra tokens, so append them without leading dashes. They work with
training, playback, and environment scripts that use Isaac Lab's task
configuration launcher.

Selectors can be combined. This command chooses Newton with the MuJoCo-Warp
solver, the Newton Warp renderer, and RGB observations:

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl \
      --task Isaac-Cartpole-Camera-Direct \
      physics=newton_mjwarp renderer=newton_renderer presets=rgb

Only combine values listed for the task. Some physics, renderer, sensor, and
observation configurations are incompatible, and the task may reject an
invalid combination with a focused error message.


Common backend choices
----------------------

The exact list depends on the environment, but these names follow shared
conventions:

.. list-table:: Physics presets
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Meaning
   * - ``isaacsim_physx``
     - Concrete Isaac Sim PhysX configuration. This is the default for tasks
       whose established default is Isaac Sim PhysX.
   * - ``physx``
     - Automatic PhysX-family selection. Isaac Sim PhysX is used when the
       runtime needs Kit; a configured OvPhysX alternative can be used for
       fully kit-less runs.
   * - ``newton_mjwarp``
     - Newton physics with the MuJoCo-Warp solver.
   * - ``newton_kamino``
     - Newton physics with the Kamino solver. Support is beta and currently
       limited to selected tasks and compatible assets.
   * - ``ovphysx``
     - Concrete OvPhysX configuration for supported kit-less tasks.

.. list-table:: Renderer presets
   :widths: 30 70
   :header-rows: 1

   * - Name
     - Meaning
   * - ``isaacsim_rtx``
     - Concrete Isaac Sim RTX renderer configuration. This is the default for
       tasks that use the multi-backend renderer preset.
   * - ``rtx``
     - Automatic RTX-family selection. Isaac Sim RTX is used when physics,
       visualization, livestreaming, or another runtime choice requires Kit;
       otherwise OVRTX is used for a fully kit-less run.
   * - ``newton_renderer``
     - Newton Warp renderer.
   * - ``ovrtx``
     - Concrete OVRTX renderer configuration for supported kit-less workflows.

Automatic selectors such as ``physics=physx`` and ``renderer=rtx`` are opt-in.
Defaults are concrete so that running a task without selectors is predictable.
A solver is not a separate backend: ``newton_mjwarp`` and ``newton_kamino``
both use Newton but configure different solvers.


Defaults, presets, and fine-tuning
----------------------------------

A preset replaces the complete configuration section at its location; it does
not merge fields from two alternatives. Isaac Lab resolves configuration in
this order:

1. Apply each preset config's ``default`` choice.
2. Apply global choices from ``presets=...``.
3. Apply a preset targeted at a specific path, such as
   ``env.sim.physics=newton_mjwarp``.
4. Apply scalar Hydra overrides, such as ``env.sim.dt=0.002``.

The last step makes it easy to start from a maintained preset and tune one
value:

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl \
      --task Isaac-Cartpole-Direct \
      physics=newton_mjwarp env.sim.dt=0.002

Prefer ``physics=`` and ``renderer=`` for backend choices because they state
intent clearly. Use ``presets=`` for task-specific modes or when one name must
update several matching sections together. Use a path selector only when you
intend to replace one particular section.

.. important::

   Keep behavior-changing presets the same when loading a checkpoint. An
   observation preset can change tensor shapes, and a policy trained with one
   observation mode may not load with another.


How task authors expose choices
-------------------------------

Task authors define alternatives with
:class:`~isaaclab_tasks.utils.hydra.PresetCfg` and choose one as the default.
For a multi-backend task, the preset wrapper belongs in
:class:`~isaaclab.sim.SimulationCfg`:

.. code-block:: python

   from isaaclab.physics import PhysxAutoCfg
   from isaaclab.sim import SimulationCfg
   from isaaclab.utils.configclass import configclass
   from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
   from isaaclab_ov.physics import OvPhysxCfg
   from isaaclab_physx.physics import PhysxCfg
   from isaaclab_tasks.utils import PresetCfg


   @configclass
   class PhysicsCfg(PresetCfg):
       isaacsim_physx = PhysxCfg()
       ovphysx = OvPhysxCfg()
       physx = PhysxAutoCfg(
           isaacsim_physx=isaacsim_physx,
           ovphysx=ovphysx,
       )
       default = isaacsim_physx
       newton_mjwarp = NewtonCfg(solver_cfg=MJWarpSolverCfg())


   @configclass
   class MyEnvCfg:
       sim: SimulationCfg = SimulationCfg(physics=PhysicsCfg())

Keep backend-specific values inside named configurations whenever possible.
This keeps task logic shared and makes every supported choice visible from the
command line.

When backend selection must also change simulation-wide settings such as the
time step, a physics preset may instead contain complete ``SimulationCfg``
alternatives. The ``physics=`` selector recognizes these bundles from their
``physics`` field and applies the complete matching simulation configuration.


Where to go next
----------------

- :doc:`/source/overview/environments` lists environments and their supported
  presets.
- :doc:`/source/features/hydra` covers scalar overrides, preset authoring,
  conflict handling, and advanced configuration behavior.
- :doc:`/source/overview/core-concepts/multi_backend_architecture` explains how
  factories, the physics manager, assets, and sensors dispatch across backends.
- :doc:`/source/overview/core-concepts/physical-backends/index` compares physics
  backend capabilities and links to backend-specific setup guides.
- :doc:`/source/overview/core-concepts/renderers` explains renderer selection and
  implementation details.
