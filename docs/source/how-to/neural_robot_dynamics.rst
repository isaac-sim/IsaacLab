:orphan:

.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

.. _newton-neural-robot-dynamics:

Training on neural robot dynamics
=================================

.. currentmodule:: isaaclab_neural

A neural robot dynamics model (NeRD) is a neural network trained on trajectories from a physics solver and run in
its place inside Newton. The models, their training tools and the Newton solvers that run them live in the
``newton-neural`` library, whose documentation explains what a model is and why it steps fast.
:mod:`isaaclab_neural` connects that library to Isaac Lab. A task offers its trained model as one more physics
preset, so you train policies on the model with the rewards, observations, resets and RL training scripts that the
task already has.

This guide covers manager-based tasks on the Newton physics backend and models of articulated robots (a
``SystemArticulation`` in ``newton-neural``). The built-in examples use the ``FlashNeRD`` model family.

.. list-table::
   :widths: 34 66
   :header-rows: 1

   * - Part
     - Purpose
   * - ``scripts/neural/record.py``, ``train.py`` and ``evaluate.py``
     - Record the data of a model, train the model, and compare policies that learned on it with policies that
       learned on the solver. Each script takes ``--task``.
   * - :class:`NeRDRecipeCfg` and :class:`NeRDRecipe`
     - The recipe of a model, as settings and as code. A task registers its settings as
       ``nerd_recipe_cfg_entry_point`` to say how its model is recorded and trained.
   * - :class:`~isaaclab_neural.physics.NeRDSolverCfg` with :class:`~isaaclab_neural.physics.NewtonNeRDManager`
     - Solver configuration and physics manager of a physics preset in which a model takes the place of the
       solver.
   * - :class:`~isaaclab_neural.physics.MJWarpNeRDSolverCfg`
     - Solver configuration of a physics preset in which a model runs inside MuJoCo Warp.

Installation
------------

``newton-neural`` is an optional extra of the Isaac Lab source checkout (see :ref:`installation-optional-extras`).
Pass ``--extra newton-neural`` to the scripts under ``scripts/neural`` and to every command that runs a task on a
model.

.. code-block:: bash

   uv run --extra newton-neural isaaclab train --rl_library rsl_rl --task Isaac-Reach-Franka physics=newton_nerd

The :mod:`isaaclab_neural` package itself is installed with Isaac Lab, and its configuration classes import without
the library. A task that declares a model preset therefore stays importable in every installation. The library is
imported when a model is built or recorded, where a missing extra shows as a ``ModuleNotFoundError`` for
``newton_neural``.

Workflow on the built-in tasks
------------------------------

Three tasks ship a model preset together with the recipe of its model.

.. list-table::
   :widths: 30 20 50
   :header-rows: 1

   * - Task
     - Physics preset
     - What the model stands for
   * - ``Isaac-Reach-Franka``
     - ``newton_nerd``
     - The arm with its joint drives, in free space. The model has no contact input.
   * - ``IsaacContrib-Velocity-Flat-AnymalC``
     - ``newton_nerd``
     - The robot on flat ground, with the actuator network of its legs. A ground probe on every foot is the
       contact input of the model.
   * - ``Isaac-Reorient-Cube-Allegro``
     - ``newton_mjwarp_nerd``
     - The free acceleration of the hand. MuJoCo Warp still solves the contact with the cube.

The workflow has four steps. Run the commands from the repository root, since the model and log paths are relative
to the working directory.

.. tab-set::

   .. tab-item:: Franka reach

      .. code-block:: bash

         # 1. Record data from MuJoCo Warp under random joint targets. No policy is needed.
         uv run --extra newton-neural python scripts/neural/record.py --task Isaac-Reach-Franka

         # 2. Train the model on that data.
         uv run --extra newton-neural python scripts/neural/train.py --task Isaac-Reach-Franka

         # 3. Train a policy on the model, and a second one on MuJoCo Warp to compare with.
         uv run --extra newton-neural isaaclab train --rl_library rsl_rl --task Isaac-Reach-Franka \
            physics=newton_nerd
         uv run isaaclab train --rl_library rsl_rl --task Isaac-Reach-Franka physics=newton_mjwarp

         # 4. Evaluate both policies in MuJoCo Warp.
         uv run --extra newton-neural python scripts/neural/evaluate.py --task Isaac-Reach-Franka

   .. tab-item:: ANYmal C flat

      .. code-block:: bash

         # 1. Record data from MuJoCo Warp under random joint targets. No policy is needed.
         uv run --extra newton-neural python scripts/neural/record.py --task IsaacContrib-Velocity-Flat-AnymalC

         # 2. Train the model on that data.
         uv run --extra newton-neural python scripts/neural/train.py --task IsaacContrib-Velocity-Flat-AnymalC

         # 3. Train a policy on the model, and a second one on MuJoCo Warp to compare with.
         uv run --extra newton-neural isaaclab train --rl_library rsl_rl \
            --task IsaacContrib-Velocity-Flat-AnymalC physics=newton_nerd
         uv run isaaclab train --rl_library rsl_rl --task IsaacContrib-Velocity-Flat-AnymalC \
            physics=newton_mjwarp

         # 4. Evaluate both policies in MuJoCo Warp.
         uv run --extra newton-neural python scripts/neural/evaluate.py \
            --task IsaacContrib-Velocity-Flat-AnymalC

   .. tab-item:: Allegro reorient

      .. code-block:: bash

         # 1. Record data from MuJoCo Warp under random joint targets. No policy is needed.
         uv run --extra newton-neural python scripts/neural/record.py --task Isaac-Reorient-Cube-Allegro

         # 2. Train the model on that data.
         uv run --extra newton-neural python scripts/neural/train.py --task Isaac-Reorient-Cube-Allegro

         # 3. Train a policy on the model, and a second one on MuJoCo Warp to compare with.
         uv run --extra newton-neural isaaclab train --rl_library rsl_rl --task Isaac-Reorient-Cube-Allegro \
            physics=newton_mjwarp_nerd
         uv run isaaclab train --rl_library rsl_rl --task Isaac-Reorient-Cube-Allegro physics=newton_mjwarp

         # 4. Evaluate both policies in MuJoCo Warp.
         uv run --extra newton-neural python scripts/neural/evaluate.py --task Isaac-Reorient-Cube-Allegro

What the steps do
~~~~~~~~~~~~~~~~~

#. **Record.** ``record.py`` builds the scene and the simulation of the task on the physics that the model learns
   from, which is ``newton_mjwarp`` for the three tasks. ``newton-neural`` restarts the robots from sampled states
   and commands random joint position targets, and Isaac Lab steps the physics. The data does not depend on a
   policy. The script saves the settings that it recorded with beside the data.
#. **Train the model.** ``train.py`` fits the model to the dataset, saves it, and then prints its error on robots
   that it was not trained on. For a model in place of the solver the report is the error of the joint coordinates
   after the model ran alone, at the median and at the 90th and 99th percentile. For a model inside MuJoCo Warp it
   is the average error of one predicted frame. Before it trains, the script checks that the held out robots can
   give the report. No simulation starts.
#. **Train a policy.** The physics preset of the model selects it, here ``physics=newton_nerd`` or
   ``physics=newton_mjwarp_nerd``. A second policy trained with ``physics=newton_mjwarp`` is the one to compare
   with.
#. **Evaluate.** ``evaluate.py`` takes, for each of the two presets, the latest RSL-RL run that was trained on
   that physics and saved a policy. It runs each policy on the physics that the model learned from, under the play
   settings of the task and on the robot that the model stands for, for the length of one episode. Every policy
   starts from the same seed. Over the first episode of every environment the script reports the reward, the
   length, how the episodes end, and the command metrics that the task logs. A policy that learned on a good model
   scores like the one that learned on the solver.

Runs on different physics can share an experiment folder, such as runs with ``physics=newton_mjwarp`` and with
another solver preset in ``logs/rsl_rl/reach_franka``. ``evaluate.py`` therefore reads the task settings that every
run saves as ``params/env.yaml`` and takes the latest run that saved a policy and whose physics manager and solver
are those of the preset.

The robot that the model stands for is the task without the events that the recipe names in
:attr:`~NeRDRecipeCfg.unseen_events`, which randomize what the model does not see. ``evaluate.py`` switches these
events off, as ``record.py`` does. Everything else is as the play settings of the task have it, with the number of
environments and the seed that the script is given. The ANYmal C task randomizes the mass and the centre of mass of
the base at startup, and its recipe names these two events, so both policies meet the nominal robot.

Where the files go
~~~~~~~~~~~~~~~~~~

All paths are relative to the working directory.

.. list-table::
   :widths: 34 16 50
   :header-rows: 1

   * - Path
     - Written by
     - Content
   * - ``logs/neural/<name>/dataset``
     - ``record.py``
     - The dataset. Inside it are the untrained model that ``train.py`` fits, and the settings of the recording
       environment and of the recipe as ``params/env.yaml`` and ``params/recipe.yaml``.
   * - ``logs/neural/<name>/flashnerd.pt``
     - ``train.py``
     - The trained model. The physics preset of the task loads this file.
   * - ``logs/rsl_rl/<experiment>``
     - ``isaaclab train``
     - The policies, one folder per run.

``<name>`` is ``reach_franka``, ``anymal_c_flat`` or ``allegro_hand``. The scripts read the model file from the
``checkpoint`` of the preset's solver configuration and place the dataset beside it. Policies trained on a model are
logged apart from those trained on the solver. Their experiments are ``reach_franka_nerd``, ``anymal_c_flat_nerd``
and ``allegro_hand_mjwarp_nerd``, next to ``reach_franka``, ``anymal_c_flat`` and ``allegro_hand``.

``record.py`` does not write into a folder that holds a dataset. Delete the folder, or name another one with
``--dataset``, to record again.

Until the model file exists, a task that selects the preset fails at startup with an error that names the two
scripts (see `Troubleshooting`_), and tools that list the runs of a task leave the preset out. The other presets of
the task are not affected.

Options of the scripts
~~~~~~~~~~~~~~~~~~~~~~

Every script takes ``--task``. ``record.py`` and ``evaluate.py`` start a simulation and also take the launcher
options of Isaac Lab scripts, such as ``--device`` and ``--viz``.

.. list-table::
   :widths: 18 30 52
   :header-rows: 1

   * - Script
     - Option
     - Meaning
   * - ``record.py``
     - ``--num_envs``, ``--num_frames``
     - Robots recorded in parallel and frames recorded per robot. The recipe gives the defaults.
   * -
     - ``--dataset``
     - Dataset folder. The default lies beside the model file of the preset.
   * -
     - ``--seed``
     - Seed of the environment and of the recorder.
   * - ``train.py``
     - ``--dataset``
     - Dataset folder to read, the one of ``record.py`` by default.
   * -
     - ``--checkpoint``
     - File of the trained model. The default is the file that the preset loads.
   * -
     - ``--max_steps``
     - Optimizer steps. The recipe gives the default.
   * -
     - ``--device``, ``--seed``
     - Device to train on, a GPU when there is one, and the seed of the training.
   * - ``evaluate.py``
     - ``--num_envs``
     - Environments that run in parallel.
   * -
     - ``--trained_on``
     - Physics presets whose policies are evaluated. The default is the preset of the model and the preset it
       learned from. Overrides go before this option.
   * -
     - ``--seed``
     - Seed of the environment, which every policy starts from.

The scripts also accept the ``key=value`` overrides of ``isaaclab train``. ``record.py`` and ``train.py`` resolve the
settings of the recipe in the place of an agent configuration, so ``agent.<path>=value`` sets a field of the recipe,
and ``env.<path>=value`` sets a field of the task configuration. The numbers that the recipe of a task adds are
fields as well, such as ``agent.state_speed=0.5`` for the Franka reach task.

``physics=<preset>`` or ``env.sim.physics=<preset>`` selects the physics that ``record.py`` learns from. Without
either, the script records from the :attr:`~NeRDRecipeCfg.reference_preset` of the recipe. The
``params/recipe.yaml`` of the dataset names the physics that was used as its ``reference_preset``. The physics has
to be Newton physics that runs a solver, and a model inside MuJoCo Warp is recorded from MuJoCo Warp only. On
``evaluate.py`` the same two overrides select the physics that the policies run on.

Of the settings of the recipe, only ``agent.train.<name>=value`` changes a training. The others shape the model
and its samples, which ``record.py`` made. ``train.py`` compares them with the ``params/recipe.yaml`` of the
dataset, and when one differs, it stops and names it. Pass ``train.py`` the values that the dataset was recorded
with, or record again. The sizes of the recording and the physics that the data came from are not compared, since
the recording decides them.

.. code-block:: bash

   # A small run that only checks that the steps work together.
   uv run --extra newton-neural python scripts/neural/record.py --task Isaac-Reach-Franka \
      --num_envs 64 --num_frames 96
   uv run --extra newton-neural python scripts/neural/train.py --task Isaac-Reach-Franka \
      --max_steps 200 agent.train.batch_size=256

The scripts report a mistake in the request as an error of the command line, before a simulation or a training
starts. ``record.py`` checks that the task has a recipe, is manager-based and resolves to Newton physics that runs a
solver, which is MuJoCo Warp for a model inside MuJoCo Warp. It also checks that the preset of the recipe runs a
model, that the task has every event named in ``unseen_events``, and that the dataset folder holds no dataset.
``train.py`` checks for the recipe, for a preset that runs a model, for the dataset, for changed settings, and that
the held out robots can give its report. For a model in place of the solver the report needs a frame count in
``train.horizons``, none of them negative, and recorded episodes of that length. ``evaluate.py`` checks for the
recipe, for the events named in ``unseen_events``, and for a policy to evaluate. A preset that the task does not
offer is reported the same way.

Running a task on a model
~~~~~~~~~~~~~~~~~~~~~~~~~

The solver settings of a model preset can be set on the command line like any other field. ``checkpoint`` names
another model file, ``precision`` the precision of the inference (``fp32``, ``tf32``, ``fp16`` or ``bf16``, the
one stored in the model file by default), and ``compile`` whether the step is compiled with ``torch.compile`` (on
by default on CUDA).

.. code-block:: bash

   uv run --extra newton-neural isaaclab train --rl_library rsl_rl --task Isaac-Reach-Franka physics=newton_nerd \
      env.sim.physics.solver_cfg.checkpoint=franka.pt \
      env.sim.physics.solver_cfg.precision=fp16 env.sim.physics.solver_cfg.compile=false

Pass ``--info`` to log the solver that was built.

To watch a policy that learned on a model in MuJoCo Warp, pass its experiment to ``isaaclab play``. For ANYmal C
also pass ``agent.clip_actions=1.5``. The task clips its joint targets under the model preset only, and this option
keeps the actions of the policy within the same range on the solver. ``evaluate.py`` does this by itself.

.. code-block:: bash

   uv run isaaclab play --rl_library rsl_rl --task Isaac-Reach-Franka physics=newton_mjwarp \
      --experiment_name reach_franka_nerd --checkpoint latest --num_envs 4 --viz newton_gl

Notes on the three tasks
~~~~~~~~~~~~~~~~~~~~~~~~

- **Franka reach.** The recipe records the arm without the table, under the joint position drives of the task,
  since random joint targets would drive the arm into it. ``evaluate.py`` keeps the table, as the task has it. The
  model has no contact input, so under ``physics=newton_nerd`` the table does not stop the arm. The preset needs the
  joint position action of the task and does not combine with the ``hold`` preset. Both combinations raise a
  ``ValueError``. A single arm without contact is already cheap to simulate, so this task is the simplest example
  to read and not a showcase of speed.
- **ANYmal C flat.** The contact forces and the joint efforts that rewards read under the model are estimates. The
  foot probes sense one ground plane, so rough terrain needs another contact input.
- **Allegro reorient.** The model predicts the free acceleration of the hand, which MuJoCo Warp computes before it
  resolves contacts, so the cube does not enter the data. A step costs about one MuJoCo Warp step. Choose this kind
  of model for a robot that moves objects, not for speed. The model stands for the nominal hand. Under its preset,
  ``presets=randomized`` keeps the randomization of the cube and of the friction, and switches off that of the
  hand's mass and gains, which the model does not see.

How a task offers a model
-------------------------

A task offers a model as one more alternative of its physics presets, next to its solvers (see
:ref:`backends-and-presets`). The alternative names the model file and the prim path of the robot. There are two
kinds of model, and the kind decides the configuration classes. A *frame* is the time step that a model was trained
on.

.. list-table::
   :widths: 22 39 39
   :header-rows: 1

   * -
     - Model in place of the solver
     - Model inside MuJoCo Warp
   * - Preset name
     - ``newton_nerd``
     - ``newton_mjwarp_nerd``
   * - Physics configuration
     - :class:`~isaaclab_newton.physics.NewtonCfg` whose ``class_type`` names the
       :class:`~isaaclab_neural.physics.NewtonNeRDManager`, with a :class:`~isaaclab_neural.physics.NeRDSolverCfg`
     - :class:`~isaaclab_newton.physics.NewtonCfg` with a :class:`~isaaclab_neural.physics.MJWarpNeRDSolverCfg`
   * - The model predicts
     - The next state of the robot (``PredictionStateIncrement``)
     - The free acceleration of the robot (``PredictionFreeMotion``)
   * - One frame
     - One environment step, ``sim.dt * decimation``
     - One solver step, ``sim.dt / num_substeps``
   * - Contact
     - Learned through the contact input of the model. Bodies other than the robot do not move.
     - MuJoCo Warp solves contacts, joint limits and friction for the robot and the objects together.
   * - Built-in tasks
     - Franka reach, ANYmal C flat
     - Allegro reorient

Use a model in place of the solver for speed, when the robot moves in free space or over static ground. Use a model
inside MuJoCo Warp when the robot moves objects that push back.

A model in place of the solver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Franka reach task declares its preset in ``FrankaReachEnvCfg.__post_init__``.

.. literalinclude:: ../../../source/isaaclab_tasks/isaaclab_tasks/core/reach/config/franka/franka_reach_env_cfg.py
   :language: python
   :start-at: # ``physics=newton_nerd`` steps the arm
   :end-at: self.sim.physics = preset(**vars(self.sim.physics), newton_nerd=nerd)
   :dedent: 8

- The physics is a plain :class:`~isaaclab_newton.physics.NewtonCfg`. ``isaaclab.utils.replace`` copies the MuJoCo
  Warp preset of the task, so the model sees the Newton model that it was recorded on.
- ``class_type`` selects the :class:`~isaaclab_neural.physics.NewtonNeRDManager`, which steps the model once per
  environment step (see `Time under a model`_). The class is named by its path, so the task configuration does not
  import ``newton-neural``.
- ``solver_cfg`` is a :class:`~isaaclab_neural.physics.NeRDSolverCfg`. ``checkpoint`` is the model file, relative
  to the working directory. ``articulation`` is the prim path of the robot.
- ``num_substeps`` is 1, since a model takes one solver step per environment step.
- ``preset(...)`` builds the physics presets of the task again with one more alternative, ``newton_nerd``.

``replace`` copies the ``collision_cfg`` of the solver preset as well. Under a model, Newton's collision pipeline
runs only if a contact sensor or the contact input of the model reads its contacts. In a task with neither, set
``collision_cfg`` to ``None``, since Isaac Lab refuses a collision configuration that no pipeline uses.

The arm of this task is driven by implicit actuators, and the model stands for the arm together with these drives,
so the preset switches nothing else. A model that includes the actuators needs ``sim.use_newton_actuators`` on,
which is the default, since Isaac Lab otherwise runs actuator models in torch, where one that is left on the robot
keeps the joint targets from the model without a trace in Newton.

Two tasks build on the configurations of the built-in examples and do not offer the preset. The OSC variant of the
Franka reach task, ``Isaac-Reach-Franka-OSC``, removes the alternative, since its controller replaces the joint
drives that the model stands for. The navigation task ``IsaacContrib-Navigation-Flat-AnymalC`` takes the ANYmal C
flat task with its default presets.

Time under a model
~~~~~~~~~~~~~~~~~~

A numerical solver takes several small steps per environment step. The task sets ``decimation`` physics steps of
``sim.dt``, and the solver may take substeps within each. A model takes one step, of its frame, and the frame of a
model that replaces the solver is one environment step.

A preset does not change ``sim.dt``, ``decimation`` or ``sim.render_interval`` for this. Its ``class_type`` selects
the :class:`~isaaclab_neural.physics.NewtonNeRDManager` instead, a
:class:`~isaaclab_newton.physics.NewtonManager` that does three things.

- It reads the frame from the model file and makes it the physics step of Newton, so one physics step is one step
  of the model.
- It tells the environment that one ``sim.step()`` covers the whole decimation loop. Once per environment step the
  actions are applied and written to the simulation, Newton detects collisions if something reads contacts, the
  model steps, and the sensors update.
- When the environment declares its decimation, it checks that ``sim.dt * decimation`` equals the frame of the
  model and raises a ``ValueError`` otherwise.

The manager requires two things of its configuration. The solver configuration is a
:class:`~isaaclab_neural.physics.NeRDSolverCfg`, and ``num_substeps`` is 1. The manager raises a ``ValueError``
when it is built on another configuration.

The manager also requires two things of the environment. The environment declares its decimation with
:meth:`~isaaclab_neural.physics.NewtonNeRDManager.set_decimation`, and it calls ``sim.step()`` once per environment
step when :meth:`~isaaclab_neural.physics.NewtonNeRDManager.handles_decimation` says so. The manager-based and the
direct environments of :mod:`isaaclab.envs` do both. A loop that calls ``sim.step()`` once per ``sim.dt`` would
advance one frame per call, so :meth:`~isaaclab_neural.physics.NewtonNeRDManager.step` raises a ``RuntimeError``
while the simulation plays and no environment has declared its decimation (see `Limits`_).

For the ANYmal C flat task this gives the following.

.. list-table::
   :widths: 40 30 30
   :header-rows: 1

   * -
     - ``physics=newton_mjwarp``
     - ``physics=newton_nerd``
   * - ``sim.dt``
     - 0.005 s
     - 0.005 s
   * - ``decimation``
     - 4
     - 4
   * - ``num_substeps``
     - 2
     - 1
   * - Solver steps per environment step
     - 8
     - 1
   * - Physics step of Newton
     - 0.005 s
     - 0.02 s, the frame of the model

The task keeps its own timing, so everything that is computed from it holds under the model as well, such as the
control period, the length of an episode, the rendering interval, and the configurations of tasks that derive from
this one. ``sim.get_physics_dt()`` still returns ``sim.dt``, and
:meth:`~isaaclab_neural.physics.NewtonNeRDManager.get_physics_dt` of the manager returns the frame.

What else a preset switches
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A model stands for whatever moved the robot while it was recorded. A preset has to switch off the parts of the task
that the model replaces and the parts that it cannot see. Where a policy could command joint targets outside the
range that the model was recorded on, the preset clips them as well. The ANYmal C flat task shows the cases.

.. literalinclude:: ../../../source/isaaclab_tasks/isaaclab_tasks/contrib/velocity/config/anymal_c/flat_env_cfg.py
   :language: python
   :start-at: # ``physics=newton_nerd`` steps the robot
   :end-at: self.events.base_com = preset(
   :dedent: 8

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Setting under ``newton_nerd``
     - Reason
   * - The legs' actuator network becomes an ``ImplicitActuatorCfg``
     - The model was recorded with the actuator network running, so it stands for the robot together with that
       network and consumes the joint position targets themselves. An implicit actuator passes the targets
       straight on. Its gains do not move the robot. Isaac Lab uses them only to estimate the joint efforts that
       the torque penalty reads.
   * - The joint position action has a ``clip``
     - The joint targets stay within the range that the model was recorded on, whatever sets the actions. The
       range is the default pose plus and minus ``target_reach`` of the recipe settings, 0.75 rad.
   * - The startup events ``physics_material``, ``add_base_mass`` and ``base_com`` are ``None``
     - The model carries the friction that it was recorded with and stands for the nominal robot. It has no input
       for a friction, mass or centre of mass that changes at startup. The recipe names ``add_base_mass`` and
       ``base_com`` in :attr:`~NeRDRecipeCfg.unseen_events`, so the recording and the evaluation run without them
       as well.
   * - ``default_shape_cfg.margin`` is 0.002 m
     - Contact sensors read forces that the solver estimates from the penetration of shapes. The margin keeps them
       on when a predicted foot hovers just above the ground.
   * - ``precision`` is ``"fp16"``
     - The preset runs the inference of this model in half precision.

Every alternative carries the name of the physics preset. ``physics=newton_nerd`` selects the physics of that name
and, with it, every other alternative of the same name in the task and agent configurations.

The default physics of this task is PhysX. ``preset(...)`` builds the alternatives again, so the task points the
default at ``isaacsim_physx`` once more.

The RSL-RL agent configuration of the task uses the same name for two settings.

.. literalinclude:: ../../../source/isaaclab_tasks/isaaclab_tasks/contrib/velocity/config/anymal_c/agents/rsl_rl_ppo_cfg.py
   :language: python
   :start-at: self.experiment_name = preset(default="anymal_c_flat"
   :end-at: self.clip_actions = preset(
   :dedent: 8

- ``experiment_name`` logs the policies that learned on the model in a folder of their own, where ``evaluate.py``
  finds them.
- ``clip_actions`` clips the actions of the policy to the range that the task clips the joint targets to, since
  1.5 times the action scale of 0.5 is 0.75 rad. The policy then observes the action that was applied.

The task clips the joint targets for every RL library. The experiment name is set for RSL-RL only (see `Limits`_).

A model inside MuJoCo Warp
~~~~~~~~~~~~~~~~~~~~~~~~~~

The Allegro reorient task declares its preset as a field of the class that holds its physics presets.
``AllegroHandManagerEnvCfg.__post_init__`` sets ``sim.physics`` to an instance of this class.

.. literalinclude:: ../../../source/isaaclab_tasks/isaaclab_tasks/core/reorient/config/allegro_hand/allegro_hand_manager_env_cfg.py
   :language: python
   :pyobject: AllegroHandManagerPhysicsCfg

- The class derives from the physics presets that the Allegro reorient tasks share. Every field of a preset class
  is one alternative, so the field ``newton_mjwarp_nerd`` is the alternative of that name.
- :meth:`MJWarpNeRDSolverCfg.from_mjwarp <isaaclab_neural.physics.MJWarpNeRDSolverCfg.from_mjwarp>` copies every
  setting of the MuJoCo Warp solver configuration, so contacts and solver options stay as the task tuned them.
- The physics stays a :class:`~isaaclab_newton.physics.NewtonCfg` with its own manager, substeps and timing. One
  frame of the model is one solver step, so ``sim.dt / num_substeps`` has to equal the frame. The solver checks this
  when it is built.
- Contacts, contact sensors, resets and the options of MuJoCo Warp behave as under
  :class:`~isaaclab_newton.physics.MJWarpSolverCfg`.

The model stands for the nominal hand. Under the preset the task sets two terms of ``presets=randomized`` to
``None``, ``robot_scale_mass`` and ``robot_joint_stiffness_and_damping``, since the model does not see a randomized
mass or randomized gains of the hand. The randomization of the cube and of the friction stays. The RSL-RL agent
configuration has a ``newton_mjwarp_nerd`` alternative for the experiment name only.

The recipe of a model
---------------------

The recipe says how the model of a task is recorded and trained. It has two parts, which live in two files beside
the task configuration. The scripts under ``scripts/neural`` read the settings, and ``record.py`` runs the code.

.. list-table::
   :widths: 22 26 52
   :header-rows: 1

   * - File
     - Class
     - Content
   * - ``nerd_recipe_cfg.py``
     - A subclass of :class:`NeRDRecipeCfg`
     - The settings. Every number of the recipe is a field, so it can be set on the command line and is saved
       with the dataset. The file imports without ``newton-neural``.
   * - ``nerd_recipe.py``
     - A subclass of :class:`NeRDRecipe`
     - The code. It builds the model, the start states and the joint targets with ``newton-neural`` and reads the
       settings as ``self.cfg``.

The settings name the recipe class as :attr:`~NeRDRecipeCfg.class_type`, and ``record.py`` builds it from them with
:func:`~isaaclab.utils.configclass.instantiate`. The settings have the following fields, and a task adds the
numbers of its own recipe.

.. list-table::
   :widths: 24 76
   :header-rows: 1

   * - Field
     - Meaning
   * - :attr:`~NeRDRecipeCfg.class_type`
     - The :class:`NeRDRecipe` that builds the parts that are code, as a class path.
   * - :attr:`~NeRDRecipeCfg.physics_preset`
     - Physics preset of the task that runs the model, ``"newton_nerd"`` by default. The trained model is saved
       where that preset loads it from.
   * - :attr:`~NeRDRecipeCfg.reference_preset`
     - Physics preset that the model learns from, ``"newton_mjwarp"`` by default.
   * - :attr:`~NeRDRecipeCfg.robot`
     - Name of the robot in the scene of the task, ``"robot"`` by default.
   * - :attr:`~NeRDRecipeCfg.unseen_events`
     - Names of the events of the task that randomize what the model does not see, such as the mass of the robot.
       Empty by default. ``record.py`` and ``evaluate.py`` both switch these events off, so both run the robot that
       the model stands for. The ANYmal C recipe names ``add_base_mass`` and ``base_com``.
   * - :attr:`~NeRDRecipeCfg.record`
     - A :class:`NeRDRecordCfg` with the amount of data. ``num_envs`` robots are recorded in parallel for
       ``num_frames`` frames each, in episodes of ``episode_frames`` frames.
   * - :attr:`~NeRDRecipeCfg.train`
     - A :class:`NeRDTrainCfg` with the training settings. They are the optimizer steps ``max_steps``, the
       ``window`` of consecutive frames per sample, ``batch_size``, ``learning_rate``, the ``noise`` added to the
       input state, and the ``horizons`` of the rollouts whose error is reported after training.

The recipe class has four methods. A task overrides the ones it needs.

.. list-table::
   :widths: 24 76
   :header-rows: 1

   * - Method
     - What it does
   * - :meth:`~NeRDRecipe.configure`
     - Changes the resolved task configuration in place before the data is recorded. Only ``record.py`` calls it.
       The default changes nothing. Override it for what only the recording needs. The Franka recipe removes the
       table, which random joint targets would drive the arm into, and the Allegro recipe gives MuJoCo Warp more
       constraint rows. Randomization that the model does not see belongs in
       :attr:`~NeRDRecipeCfg.unseen_events`, which holds for the evaluation as well.
   * - :meth:`~NeRDRecipe.model`
     - Builds the untrained model. The default is a ``FlashNeRD`` of the robot without a contact input.
   * - :meth:`~NeRDRecipe.start_states`
     - Builds the sampler of the states that episodes start from, or returns ``None`` for the default of the
       recorder.
   * - :meth:`~NeRDRecipe.joint_targets`
     - Builds the generator of joint position targets, or returns ``None`` for the default of the recorder.

The last three methods receive a :class:`RecordingSimulation`, the recording environment as a recipe sees it (see
`How recording works`_).

A worked example
~~~~~~~~~~~~~~~~

The settings of the Franka reach task name its recipe class, set the amount of data and the training, and add the
numbers that the recipe reads. ``{DIR}`` stands for the package of the settings file.

.. literalinclude:: ../../../source/isaaclab_tasks/isaaclab_tasks/core/reach/config/franka/nerd_recipe_cfg.py
   :language: python
   :pyobject: FrankaReachNeRDRecipeCfg

The recipe class overrides three of the four methods.

.. literalinclude:: ../../../source/isaaclab_tasks/isaaclab_tasks/core/reach/config/franka/nerd_recipe.py
   :language: python
   :pyobject: FrankaReachNeRDRecipe

- The settings ask for short episodes, which follow the reset cadence of the task, so the model also sees the first
  frames after a reset.
- ``configure`` removes the table for the recording. The model has no contact input, and random joint targets
  would drive the arm into the table.
- ``start_states`` starts the arm anywhere within its joint limits, moving, with the gripper at its default opening.
- ``joint_targets`` mixes four kinds of targets (large jumps that are held, a new target near the arm every frame
  at two reaches, and smooth drift) so that the data covers what an exploring policy and a trained policy command.
- ``model`` is not overridden. The default ``FlashNeRD`` without a contact input fits an arm in free space.

Contact and free motion
~~~~~~~~~~~~~~~~~~~~~~~

The ANYmal C recipe gives its model a contact input, a probe of the ground on every foot. It starts episodes above
the ground from tilted and moving poses, and holds random joint targets around the default pose. Its settings name
the startup randomization of mass and centre of mass in :attr:`~NeRDRecipeCfg.unseen_events`, and the recipe class
does not override ``configure``.

.. literalinclude:: ../../../source/isaaclab_tasks/isaaclab_tasks/contrib/velocity/config/anymal_c/nerd_recipe.py
   :language: python
   :pyobject: AnymalCFlatNeRDRecipe.model
   :dedent: 4

The Allegro recipe builds a model for MuJoCo Warp. Its prediction is the free acceleration, which follows from the
state and the control of one frame, so the backbone keeps one frame of history.

.. literalinclude:: ../../../source/isaaclab_tasks/isaaclab_tasks/core/reorient/config/allegro_hand/nerd_recipe.py
   :language: python
   :pyobject: AllegroHandNeRDRecipe.model
   :dedent: 4

The settings of this recipe name ``physics_preset = "newton_mjwarp_nerd"``, record independent samples with
episodes of one frame, and train on single frames without input noise. The recipe defines its own sampler and
command generator in ``nerd_recipe.py``, which draw each finger from one of three regimes of speed and target
offset. In ``configure`` the recipe gives MuJoCo Warp more constraint rows for the recording, since random poses of
the hand touch themselves.

How recording works
~~~~~~~~~~~~~~~~~~~

``record.py`` resolves the task on the physics to learn from, switches off the events that the recipe names in
:attr:`~NeRDRecipeCfg.unseen_events`, builds the recipe, calls :meth:`~NeRDRecipe.configure`, and builds a
recording environment from the result. This is a plain ``ManagerBasedEnv`` with the following parts.

- The scene and the simulation settings of the task.
- The events of the task that run at startup, which set the task up. Events at resets and intervals are left out,
  since they would move the robots. A task without events records as well.
- One action, the joint position targets of every joint of the robot, passed on as they are.
- No observations, rewards, terminations or commands.

Isaac Lab steps the physics of this environment with ``env.step``, with the actuators of the task running.
``newton-neural`` samples the states that episodes start from, writes them into the environment, and generates the
joint targets (its ``TrajectoryRecorder``). A frame is one environment step for a model in place of the solver. For
a model inside MuJoCo Warp the script records one solver step per frame and makes MuJoCo Warp start every step
from the Newton state, which a restart writes.

After the recording the script saves the configuration of the recording environment and the settings of the recipe
in the dataset folder, as ``params/env.yaml`` and ``params/recipe.yaml``. The saved ``reference_preset`` names the
physics that the data came from.

The methods of a recipe receive the recording environment as a :class:`RecordingSimulation`, the interface that
``record.py`` implements. Besides the members of the ``SimulationBase`` of ``newton-neural``, such as ``system``,
``model``, ``device`` and ``state_bounds()``, it holds what a recipe needs from Isaac Lab.

.. list-table::
   :widths: 24 76
   :header-rows: 1

   * - Member
     - Content
   * - :attr:`~RecordingSimulation.env`
     - The recording environment.
   * - :attr:`~RecordingSimulation.asset`
     - The robot, an Isaac Lab articulation.
   * - :attr:`~RecordingSimulation.articulation`
     - The Newton articulation of every robot, which contact inputs take.
   * - :attr:`~RecordingSimulation.default_joint_pos`
     - The default joint positions of the robot, in the joint order of the model.

Registering the recipe
~~~~~~~~~~~~~~~~~~~~~~

A task names the settings of its recipe in its Gym registration as ``nerd_recipe_cfg_entry_point``, next to its
agent configurations.

.. literalinclude:: ../../../source/isaaclab_tasks/isaaclab_tasks/core/reach/config/franka/__init__.py
   :language: python
   :start-at: gym.register(
   :end-before: gym.register(

Adding a model to your own task
-------------------------------

#. Decide the kind of model. A model in place of the solver suits a robot in free space or on static ground. A
   model inside MuJoCo Warp suits a fixed-base robot that moves objects.
#. Add the physics preset to the task configuration. For a model in place of the solver, copy the preset that the
   model learns from with ``isaaclab.utils.replace``. Set ``class_type`` to
   ``"isaaclab_neural.physics.newton_nerd_manager:NewtonNeRDManager"``, ``solver_cfg`` to a
   :class:`~isaaclab_neural.physics.NeRDSolverCfg` and ``num_substeps`` to 1, and name the alternative
   ``newton_nerd``. For a model inside MuJoCo Warp, replace the solver configuration of the MuJoCo Warp preset by
   one from :meth:`~isaaclab_neural.physics.MJWarpNeRDSolverCfg.from_mjwarp`, and name the alternative
   ``newton_mjwarp_nerd``. Give the solver configuration a ``checkpoint`` under ``logs/neural/<name>/`` and the prim
   path of the robot as ``articulation``. Leave ``sim.dt``, ``decimation`` and ``sim.render_interval`` alone.
#. For a model in place of the solver, switch what it replaces and what it cannot see, each as an alternative
   named after the preset. Replace actuators that are not implicit by an ``ImplicitActuatorCfg``, and leave
   ``sim.use_newton_actuators`` on. Switch off startup randomization of the robot, which the recipe names in
   ``unseen_events``. Give the shapes a small margin if the task reads contact sensors. Set ``collision_cfg`` to
   ``None`` if nothing reads Newton's contacts. Give the joint position action a ``clip`` to the range of joint
   targets that the recipe records, if a policy could command targets outside it.
#. Give the RSL-RL agent configuration an ``experiment_name`` alternative of the same name, so that ``evaluate.py``
   finds the policies that learned on the model. Where the task clips its joint targets, a ``clip_actions``
   alternative of the same range lets the policy observe the action that was applied.
#. Write the recipe in two files beside the task configuration. ``nerd_recipe_cfg.py`` holds the settings, a
   subclass of :class:`NeRDRecipeCfg` with the numbers of the recipe as fields. ``nerd_recipe.py`` holds the code, a
   subclass of :class:`NeRDRecipe` that the settings name as ``class_type``. Set
   :attr:`~NeRDRecipeCfg.physics_preset` if the preset is not ``newton_nerd``. Name the events that randomize what
   the model does not see in :attr:`~NeRDRecipeCfg.unseen_events`, so that the recording and the evaluation run
   without them. Override the methods that the robot needs. A robot that touches the ground needs a contact input
   in :meth:`~NeRDRecipe.model`, and the start states and joint targets should cover what the policy will visit,
   including the first frames after a reset.
#. Register the settings as ``nerd_recipe_cfg_entry_point``.
#. Run ``record.py``, ``train.py``, ``isaaclab train`` with the preset, and ``evaluate.py``, each with ``--task``.

Limits
------

- **Manager-based tasks.** ``record.py`` builds its environment from the scene, simulation and events of a
  manager-based task configuration.
- **Warp frontend.** The environments of the Warp frontend (``isaaclab train --frontend warp``, the classes
  ``ManagerBasedRLEnvWarp`` and ``DirectRLEnvWarp``) call ``sim.step()`` once per ``sim.dt`` and do not declare
  their decimation. A model in place of the solver refuses to step there with a ``RuntimeError``. The limit comes
  from the :class:`~isaaclab_neural.physics.NewtonNeRDManager`, which a model inside MuJoCo Warp does not use.
- **RSL-RL.** Two things are RSL-RL only. The experiment names of the model presets are set in the RSL-RL agent
  configurations of the tasks, and ``evaluate.py`` finds and loads RSL-RL policies. The three tasks also register
  RL-Games and skrl agents, whose configurations have no alternative for the preset, so their runs on a model and on
  the solver share one experiment name. The clip of the joint targets is part of the task and holds for every RL
  library.
- **Joint position targets.** The recording environment commands joint position targets. The actuators of the robot
  have to run as Newton actuators (``sim.use_newton_actuators``, on by default) or be implicit, so that the targets
  reach the control arrays of Newton, where the recorder reads them.
- **One robot per model.** A model in place of the solver moves only the robot it is bound to. Every other body
  stays where it is. Use a model inside MuJoCo Warp when the robot moves objects.
- **Fixed base inside MuJoCo Warp.** A model inside MuJoCo Warp stands for one fixed-base robot per environment and
  takes no contact input.
- **Sensors a model cannot serve.** A model in place of the solver computes no accelerations and no joint reaction
  forces. A task with IMU, PVA or joint wrench sensors, or with ``enable_joint_wrench`` on a robot, is rejected. A
  model inside MuJoCo Warp does not have this limit.
- **Contact sensors.** Under a model in place of the solver, contact sensors read normal forces estimated from the
  penetration of shapes. They keep contact rewards and terminations working and are not meant for force control.
- **External forces.** Forces and torques applied to bodies have no effect on a model in place of the solver.
- **Changes of the model after startup.** Masses, gains, friction and other properties of the Newton model that
  change after startup do not reach the weights of a model. The solver warns once per kind of change with a
  ``NewtonNeuralWarning``. Inside MuJoCo Warp such changes still reach the contact solve.
- **History.** A model keeps a short history per environment. A write of state into an environment, at a reset or
  by an event that pushes the robot, clears that history. Record episodes that start from moving states so that the
  model learns these first frames.
- **Coupled solvers.** Neither solver runs as an entry of Isaac Lab's coupled solver.

Troubleshooting
---------------

.. list-table::
   :widths: 46 54
   :header-rows: 1

   * - Error
     - Cause and remedy
   * - ``the task runs on the model saved at <path>, and that file does not exist``
     - The preset was selected before its model was trained, or from another working directory. Run ``record.py``
       and ``train.py`` for the task from the same working directory, or name another file with
       ``env.sim.physics.solver_cfg.checkpoint=<path>``.
   * - ``the model at <path> steps frames of <frame> s, and one environment step of the task lasts <step> s``
     - The frame of a model in place of the solver differs from ``sim.dt * decimation`` of the task. Train a model
       on frames of the environment step.
   * - ``the checkpoint steps frames of <frame> s, and the task takes solver steps of <step> s``
     - The solver step differs from the frame of the model. For a model inside MuJoCo Warp, set ``sim.dt`` so that
       ``sim.dt / num_substeps`` equals the frame, or train a model on that solver step. For a model that replaces
       the solver, set ``class_type`` of the physics configuration to the path of the
       :class:`~isaaclab_neural.physics.NewtonNeRDManager`.
   * - ``a model advances one environment step per solver step, got num_substeps <n>``
     - Set ``num_substeps`` of the physics configuration of the preset to 1.
   * - ``NewtonNeRDManager steps a model, and the solver of its NewtonCfg is configured by a <class>``
     - The :class:`~isaaclab_neural.physics.NewtonNeRDManager` runs a
       :class:`~isaaclab_neural.physics.NeRDSolverCfg` only. Give the physics configuration one as ``solver_cfg``.
   * - ``no environment declared its decimation to the physics manager``
     - The environment calls ``sim.step()`` once per ``sim.dt``, as those of the Warp frontend do. Run the task on
       the default frontend, whose environments declare their decimation. A loop of your own calls
       ``sim.backend_manager.set_decimation`` and steps once per environment step.
   * - ``the system of the model is a <class>, and an Isaac Lab task runs on models of articulated robots``
     - The model file holds a model of another kind of system. Train a model of the robot of the task.
   * - ``no articulation of the scene lies at <path>``
     - ``articulation`` of the solver configuration is not the prim path of the robot. Set it to
       ``scene.robot.prim_path``.
   * - ``Newton actuators still drive the robot's joints``
     - The model consumes joint targets and stands for the robot with its actuators. Under the preset, replace the
       actuators of the robot by an ``ImplicitActuatorCfg``.
   * - ``it stands for the robot together with its actuators, and sim.use_newton_actuators is off``
     - The model includes the actuators. With the setting off, Isaac Lab runs actuator models in torch, where one
       that is left on the robot keeps the joint targets from the model without a trace in Newton. Set
       ``sim.use_newton_actuators`` to ``True``, its default.
   * - ``the task reads <attribute> from the Newton state``, with ``body_qdd`` or ``body_parent_f``
     - The scene holds IMU, PVA or joint wrench sensors, or a robot with ``enable_joint_wrench``. A model in place of
       the solver cannot serve them. Remove them from the scene under the preset.
   * - ``the task <task> registers no recipe for a neural robot dynamics model``
     - The task has no recipe. Add a :class:`NeRDRecipeCfg` to the Gym registration of the task as
       ``nerd_recipe_cfg_entry_point``.
   * - ``a model is recorded from the scene of a manager-based task``
     - ``record.py`` was given a task of another workflow. Record from a manager-based task.
   * - ``a model is recorded from Newton physics``
     - ``record.py`` was given a physics preset of another backend. Select a Newton preset.
   * - ``a model is recorded from a solver, and physics=<preset> runs a model``
     - ``record.py`` was given the preset of a model, as ``physics=<preset>`` or as
       ``agent.reference_preset=<preset>``. Select the physics that the model learns from.
   * - ``the model of the physics preset <preset> predicts the free acceleration inside MuJoCo Warp and learns it
       from MuJoCo Warp, and physics=<preset> steps with a <class>``
     - ``record.py`` was given another solver as the source of a model inside MuJoCo Warp. Select a MuJoCo Warp
       preset.
   * - ``the recipe of <task> names <events> in unseen_events, and the task has no event of that name``
     - ``record.py`` or ``evaluate.py`` found a name in :attr:`~NeRDRecipeCfg.unseen_events` that is no event of the
       task. Name events of the task configuration.
   * - ``the recipe of <task> names the physics preset <preset>, whose solver is configured by a <class>``
     - :attr:`~NeRDRecipeCfg.physics_preset` names a preset that runs no model. Name the preset of the model.
   * - ``the recipe builds a model with a <prediction>, which does not fit the solver of the physics preset``
     - The model of the recipe and the preset named by :attr:`~NeRDRecipeCfg.physics_preset` are of different
       kinds. A :class:`~isaaclab_neural.physics.NeRDSolverCfg` runs a model that predicts the next state, and a
       :class:`~isaaclab_neural.physics.MJWarpNeRDSolverCfg` one that predicts the free acceleration.
   * - ``<folder> holds a dataset``
     - ``record.py`` does not write into a folder that holds a dataset. Delete the folder, or name another one with
       ``--dataset``.
   * - ``<folder> holds no dataset of record.py``
     - ``train.py`` found no untrained model or no ``params/recipe.yaml`` in the folder. Run ``record.py`` for the
       task first, or name the folder with ``--dataset``.
   * - ``the dataset in <folder> was recorded with other values of <settings> than this run names``
     - Settings of the recipe that shape the model or its samples differ from the ``params/recipe.yaml`` of the
       dataset. Pass ``train.py`` the values that the dataset was recorded with, or record again.
   * - ``the report rolls the model out over the frame counts in train.horizons, got <horizons>``, or
       ``the held out robots have no episode of <count> frames``
     - ``train.py`` cannot give its report, and stops before it trains. ``train.horizons`` is empty, holds a
       negative frame count, or is longer than the recorded episodes. Name frame counts with
       ``agent.train.horizons`` that the recorded episodes cover, or record longer episodes with
       ``agent.record.episode_frames``.
   * - ``No policy trained with physics=<preset> yet, <folder> holds no run on it that saved a policy``, or
       ``no policy of <task> to evaluate``
     - ``evaluate.py`` found no run that was trained on that physics and saved a policy in the experiment folder of
       the preset. Train a policy with ``isaaclab train`` and ``physics=<preset>`` from the same working directory.
   * - ``Isaac Lab runs actuator models of 'robot' in torch``
     - ``record.py`` reads the joint targets from Newton. Run the actuators as Newton actuators, with
       ``sim.use_newton_actuators = True``.
   * - ``ModuleNotFoundError`` for ``newton_neural``
     - The extra is not installed. Pass ``--extra newton-neural`` to ``uv run``.
