.. _reinforcement-learning:
.. _rl-frameworks:

Reinforcement Learning
======================

Isaac Lab provides a common workflow for training and evaluating policies with
several reinforcement learning (RL) libraries. This page helps you choose a
library, run your first experiment, and understand the options you are most
likely to use.

If you are starting a new robotics project and do not need a feature unique to
another library, start with **RSL-RL**.

Quick start
-----------

From the Isaac Lab repository, train a Cartpole policy with RSL-RL:

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole

Training runs headless by default and writes checkpoints and TensorBoard logs
under ``logs/rsl_rl/``.

To evaluate the latest compatible checkpoint in the interactive visualizer:

.. code-block:: bash

   uv run isaaclab play --rl_library rsl_rl --task Isaac-Cartpole \
       --checkpoint latest --num_envs 32 --viz newton

The ``train`` and ``play`` commands work across the supported RL libraries.

.. figure:: https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/rl_progression_learning_anymald.gif
   :width: 100%
   :align: center
   :alt: ANYmal-D locomotion at RSL-RL training iterations 0, 100, and 299

   An ANYmal-D locomotion policy at three checkpoints during training.

.. _choose-an-rl-library:

Choose an RL library
--------------------

All supported libraries use the same Isaac Lab environments, but they differ in
algorithms, configuration style, and intended workflow. Choose based on the
features your experiment needs, not on a single throughput result.

.. list-table::
   :widths: 18 28 36 18
   :header-rows: 1

   * - Library
     - Choose it for
     - Notable features in Isaac Lab
     - Installation
   * - **RSL-RL**
     - Most GPU-parallel locomotion and manipulation tasks
     - PPO, teacher-student distillation, symmetry augmentation, random network
       distillation, and recurrent and CNN-based policies
     - Included by default
   * - **skrl**
     - JAX workflows or explicit multi-agent training
     - PPO, AMP, IPPO, and MAPPO with PyTorch or JAX
     - ``--extra skrl``
   * - **RL-Games**
     - Existing RL-Games projects or population-based workflows
     - PPO, SAC, A2C, population-based training, distributed training, and Ray
       integration
     - ``--extra rl-games``
   * - **Stable-Baselines3**
     - Familiar baselines and Stable-Baselines3 ecosystem tooling
     - PPO with Stable-Baselines3 callbacks, logging, and model APIs; best suited
       to smaller experiments
     - ``--extra sb3``
   * - **RLinf**
     - RL fine-tuning of Vision-Language-Action models
     - Distributed GR00T and OpenVLA post-training with Ray and FSDP
     - Specialized setup

Install optional dependencies by selecting the corresponding ``uv`` extra when running a command:

.. code-block:: bash

   uv run --extra skrl isaaclab train --rl_library skrl --task Isaac-Cartpole
   uv run --extra rl-games isaaclab train --rl_library rl_games --task Isaac-Cartpole
   uv run --extra sb3 isaaclab train --rl_library sb3 --task Isaac-Cartpole


Typical training workflow
-------------------------

Before committing to a long run, verify the environment and the complete
training path with small smoke tests.

First, check that the task resets and steps without an RL library:

.. code-block:: bash

   uv run isaaclab random_agent --task Isaac-Cartpole --num_envs 16

Then run a few training iterations on a limited number of environments:

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole \
       --num_envs 64 --max_iterations 10

If both commands succeed, remove the smoke-test overrides and use the task's
tuned defaults.

Monitor training
~~~~~~~~~~~~~~~~

Start TensorBoard in a second terminal:

.. code-block:: bash

   uv run python -m tensorboard.main --logdir logs

Focus on trends across multiple iterations:

* **Mean reward** should generally improve, although its absolute value is
  task-specific.
* **Mean episode length** depends on the task. Longer episodes are useful for a
  balance task but not necessarily for a goal-completion task.
* **Throughput** helps identify systems bottlenecks after changes to the number
  of environments, sensors, or rendering.
* **Losses, entropy, KL divergence, and action standard deviation** help reveal
  unstable updates or stalled exploration. Exact metric names vary by library.

Evaluate a policy
~~~~~~~~~~~~~~~~~

Use ``play`` with one of the following checkpoint selectors:

.. list-table::
   :widths: 22 78
   :header-rows: 1

   * - Selector
     - Behavior
   * - ``latest``
     - Loads the highest-step checkpoint from the newest compatible local run.
   * - ``best``
     - Loads the best or final checkpoint when the library records one, and
       otherwise falls back to ``latest``.
   * - ``pretrained``
     - Downloads a published checkpoint for a supported task and backend.
   * - ``/path/to/checkpoint``
     - Loads a specific checkpoint.

For example, play the best local checkpoint and record a short video:

.. code-block:: bash

   uv run --extra video isaaclab play --rl_library rsl_rl \
       --task Isaac-Cartpole --checkpoint best --video --video_length 200

.. hint::

   ``play`` applies the environment config's ``play_mode()`` method: it caps the
   scene at 50 environments and disables observation noise or corruption,
   depending on the workflow. Override that method on a task config to customize
   playback.

.. _pretrained-checkpoints:

Pretrained checkpoints
~~~~~~~~~~~~~~~~~~~~~~

Published pretrained checkpoints are available only for supported core tasks,
and availability may vary by RL library and backend combination. Other
registered tasks, including contributed tasks, are not covered by the
published checkpoint set.

Pass ``--checkpoint pretrained`` to load the published policy matching the
resolved task configuration. The selector does not guarantee that an artifact
exists for every registered task: if the matching artifact has not been
published, the command reports that it is unavailable and exits. In that case,
train the task locally and use ``latest`` or an explicit checkpoint path.

Maintainers can generate the preferred core-task checkpoint matrix with
``scripts/tools/train_and_publish_checkpoints.py``. Use ``--list --all --core``
to list the tasks and backend combinations targeted for publication by the
current source tree. This matrix is not a live check of the remote asset store;
a listed combination becomes usable with ``--checkpoint pretrained`` only
after its checkpoint has been uploaded.

.. code-block:: bash

   uv run python scripts/tools/train_and_publish_checkpoints.py \
       --list --all --core

Resume training
~~~~~~~~~~~~~~~

Resume from the latest compatible checkpoint:

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole \
       --checkpoint latest

When loading an explicit checkpoint, use the same task, agent, ML framework,
and observation/action presets used during training. A checkpoint stores policy
state; it cannot make incompatible observations or actions compatible.

Configure an experiment
-----------------------

Each task registers one or more agent configurations for the libraries it
supports. Start from the closest maintained configuration under
``source/isaaclab_tasks/isaaclab_tasks/<task-family>/.../agents/`` instead of
creating one from scratch.

Discover tasks and agents
~~~~~~~~~~~~~~~~~~~~~~~~~

List registered tasks, then inspect the task-specific help:

.. code-block:: bash

   uv run python scripts/environments/list_envs.py
   uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole --help

Use ``--agent`` to select an alternate registered configuration. For example:

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole \
       --agent rsl_rl_with_symmetry_cfg_entry_point \
       --run_name ppo_with_symmetry

Agent configuration formats are library-specific:

* RSL-RL uses Python configuration classes derived from
  ``RslRlBaseRunnerCfg``.
* RL-Games, skrl, and Stable-Baselines3 use library-specific dictionaries or
  YAML files.
* RLinf uses a dedicated VLA post-training configuration.

When a task offers multiple observation or action presets, choose an agent
whose network matches the selected preset.

.. hint::

   If you omit ``--rl_library``, ``train`` and ``play`` load the task's
   registered ``default_agent``. Most core tasks default to RSL-RL; the
   multi-agent Pendulum task defaults to skrl.

Override configuration with Hydra
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hydra allows you to override environment and agent configurations after the
regular command-line options:

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole \
       --seed 2024 --run_name cartpole_seed_2024 \
       env.actions.joint_effort.scale=10.0 \
       agent.algorithm.learning_rate=0.0005

See :doc:`/source/features/hydra` for quoting rules, list values, and multirun sweeps.

Select backends and presets
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tasks can expose named selectors for physics, rendering, observations, actions,
and other domains. Append selectors without leading dashes:

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole \
       physics=newton_mjwarp

   uv run isaaclab train --rl_library rsl_rl \
       --task Isaac-Cartpole-Camera-Direct \
       physics=newton_mjwarp renderer=newton_renderer presets=rgb

Not every task supports every backend. See
:doc:`/source/concepts/backends_and_presets` for the presets declared by a
task.

.. important::

   Use the same physics, renderer, domain, and observation/action presets for
   training and playback. Changing an observation or action preset can change
   the policy architecture and cause checkpoint shape errors. This is a feature
   and reflects that the selected configuration defines the policy contract.

Scale training
--------------

Isaac Lab collects experience from many environments in parallel.
``--num_envs`` controls the number of simulation instances. Increasing it can
improve sample collection until simulation cost, optimizer structure, or GPU
memory becomes the bottleneck. Use ``--num_envs`` to scale up or down as needed,
but recognize that more environments do not always make training faster.

Isaac Lab also supports distributed training for RSL-RL, RL-Games, and skrl.
See :doc:`/source/features/multi_gpu` for details on scaling training across
multiple GPUs.

Specialized workflows
---------------------

The standard training workflow is sufficient for most tasks. The following
workflows are useful when transferring an existing policy, training multiple
agents, using JAX, or post-training a vision-language-action model.

RSL-RL distillation
~~~~~~~~~~~~~~~~~~~

Distillation transfers behavior from a trained *teacher* policy to a *student*
policy. The student learns to reproduce the teacher's actions, often using a
different observation set or a smaller network. This is useful when the teacher
uses information that will not be available at deployment time. For example,
you can train a state-based policy and distill it to a vision-based policy.

First, train the teacher:

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl \
       --task Isaac-Velocity-Flat-AnymalD

Then select the registered distillation agent and load the teacher checkpoint:

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl \
       --task Isaac-Velocity-Flat-AnymalD \
       --agent rsl_rl_distillation_cfg_entry_point \
       --checkpoint /path/to/teacher.pt

The task and distillation configuration determine which observations are
available to the teacher and student. Make sure the checkpoint was produced by
a compatible teacher configuration.

skrl multi-agent training
~~~~~~~~~~~~~~~~~~~~~~~~~

Multi-agent reinforcement learning trains multiple agents that act in the same
environment. skrl supports two common approaches:

* **MAPPO** uses information from all agents while training the value function,
  but allows each agent to act from its own observations at runtime. This
  centralized-training, decentralized-execution approach can help agents learn
  coordinated behavior.

* **IPPO** trains each agent independently. It is simpler and provides a useful
  baseline, but it does not explicitly use information from the other agents
  during optimization.

For example, train the Shadow Hand handover task with MAPPO:

.. code-block:: bash

   uv run --extra skrl isaaclab train --rl_library skrl \
       --task Isaac-Shadow-Handover-Direct \
       --algorithm MAPPO

Use ``--algorithm IPPO`` to run the corresponding independent-policy workflow.
The selected task must expose a multi-agent interface supported by the
algorithm.

skrl training with JAX
~~~~~~~~~~~~~~~~~~~~~~

skrl can also train policies with JAX instead of PyTorch. This is useful when
integrating with a JAX-based learning stack or experimenting with JAX
compilation and execution:

.. code-block:: bash

   uv run --extra skrl isaaclab train --rl_library skrl \
       --task Isaac-Reach-Franka \
       --ml_framework jax

Install a CUDA-enabled JAX build that matches your platform before using this
workflow. JAX preallocates GPU memory by default, so ensure that the simulation
and learner can share the available memory. See the `skrl installation guide
<https://skrl.readthedocs.io/en/latest/intro/installation.html>`__ and the `JAX
installation guide <https://docs.jax.dev/en/latest/installation.html>`__.

RLinf VLA post-training
~~~~~~~~~~~~~~~~~~~~~~~

Vision-language-action (VLA) models map visual observations and language
instructions to robot actions. RLinf uses reinforcement learning to adapt a
pretrained VLA model to a task, improving its behavior using task rewards
rather than demonstrations alone.

Provide an RLinf experiment configuration and the base model to post-train:

.. code-block:: bash

   uv run --extra rlinf isaaclab train --rl_library rlinf \
       --config_name isaaclab_ppo_gr00t_assemble_trocar \
       --model_path /path/to/base_model

VLA post-training typically requires substantially more GPU memory than
training a small policy from scratch. See :ref:`rlinf-post-training` for
installation, distributed execution, configuration, and evaluation.

Programmatic use
----------------

Downstream applications and automated scripts can invoke the same training
dispatcher used by the command line. Constructing a request directly avoids creating an
``argparse.Namespace`` and provides a stable interface for launching training
from Python:

.. code-block:: python

   from isaaclab_rl import TrainingRequest, train

   train(
       TrainingRequest(
           backend="rsl_rl",
           task="Isaac-Cartpole",
           max_iterations=100,
       )
   )

Use :class:`~isaaclab_rl.entrypoints.PlaybackRequest` with
:func:`~isaaclab_rl.entrypoints.play` for playback. Place options understood by
the selected RL library in ``backend_args``. Place Hydra overrides and preset
selectors in ``hydra_args``.

Troubleshooting
---------------

Out of memory
~~~~~~~~~~~~~

Reduce ``--num_envs`` first. This usually provides the largest memory reduction
without changing the task itself. For camera-based tasks, also consider
reducing the image resolution, number of captured views, rollout length, or
policy size.

Disable the interactive visualizer during performance runs. Remember that
collision geometry, contacts, and deformable bodies consume simulation memory
independently of the policy and rollout buffers.

When using JAX, account for its default GPU-memory preallocation in addition to
the memory required by the simulation.

NaNs in observations or the policy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NaNs often originate in an unstable simulation or invalid reset state and then
propagate through the observations, rewards, and learner. Reproduce the failure
without training when possible, and locate the first value that becomes
non-finite.

Check:

* reset joint positions, velocities, and object poses;
* actuator gains, limits, and action scaling;
* contact geometry, masses, inertias, and unexpectedly large impulses;
* the physics time step, substeps, and solver iterations; and
* reward or observation terms containing unsafe division, normalization, or
  square roots.

Fix the invalid state or calculation rather than masking the problem by
clamping every policy input. Then repeat the training smoke test.

Training is stable but does not learn
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Verify the environment contract before tuning hyperparameters:

* actions control the intended joints;
* observations change as the environment changes;
* rewards increase when the agent makes progress; and
* terminations and resets occur under the expected conditions.

Compare the configuration with the closest maintained task, then change one
factor at a time. For each experiment, record the task, RL library, physics
backend, domain presets, seed, number of environments, agent configuration, and
Hydra overrides. Without this information, apparently small configuration
differences can be mistaken for algorithmic regressions.

Checkpoint does not load
~~~~~~~~~~~~~~~~~~~~~~~~

Verify that the playback or resumed-training command uses the same RL library,
task, agent configuration, ML framework, and observation and action presets as
the original run. Tensor-shape errors usually indicate that one of these
settings changed.

Prefer the ``latest`` or ``best`` checkpoint selectors when available. Manifest
discovery filters incompatible runs before attempting to load their
checkpoints.

Next steps
----------

* :ref:`tutorial-run-rl-training` explains how an Isaac Lab environment is
  wrapped for an RL library.

* :ref:`tutorial-configure-rl-training` walks through registering and selecting
  agent configurations.

* :doc:`/source/features/hydra` covers command-line overrides and sweeps.

* :doc:`/source/features/multi_gpu` covers multi-GPU and multi-node training.

* :doc:`/source/features/reproducibility` covers reproducible experiments.

* :doc:`/source/how-to/capture_sensor_frames` covers image-observation
  diagnostics.

.. _RSL-RL: https://github.com/leggedrobotics/rsl_rl
.. _skrl: https://skrl.readthedocs.io/
.. _RL-Games: https://github.com/Denys88/rl_games
.. _Stable-Baselines3: https://stable-baselines3.readthedocs.io/
.. _RLinf: https://github.com/RLinf/RLinf

.. seealso::

   This page is the source of truth for the ``isaaclab-training-rl-agents`` and
   ``isaaclab-debugging-rl-training`` agent skills
   (`skills/user/train-rl-agents/
   <../../../skills/user/train-rl-agents/SKILL.md>`__,
   `skills/user/debug-rl-training/
   <../../../skills/user/debug-rl-training/SKILL.md>`__). When you change this
   page, update those skills so their guidance stays in sync. See
   :doc:`/source/overview/developer-guide/agent_skills`.
