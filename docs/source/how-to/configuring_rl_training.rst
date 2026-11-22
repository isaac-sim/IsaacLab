:orphan:

.. _tutorial-configure-rl-training:

Configuring an RL Agent
=======================

.. currentmodule:: isaaclab

Each task registers an agent configuration for every supported RL library. The
configuration selects the algorithm, network, and training settings.

This tutorial creates an agent configuration and registers it with the task using
``gymnasium.register``, building on :ref:`tutorial-register-rl-env-gym`.
For library choices and training commands, see :ref:`reinforcement-learning`.

The Code
--------

As an example, we will look at the configuration included for the task ``Isaac-Cartpole``
in the ``isaaclab_tasks`` package. This is the same task that we used in the
:ref:`tutorial-run-rl-training` tutorial.

.. literalinclude:: ../../../source/isaaclab_tasks/isaaclab_tasks/core/cartpole/__init__.py
   :language: python
   :lines: 54-70

The Code Explained
------------------

Under the attribute ``kwargs``, we can see the configuration for the different learning libraries.
The key is the name of the library and the value is the path to the configuration instance.
This configuration instance can be a string, a class, or an instance of the class.
For example, the value of the key ``"rl_games_cfg_entry_point"`` is a string that points to the
configuration YAML file for the RL-Games library. Meanwhile, the value of the key
``"rsl_rl_cfg_entry_point"`` points to the configuration class for the RSL-RL library.

The pattern used for specifying an agent configuration class follows closely to that used for
specifying the environment configuration entry point. This means that while the following
are equivalent:


.. dropdown:: Specifying the configuration entry point as a string
   :icon: code

   .. code-block:: python

      from . import agents

      gym.register(
         id="Isaac-Cartpole",
         entry_point="isaaclab.envs:ManagerBasedRLEnv",
         disable_env_checker=True,
         kwargs={
            "env_cfg_entry_point": f"{__name__}.cartpole_manager_env_cfg:CartpoleEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
         },
      )

.. dropdown:: Specifying the configuration entry point as a class
   :icon: code

   .. code-block:: python

      from . import agents

      gym.register(
         id="Isaac-Cartpole",
         entry_point="isaaclab.envs:ManagerBasedRLEnv",
         disable_env_checker=True,
         kwargs={
            "env_cfg_entry_point": f"{__name__}.cartpole_manager_env_cfg:CartpoleEnvCfg",
            "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.CartpolePPORunnerCfg,
         },
      )

The first code block is the preferred way to specify the configuration entry point.
The second code block is equivalent to the first one, but it leads to import of the configuration
class which slows down the import time. This is why we recommend using strings for the configuration
entry point.

The reinforcement learning entrypoints are configured by default to read the
``<library_name>_cfg_entry_point`` from the ``kwargs`` dictionary to retrieve the configuration instance.

For instance, the following code block shows how the Stable-Baselines3 training implementation
reads the configuration instance:

.. dropdown:: Code for train_sb3.py with SB3
    :icon: code

    .. literalinclude:: ../../../source/isaaclab_rl/isaaclab_rl/entrypoints/backends/train_sb3.py
      :language: python
      :linenos:
      :emphasize-lines: 71-76, 112

The argument ``--rl_library`` selects the reinforcement learning library. The ``--agent``
argument selects the library-specific configuration entry point from the ``kwargs``
dictionary, so you can manually specify alternate configuration instances.

The Code Execution
------------------

Since the cartpole balancing task offers several RSL-RL configuration instances,
we can use the ``--agent`` argument to specify the configuration instance to use.

* Training with the standard PPO configuration:

  .. tab-set::

     .. tab-item:: uv (Recommended)

        .. code-block:: bash

          # standard PPO training
          uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole \
            --run_name ppo

* Training with the PPO configuration with symmetry augmentation:

  .. tab-set::

     .. tab-item:: uv (Recommended)

        .. code-block:: bash

          # PPO training with symmetry augmentation
          uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole \
            --agent rsl_rl_with_symmetry_cfg_entry_point \
            --run_name ppo_with_symmetry_data_augmentation

          # you can use hydra to disable symmetry augmentation but enable mirror loss computation
          uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole \
            --agent rsl_rl_with_symmetry_cfg_entry_point \
            --run_name ppo_without_symmetry_data_augmentation \
            agent.algorithm.symmetry_cfg.use_data_augmentation=false

The ``--run_name`` argument is used to specify the name of the run. This is used to
create a directory for the run in the ``logs/rsl_rl/cartpole`` directory.

When continuing RSL-RL training with a different policy learning rate, use
``--checkpoint /path/to/model.pt --no-resume_optimizer_state``. This restores model weights and the
iteration count while keeping the newly configured policy optimizer and its learning rate.
By default, ``--resume_optimizer_state`` restores the checkpoint's optimizer state as usual. RND models and their
separate optimizer retain their checkpoint state.
For distillation, this option requires a student checkpoint containing both student and teacher;
omit it when initializing a new student from a PPO teacher checkpoint.

Installed downstream tasks are discovered through the ``isaaclab.tasks`` entry-point group.
Custom launchers can call ``isaaclab.cli.load_external_tasks()`` before looking up task IDs,
as the built-in CLI and direct RL workers do. A package that fails to import is reported with a
warning instead of stopping discovery.

Training with SAC
~~~~~~~~~~~~~~~~~

Several tasks provide a SAC agent entry point next to their PPO agent: ``Isaac-Cartpole``, ``Isaac-Ant``,
``Isaac-Humanoid``, ``Isaac-Open-Drawer-Franka``, ``Isaac-Reorient-Cube-Allegro``, and the flat and rough
ANYmal-D and G1 velocity tasks. Their configurations use the FlashSAC recipe of RSL-RL, including residual BatchNorm
networks, categorical critics, held exploration and return scaling, with settings tuned per task. The environment's
rewards, commands and events are shared with its PPO agent. The G1 agents add left/right symmetry augmentation,
which they share with the G1 PPO teacher.

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl \
      --task Isaac-Velocity-Flat-G1 --agent rsl_rl_sac_cfg_entry_point \
      physics=newton_mjwarp visualizer=none

The rough ANYmal-D and G1 agents were tuned with ``--num_envs 1024``.

The actor's ``action_range`` specifies bounds in the raw coordinates accepted by the
environment. ``clip_actions=None`` preserves that range.

Use the same agent entry point with the playback and export commands. Published
``pretrained`` checkpoints are PPO policies and cannot be loaded by ``OffPolicyRunner`` agents.

Training the G1 velocity tasks with SuperMimic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SuperMimic uses the same flat and rough G1 velocity MDPs with noise-free physical observation groups.
Select its agent and the shared ``state_obs`` observation preset together:

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl \
      --task Isaac-Velocity-Rough-G1 --agent rsl_rl_supermimic_cfg_entry_point \
      presets=state_obs physics=newton_mjwarp visualizer=none

``rsl_rl_supermimic_cfg_entry_point`` follows the settings of the official SuperMimic code.
``rsl_rl_blinkmimic_cfg_entry_point`` trains with the BlinkMimic settings instead: Muon, a raised
actor learning rate, mirrored world-model batches, and a smaller world model and policy batch.

The agent configuration enables ``final_obs_every_step`` before environment creation.
These observations describe the physical transition before resets, command resampling
and interval pushes. The environment signals this capture explicitly. SuperMimic cuts
world-model training windows at continuing-state edits and retains the physical successor.

The differentiable velocity task owns the state coordinates, deployable actor features,
reward terms and contact probabilities. Each frame of the actor history is the default policy
observation, including the command, so an exported policy reads the same inputs as the PPO
policy. Feet and the height scanner follow from differentiable
forward kinematics of the simulated articulation. The actor's height scan samples a heightfield
of the terrain mesh, ray-cast once at startup, so it matches the ray-cast scan except within
2.5 cm of vertical steps. The world model sees the terrain through the root and sole heights
above it. Joint torque rewards use a PD approximation.
Acceleration rewards use policy-rate differences. Contact timers advance analytically
at the policy timestep using predicted event labels.

Pass the same agent and ``presets=state_obs`` to playback, which also writes JIT and ONNX exports.
The preset collects one step before applying fractional learner updates. See the
`RSL-RL learned dynamics configuration <https://leggedrobotics.github.io/rsl_rl/guide/configuration.html#learned-dynamics-configuration>`__
for checkpoint and deployment inputs. LEAPP requires an MLPModel policy and rejects this actor representation.

``Isaac-Motion-Tracking-G1`` provides a SuperMimic agent for the same reference-motion MDP as its PPO agent:

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl \
      --task Isaac-Motion-Tracking-G1 --agent rsl_rl_supermimic_cfg_entry_point \
      presets=state_obs visualizer=none

The world model predicts the root link and joint state. Tracked-body poses and velocities follow from differentiable
forward kinematics of the simulated articulation, so imagined tracking rewards use the environment's reward
definitions. The reference frame advances one frame per imagined step. The actor receives the default policy's
proprioceptive and anchor-orientation terms in its history and the reference joint command as its goal.

.. _Stable-Baselines3: https://stable-baselines3.readthedocs.io/en/master/
.. _RL-Games: https://github.com/Denys88/rl_games
.. _RSL-RL: https://github.com/leggedrobotics/rsl_rl
.. _SKRL: https://skrl.readthedocs.io
