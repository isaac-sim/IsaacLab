Reproducibility and Determinism
-------------------------------

Given the same hardware and Isaac Sim (and consequently PhysX) version, the simulation produces
identical results for scenes with rigid bodies and articulations. However, the simulation results can
vary across different hardware configurations due to floating point precision and rounding errors.
At present, PhysX does not guarantee determinism for any scene with non-rigid bodies, such as cloth
or soft bodies. For more information, please refer to the `PhysX Determinism documentation`_.

Based on above, Isaac Lab provides a deterministic simulation that ensures consistent simulation
results across different runs. This is achieved by using the same random seed for the
simulation environment and the physics engine. At construction of the environment, the random seed
is set to a fixed value using the :meth:`~isaaclab.utils.seed.configure_seed` method. This method sets the
random seed for both the CPU and GPU globally across different libraries, including PyTorch and
NumPy.

In the included workflow scripts, the seed specified in the learning agent's configuration file or the
command line argument is used to set the random seed for the environment. This ensures that the
simulation results are reproducible across different runs. The seed is set into the environment
parameters :attr:`isaaclab.envs.ManagerBasedEnvCfg.seed` or :attr:`isaaclab.envs.DirectRLEnvCfg.seed`
depending on the manager-based or direct environment implementation respectively.

App-level deterministic experience selection is exposed through ``AppLauncher``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``--deterministic`` flag is provided by :meth:`isaaclab.app.AppLauncher.add_app_launcher_args`.
When used with the default experience selection logic in a compatible headless launch, AppLauncher
automatically selects ``isaaclab.python.headless.determinism.kit``.

**Strict PyTorch determinism** (calling :meth:`~isaaclab.utils.seed.configure_seed` with
``torch_deterministic=True``) is wired into the **RL-Games** training script only. Other frameworks
still honor ``--seed`` / agent configuration for the environment; use
:meth:`~isaaclab.utils.seed.configure_seed` from your own entry point if you need the same PyTorch-wide
behavior elsewhere.

To enable deterministic rendering/app settings, launch workflows with ``--deterministic``
**and** ``--enable_cameras`` **and** ``--headless`` (without livestream/XR):

.. code-block:: bash

  ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Cartpole-v0 \
    --enable_cameras --headless --deterministic

You can still pass ``--experience isaaclab.python.headless.determinism.kit`` explicitly if you prefer.

Gymnasium registry (``gym.register``) and training scripts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Isaac Lab tasks are registered with `Gymnasium <https://gymnasium.farama.org/>`_ using ``gym.register``.
Besides ``id`` and ``entry_point``, the ``kwargs`` dict lists **string entry points** that tell each
training script where to load configs from—for example:

* ``env_cfg_entry_point`` — environment configuration class (always present for Isaac tasks).
* ``rl_games_cfg_entry_point``, ``sb3_cfg_entry_point``, ``skrl_cfg_entry_point``, ``rsl_rl_cfg_entry_point`` —
  optional; **only keys that appear in ``kwargs`` are valid** for that task id.

When you run ``scripts/reinforcement_learning/<framework>/train.py --task <TASK_ID>``, the script resolves
``<framework>_cfg_entry_point`` (or the name you pass with ``--agent``) against the registry. If the task
was registered **without** that key—for example ``Isaac-Cartpole-RGB-v0`` currently lists only
``rl_games_cfg_entry_point``—you will get a ``ValueError`` such as “Could not find configuration …
``sb3_cfg_entry_point``”. To use another framework you must either pick a task that registers that entry
point or extend ``gym.register(..., kwargs={...})`` for that task with matching agent YAML/Python configs.

Regression test for training scripts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The file ``source/isaaclab_tasks/test/test_train_scripts_deterministic.py`` checks that:

* ``AppLauncher`` exposes ``--deterministic``;
* the RL-Games training script calls :meth:`~isaaclab.utils.seed.configure_seed` after ``Runner`` construction;
* (optional, heavy) for RL-Games on ``Isaac-Cartpole-RGB-v0``, two runs **without** ``--deterministic`` diverge
  in logged ``rewards/iter``, while two runs **with** ``--deterministic`` match.

Run all tests in that file from the repository root:

.. code-block:: bash

  cd /path/to/isaaclab
  ./isaaclab.sh -p -m pytest source/isaaclab_tasks/test/test_train_scripts_deterministic.py

**Heavy reproducibility test (starts Kit and trains multiple times).** It is skipped unless you set
``ISAACLAB_RUN_DETERMINISM_TRAIN_TEST=1``:

.. code-block:: bash

  cd /path/to/isaaclab
  ISAACLAB_RUN_DETERMINISM_TRAIN_TEST=1 ./isaaclab.sh -p -m pytest \
    source/isaaclab_tasks/test/test_train_scripts_deterministic.py -k reproducibility

**Pytest ``-k`` (keyword expression).** ``-k`` filters which tests run by matching their **names** (test
function name, class name, and parametrized case ids). It is **not** the same as ``-m`` (markers).

* ``-k reproducibility`` — runs only tests whose full name contains that substring (for example the
  heavy RL-Games test ``test_rl_games_deterministic_flag_affects_rewards_reproducibility``).
* ``-k "not reproducibility"`` — runs the lighter checks and **skips** the heavy training comparison.
* Inspect names with ``./isaaclab.sh -p -m pytest ... --collect-only -q``; combine filters with
  ``and`` / ``or`` / ``not`` and parentheses as needed.

For results on our determinacy testing for RL training, please check the GitHub Pull Request `#940`_.

.. tip::

  Due to GPU work scheduling, there's a possibility that runtime changes to simulation parameters
  may alter the order in which operations take place. This occurs because environment updates can
  happen while the GPU is occupied with other tasks. Due to the inherent nature of floating-point
  numeric storage, any modification to the execution ordering can result in minor changes in the
  least significant bits of output data. These changes may lead to divergent execution over the
  course of simulating thousands of environments and simulation frames.

  An illustrative example of this issue is observed with the runtime domain randomization of object's
  physics materials. This process can introduce both determinacy and simulation issues when executed
  on the GPU due to the way these parameters are passed from the CPU to the GPU in the lower-level APIs.
  Consequently, it is strongly advised to perform this operation only at setup time, before the
  environment stepping commences.


.. _PhysX Determinism documentation: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/API.html#determinism
.. _#940: https://github.com/isaac-sim/IsaacLab/pull/940
