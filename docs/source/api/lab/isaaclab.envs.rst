isaaclab.envs
=============

.. automodule:: isaaclab.envs

  .. rubric:: Submodules

  .. autosummary::

    mdp
    ui

  .. rubric:: Classes

  .. autosummary::

    ManagerBasedEnv
    ManagerBasedEnvCfg
    ManagerBasedRLEnv
    ManagerBasedRLEnvCfg
    DirectRLEnv
    DirectRLEnvCfg
    DirectMARLEnv
    DirectMARLEnvCfg
    ManagerBasedRLMimicEnv
    MimicEnvCfg
    SubTaskConfig
    SubTaskConstraintConfig
    ViewerCfg

Manager Based Environment
-------------------------

.. autoclass:: ManagerBasedEnv
    :members:

.. autoclass:: ManagerBasedEnvCfg
    :members:
    :exclude-members: __init__, class_type

Manager Based RL Environment
----------------------------

Finite episode evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~

``ManagerBasedRLEnv`` separates episode completion from starting a replacement.
Subclasses can implement a finite evaluation scheduler through the following hooks:

* ``active_episode_mask`` returns the scheduler's current assignments as a boolean tensor
  of shape ``(num_envs,)``. The base implementation keeps every parallel environment active.
* ``_finish_episodes(env_ids)`` collects terminal records before reset events run. Override
  this hook to write application results, call the base implementation, and retire assignments.
  Restrict explicit reset calls to assigned episodes so initial reset does not produce results.
* ``_select_episode_start_env_ids(candidate_env_ids)`` assigns new episodes and returns the
  subset to reset. Initial ``reset()`` and automatic replacements use the same hook, so an
  evaluation can request fewer episodes than there are parallel environments.
* ``_validate_reset_request(reset_kind)`` can reject external ``reset``, ``reset_to``, and
  visualizer ``manual`` resets. ``reset_to`` also selects environments for new episodes and requires
  every requested environment to start one. It retains the supplied order when writing caller-provided
  states; a finite evaluator should reject state restoration before changing any assignments.

The scheduler owns the assignments; the environment does not maintain a second active mask.
Each step snapshots the mask before stepping. Completed assignments emit termination or
truncation once and are recorded even when no replacement starts. Recorder pre-step,
post-step, and physics-substep callbacks continue to return full batches; the recorder
manager selects the active rows. The existing recorder ``record_pre_reset`` callbacks
collect final fields and export each finished episode, including episodes without a replacement.
Post-reset callbacks run only for environments starting new episodes.

Environments without assigned episodes continue simulating. Physics, episode clocks,
progress terms, observations, and interval events continue to compute for the full scene.
Inactive rows return zero rewards and false completion flags, and have no trajectory records
or reset events. Their observations and other manager state are not evaluation results.
Video consumers should use the scheduler's assignments to exclude inactive rows. Policy callers
can use ``active_episode_mask`` to skip unused inference; this is optional for exact episode limits.
Completed-step recording must use the assignments before stepping; the next policy call
sees the assignments after replacement episodes have started.

.. autoclass:: ManagerBasedRLEnv
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: ManagerBasedRLEnvCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, class_type

Direct RL Environment
---------------------

.. autoclass:: DirectRLEnv
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: DirectRLEnvCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, class_type

Direct Multi-Agent RL Environment
---------------------------------

.. autoclass:: DirectMARLEnv
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: DirectMARLEnvCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, class_type

Mimic Environment
-----------------

.. autoclass:: ManagerBasedRLMimicEnv
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: MimicEnvCfg
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, class_type

.. autoclass:: SubTaskConfig
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, class_type

.. autoclass:: SubTaskConstraintConfig
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: __init__, class_type

Common
------

.. autoclass:: ViewerCfg
    :members:
    :exclude-members: __init__

Additional Public Classes
-------------------------

The following classes are part of the public :mod:`isaaclab.envs` API.

.. currentmodule:: isaaclab.envs

.. autosummary::
   :nosignatures:

   DataGenConfig
   SubTaskConstraintCoordinationScheme
   SubTaskConstraintType
   VideoRecorderCfg

.. autoclass:: DataGenConfig
   :show-inheritance:

.. autoclass:: SubTaskConstraintCoordinationScheme
   :show-inheritance:

.. autoclass:: SubTaskConstraintType
   :show-inheritance:

.. autoclass:: VideoRecorderCfg
   :show-inheritance:

.. toctree::
   :hidden:

   isaaclab.envs.leapp_deployment_env
