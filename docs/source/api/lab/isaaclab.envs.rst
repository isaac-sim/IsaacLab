isaaclab.envs
=============

.. automodule:: isaaclab.envs

  .. rubric:: Submodules

  .. autosummary::

    mdp
    reset
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

Custom step functions
~~~~~~~~~~~~~~~~~~~~~

Set :attr:`ManagerBasedRLEnvCfg.step_fn` before constructing the environment to replace
its complete step implementation with a function. The function receives the environment
as its first argument and the action batch as its second argument. It is bound once
during initialization, so callers continue to use ``env.step(action)``.

For example, this function delegates to the standard implementation and scales its rewards:

.. code-block:: python

    from isaaclab.envs import ManagerBasedRLEnv

    def custom_step(env, action):
        observations, rewards, terminated, truncated, extras = ManagerBasedRLEnv.step(env, action)
        return observations, 2.0 * rewards, terminated, truncated, extras

    env_cfg.step_fn = custom_step
    env = ManagerBasedRLEnv(cfg=env_cfg)

The default ``step`` keeps action processing, physics stepping, reward and termination computation,
reset logic, command/event updates, and observation computation visible in one function. Custom
functions can write out the same sequence and insert their own work between any of these operations.

For per-physics-step capture, write the decimation loop in your script and set backend
decimation to one before the first environment step. The :ref:`recorder-capture` example shows
where to place capture relative to physics, episode export, and reset.

Dataset recording is explicit. Environments do not construct a recorder manager or dispatch
capture callbacks during step, reset, reset_to, or close. Recording workflows own their episode
storage, initial-state capture after reset, success classification, export, and file closure.
See :ref:`recorder-capture` for the dataset API and migration.

Calling ``env.step(action)`` inside the custom function recurses; use the explicit class
method call shown above when delegating to the standard implementation. Leaving
``step_fn=None`` uses the environment class's step method, including subclass overrides.

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
   :members:
   :show-inheritance:
   :exclude-members: __init__

.. toctree::
   :hidden:

   isaaclab.envs.leapp_deployment_env
