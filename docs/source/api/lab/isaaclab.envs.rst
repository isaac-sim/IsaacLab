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

To implement a different stepping algorithm, replace the function body with your own
action processing, simulation, rewards, resets, and observation computation. The function
owns counters, command and event updates, rendering, and recorder calls as well. In
particular, it must capture terminal data before resetting terminated environments and
return the observation, reward, terminated, truncated, and extras tuple expected by RL wrappers.

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
