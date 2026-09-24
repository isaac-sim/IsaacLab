isaaclab.managers
=================

.. automodule:: isaaclab.managers

  .. rubric:: Classes

  .. autosummary::

    SceneEntityCfg
    ManagerBase
    ManagerTermBase
    ManagerTermBaseCfg
    ObservationManager
    ObservationGroupCfg
    ObservationTermCfg
    ActionManager
    ActionTerm
    ActionTermCfg
    EventManager
    EventTermCfg
    CommandManager
    CommandTerm
    CommandTermCfg
    RewardManager
    RewardTermCfg
    TerminationManager
    TerminationTermCfg
    CurriculumManager
    CurriculumTermCfg
    RecorderManager
    RecorderTermCfg

Scene Entity
------------

.. autoclass:: SceneEntityCfg
    :members:
    :exclude-members: __init__

Manager Base
------------

.. autoclass:: ManagerBase
    :members:

.. autoclass:: ManagerTermBase
    :members:

Observation term classes can override ``prepare_scene(cfg, env)`` to request scene inputs after
spawning and prestartup events, before simulation reset. Returning an instance transfers it to the
observation manager; returning ``None`` uses normal construction after startup. A hook that fails
before returning must release its own incomplete allocations. Implement ``close()`` for owned
resources and make repeated calls safe. See :ref:`camera-post-processing` for image processing that
uses these hooks.

.. autoclass:: ManagerTermBaseCfg
    :members:
    :exclude-members: __init__

Observation Manager
-------------------

Observation delay
~~~~~~~~~~~~~~~~~

Configure observation delay directly on the term:

.. code-block:: python

    from isaaclab.envs import mdp
    from isaaclab.managers import ObservationTermCfg

    joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, delay_min_lag=1, delay_max_lag=3)

Each environment samples its lag uniformly from the inclusive bounds at initialization and reset.
By default, ``delay_hold_prob=1.0`` keeps that lag for the episode, matching the reset-based lag policy of
:class:`~isaaclab.actuators.DelayedPDActuator`. Equal bounds give a constant delay; both zero disable delay.
Both use :class:`~isaaclab.utils.buffers.DelayBuffer`; observation lag counts recorded observation
samples, while actuator lag counts physics steps.

Set ``delay_hold_prob`` below 1.0 to vary latency within an episode:

.. code-block:: python

    joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, delay_min_lag=1, delay_max_lag=3, delay_hold_prob=0.8)

On each recorded sample, each environment retains its lag with probability 0.8 and otherwise draws a new
one. Zero redraws on every sample. Holding the lag keeps the latency constant while frames continue to
advance; it does not freeze the returned frame. Reset always redraws the selected environments' lags.

The processing order is observation function, modifiers, noise, clipping, scaling, delay, then history.
The ``delay_min_lag`` and ``delay_max_lag`` field names and delay placement follow
`mjlab's observation configuration <https://mujocolab.github.io/mjlab/main/source/observations.html>`_.
``delay_hold_prob`` follows mjlab's lag-retention semantics, with a default of 1.0 to preserve Isaac Lab's
reset-based delay policy.

``compute(update_history=True)`` records a sample in both delay and observation history buffers.
``compute()`` and ``compute_group()`` read without advancing either buffer or resampling lag.
For a delay buffer with no recorded sample after initialization or reset, the current input is returned
without recording it. Once recording starts, delays exceeding the available history return the oldest
sample. Partial resets invalidate only the selected environments' histories.

.. autoclass:: ObservationManager
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: ObservationGroupCfg
    :members:
    :exclude-members: __init__

.. autoclass:: ObservationTermCfg
    :members:
    :exclude-members: __init__

Action Manager
--------------

.. autoclass:: ActionManager
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: ActionTerm
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: ActionTermCfg
    :members:
    :exclude-members: __init__

Event Manager
-------------

.. autoclass:: EventManager
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: EventTermCfg
    :members:
    :exclude-members: __init__


Command Manager
---------------

.. autoclass:: CommandManager
    :members:

.. autoclass:: CommandTerm
    :members:
    :exclude-members: __init__, class_type

.. autoclass:: CommandTermCfg
    :members:
    :exclude-members: __init__, class_type


Reward Manager
--------------

.. autoclass:: RewardManager
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: RewardTermCfg
    :exclude-members: __init__

Termination Manager
-------------------

.. autoclass:: TerminationManager
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: TerminationTermCfg
    :members:
    :exclude-members: __init__

Curriculum Manager
------------------

.. autoclass:: CurriculumManager
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: CurriculumTermCfg
    :members:
    :exclude-members: __init__

Recorder Manager
----------------

.. autoclass:: RecorderManager
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: RecorderTermCfg
    :members:
    :exclude-members: __init__

Additional Public Classes
-------------------------

The following classes are part of the public :mod:`isaaclab.managers` API.

.. currentmodule:: isaaclab.managers

.. autosummary::
   :nosignatures:

   DatasetExportMode
   RecorderManagerBaseCfg
   RecorderTerm

.. autoclass:: DatasetExportMode
   :show-inheritance:

.. autoclass:: RecorderManagerBaseCfg
   :show-inheritance:

.. autoclass:: RecorderTerm
   :show-inheritance:
