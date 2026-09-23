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

.. autoclass:: ManagerTermBaseCfg
    :members:
    :exclude-members: __init__

Observation Manager
-------------------

Observation delay
~~~~~~~~~~~~~~~~~

Configure a fixed or per-episode random observation delay directly on the term:

.. code-block:: python

    from isaaclab.envs import mdp
    from isaaclab.managers import ObservationTermCfg

    joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, delay_min_lag=1, delay_max_lag=3)

Each environment samples its lag uniformly from the inclusive bounds at initialization and reset,
and keeps it until its next reset. Equal bounds give a constant delay; both zero disable delay.
This matches the reset-based lag policy of :class:`~isaaclab.actuators.DelayedPDActuator`.
Both use :class:`~isaaclab.utils.buffers.DelayBuffer`; observation lag counts recorded observation
samples, while actuator lag counts physics steps.

The processing order is observation function, modifiers, noise, clipping, scaling, delay, then history.
The ``delay_min_lag`` and ``delay_max_lag`` field names and delay placement follow
`mjlab's observation configuration <https://mujocolab.github.io/mjlab/main/source/observations.html>`_.
The lag sampling policy here follows Isaac Lab's delayed actuator, rather than mjlab's per-step lag updates.

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
