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
    Delay
    DelayCfg
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

Delay configuration
-------------------

:class:`~isaaclab.managers.DelayCfg` wraps an observation callable with latency and sample holds. It uses the same
:class:`~isaaclab.utils.buffers.DelayBuffer` primitive as the legacy :class:`~isaaclab.actuators.DelayedPDActuator`.
Native Newton actuators currently use Newton's own delay implementation. Each stream owns its
history and advances an index without shifting stored frames. The caller owns lag sampling and delivery cadence.

For example, add latency to a proprioceptive observation:

.. code-block:: python

    from isaaclab.envs import mdp
    from isaaclab.managers import DelayCfg, ObservationTermCfg

    joint_observation = ObservationTermCfg(
        func=DelayCfg(term=mdp.joint_pos_rel, params={}, min_lag=1, max_lag=3)
    )

Parameters of the wrapped callable belong in ``DelayCfg.params``; leave the outer term's ``params`` empty.
Delay runs before observation modifiers, noise, clipping, and scaling. The manager constructs the callable
and resets both its state and its delay history through the normal term lifecycle.

Lags and periods count wrapped term calls. Observation delay advances whenever the term is computed,
including extra observation queries. Actuator-target latency counts physics steps. Sharing buffer storage and
indexing does not couple the two clocks or their sampling policies.

At call :math:`t`, a lag :math:`\ell_t` is sampled uniformly from the inclusive configured bounds. The
candidate sample has timestamp :math:`s_t = \max(0, t - \ell_t)`, relative to that environment's last reset.
A refresh delivers this candidate only if its timestamp is at least as recent as the last delivered sample.
Otherwise, the previous output is held. Thus variable latency never delivers samples out of order.

``update_period`` controls refresh opportunities; ``per_env_phase`` randomizes their phase at reset.
``hold_prob`` independently skips a refresh for each environment. For example,
``DelayCfg(term=mdp.joint_pos_rel, update_period=3, per_env_phase=False)`` delivers inputs at calls 0, 3, 6, and so on,
holding each delivered frame between those calls. ``hold_prob=1`` holds the first frame until reset.
Held frames may become older than ``max_lag``: that bound limits sampled latency, not the duration of a hold.
``per_env=False`` shares the sampled lag across environments; their holds and reset schedules remain independent.

The first input after reset is delivered immediately. During warmup, unavailable history resolves to that first
input. Partial resets invalidate only the selected environments, so history from a previous episode cannot leak
into a new one. Invalid lag updates through ``DelayBuffer.set_time_lag()`` raise before changing stored lags.

.. autoclass:: Delay
   :members:
   :special-members: __call__

.. autoclass:: DelayCfg
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
