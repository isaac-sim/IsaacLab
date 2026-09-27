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

Resolved selections
~~~~~~~~~~~~~~~~~~~

``SceneEntityCfg.resolve(scene)`` creates cached ``torch.long`` selectors on the
scene device, available as ``joint_ids_torch``, ``body_ids_torch``,
``fixed_tendon_ids_torch``, and ``object_collection_ids_torch``. The existing
``*_ids`` fields keep their Python lists or slices for host consumers.
Managers resolve their copies of term configurations before calling the terms.

.. code-block:: python

    cfg = SceneEntityCfg("robot", body_names=["left_foot", "right_foot"])
    cfg.resolve(env.scene)
    positions = env.scene["robot"].data.body_pos_w.torch[:, cfg.body_ids_torch]
    first_body = cfg.body_ids[0]  # Python int; no device readback
    body_names = [env.scene["robot"].body_names[i] for i in cfg.body_ids]

Migration is optional: existing ``*_ids`` accesses retain their behavior. Use the
``*_ids_torch`` accessors for Torch indexing to avoid rebuilding device indices.
Full selections (``slice(None)``) remain slices, preserving ordinary tensor view
semantics. Partial slices become explicit device indices so backend write APIs use
the same selection as Torch reads. Negative indices are normalized to non-negative
device indices for operations such as ``index_select``. Host fields retain their
original lists and slices.
Before resolution, these accessors return the configured host selectors for direct
term calls. Caching begins only after resolution.

Treat resolved host selections and cached tensors as read-only during stepping.
After editing a selection, clear its other specification (names or indices) if
necessary and call ``resolve`` before using it again:

.. code-block:: python

    cfg.body_names = None
    cfg.body_ids = [1, 3]
    cfg.resolve(env.scene)

There is no per-step mutation tracking or cache validation. Failed resolution clears
old caches; successful re-resolution rebuilds them on the new scene device.
Replacing or copying a configuration with ``replace`` or ``copy`` clears the caches
and requires resolution to populate them.
``copy.deepcopy`` retains independent copies of the resolved tensors, including
their device. Configuration serialization excludes the caches, including selectors
nested in manager configurations. Runtime dataclass fields use
``metadata={"serialize": False}`` to opt out of ``class_to_dict`` serialization.

Use host ``*_ids`` for scalar selection (``ids[0]``), Python name lookup, and metadata.
Use ``*_ids_torch`` for vector tensor indexing and device writer APIs. Existing custom
terms can adopt these accessors independently. Class terms that also accept unresolved
configurations can call ``convert_to_torch`` once at construction; already-resolved
tensors are reused.

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

Observation outputs are independent snapshots: later simulation steps, modifier updates, and resets
do not overwrite previously returned tensors. The manager automatically copies borrowed storage before
mutation or retention. Term and custom callback outputs are treated conservatively, without decorators
or task-level copy settings. Clipping and scaling create independent storage when needed, which later
processing can reuse. A term that already returns an independent tensor may still be copied when the
pipeline cannot establish ownership itself.

Observation callables may declare an optional keyword-only ``out`` parameter to write directly
into a manager-provided destination, for example ``def observation(env, ..., *, out=None)``.
When ``out`` is None, the callable returns an observation normally. When supplied, it must fill and
return that exact tensor without retaining it, resizing it, or replacing its storage.

The manager checks the signature once during initialization. Only an explicitly declared keyword-only
``out=None`` enables destination writes; accepting arbitrary ``**kwargs`` does not. The manager probes
the ordinary call to infer shape, dtype and device, which must remain constant, then allocates a fresh
contiguous destination for each computation. ``out`` is reserved and must not appear in term ``params``.
Custom modifiers, noise and history retain their usual snapshot protections. Built-in
:class:`~isaaclab.envs.mdp.observations.image_rgb` terms use this interface automatically to avoid a
second full-image copy for normalized uint8 images. Existing observation configurations require no changes.

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
