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

``SceneEntityCfg.resolve(scene)`` replaces each non-slice joint, body, fixed-tendon,
and object-collection selection with a one-dimensional ``torch.long`` tensor on the
scene device. Slices remain slices. Names and integer lists are still accepted when
configuring a selection. Managers resolve their copies before calling terms.

.. code-block:: python

    cfg = SceneEntityCfg("robot", body_names=["left_foot", "right_foot"])
    cfg.resolve(env.scene)
    positions = env.scene["robot"].data.body_pos_w.torch[:, cfg.body_ids]
    first_position = env.scene["robot"].data.body_pos_w.torch[:, cfg.body_ids[:1]].squeeze(1)

Migration: resolved ``*_ids`` fields are tensors rather than Python lists. Tensor
consumers use them directly. Replace scalar-index expressions such as
``data[:, cfg.body_ids[0]]`` with the one-element selection above: converting a CUDA
scalar index to a Python integer synchronizes the device. ``len(ids)`` and ``ids.shape``
only inspect metadata and do not read values back.

For consumers that require Python integers, use ``ids.tolist()`` once during setup or
inspection. This transfers CUDA indices to the host and synchronizes. Avoid it, Python
iteration over CUDA indices, and ``ids[0].item()`` in stepping code. There is no retained
host copy, wrapper, second selector field, or process-wide cache.

Use ``ids.to(device=device, dtype=torch.long)`` to adapt an existing tensor without
unnecessarily copying it with ``torch.tensor(ids)``. Before resolution, callers may still
receive configured lists or slices; use ``isaaclab.utils.convert_to_torch`` when supporting
unresolved configurations and tensor input together.

Treat resolved indices as read-only. Replace the selection and call ``resolve`` after
configuration changes. Re-resolution reads tensors back for host name validation, then
creates the final device selectors. Configuration copies own copies of their tensors;
serialization emits ordinary lists using a dataclass field ``serializer`` callback.
Serialization of resolved CUDA configurations therefore also performs a host readback.
Keep both resolution and serialization outside the step loop.

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
