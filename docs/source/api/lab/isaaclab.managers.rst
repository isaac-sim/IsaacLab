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

Scene-entity selections follow a build-then-finalize lifecycle, similar to Newton's model builder.
Managers call :meth:`SceneEntityCfg.resolve` while preparing terms, which fills ``joint_ids``,
``body_ids``, ``fixed_tendon_ids``, and ``object_collection_ids`` with host lists. Class-based terms
can read these lists in ``__init__``. Once every term is constructed, managers call
:meth:`SceneEntityCfg.finalize`, which replaces each list with a read-only
:class:`~isaaclab.utils.warp.ProxyArray` of ``int64`` indices on the simulation device. Slices stay
slices.

.. code-block:: python

    from isaaclab.utils import torch_index

    def feet_height(env, asset_cfg: SceneEntityCfg):
        asset = env.scene[asset_cfg.name]
        # a slice or the cached device tensor; no upload or synchronization
        return asset.data.body_pos_w.torch[:, torch_index(asset_cfg.body_ids), 2]

Term calls should not read selections on the host. For a single selected body, gather and select
on the device, for example ``data[:, torch_index(cfg.body_ids)][:, 0]``. Pass
``torch_index(cfg.joint_ids)`` to asset write methods. Outside stepping, ``ids.warp.numpy()``
returns a host copy. Serialization writes host lists, and :meth:`SceneEntityCfg.resolve` returns a
finalized configuration to host lists.

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
