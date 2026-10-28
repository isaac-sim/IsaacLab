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
:meth:`SceneEntityCfg.finalize`, which replaces each list with a ``torch.long`` tensor on the simulation
device. Terms share this tensor, so do not modify it in place. Slices stay slices. Finalization is one-way.

.. code-block:: python

    def feet_height(env, asset_cfg: SceneEntityCfg):
        asset = env.scene[asset_cfg.name]
        # a slice or the cached device tensor; no upload or synchronization
        return asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2]

Term calls should not read selections on the host. For a single selected body, gather and select
on the device, for example ``data[:, cfg.body_ids][:, 0]``. Outside stepping, ``ids.tolist()``
returns a host copy.

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

.. _recorder-capture:

Recorder Manager
----------------

Tasks configure recorder terms to capture the state, controls and other fields their consumers need.
Terms run at pre/post step and reset callbacks; reset callbacks receive slices or indices directly.
Consumers own subsequent processing, replay and learning.

Returning a field path and a batched tensor (or nested dictionary of tensors) retains samples.
For online use without retention, publish the batch on the term and return ``(None, None)``.
Consumers access it through ``env.recorder_manager.get_term(name)`` and copy data before the next capture.
``EXPORT_NONE`` disables file export only; disabling the term removes capture itself.

Fields may have different sample rates. ``buffer_initial_capacity`` sets the initial samples per world
and field; choose it with payload size and episode length in mind. Storage grows without dropping
samples and is reused across resets.

**Observation migration:** RL post-step callbacks run after rewards and terminations, before reset
and observation updates. ``env.obs_buf`` still contains the preceding policy input. Read physical
state from the scene or call ``env.observation_manager.compute_group("recording", update_history=False)``
for current observations. Use a separate noiseless group without history or delay to avoid stale
measurements and additional policy noise draws. History is not advanced, but noise and modifiers still
run. Non-RL environments update their normal observations before post-step capture.

**Storage migration:** ``get_episode(env_id)`` returns an owned tensor snapshot with a leading sample
dimension; reading copies data and may synchronize the device. Modify live recording through
``add_to_episodes`` and ``set_success_to_episodes``, not the snapshot. ``EpisodeData.add()`` and
``pre_export()`` are removed: collect through the manager, or assign completed tensors to
``episode.data``. HDF5 loading and playback keep the same tensor format.

The unused ``RecorderTerm.close(file_path)`` hook is removed. Dataset handlers finalize files;
applications explicitly close any additional resources they own.

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
