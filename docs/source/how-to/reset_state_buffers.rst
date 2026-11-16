:orphan:

Native reset-state buffers
==========================

A reset bank separates prepared starting states from the policy that selects them.
The shared API lives in :mod:`isaaclab.envs.reset`: :class:`~isaaclab.envs.reset.ResetStateBuffer`
owns Warp snapshots, and :class:`~isaaclab.envs.reset.ResetStateSampler` records outcomes
and selects entries. Tasks own generation, physical validity, and restoration through
public asset writers. The existing core Lift task demonstrates this as a reset event.

Prepare storage
---------------

Construct a buffer from a nested dictionary of Warp arrays. Every leaf has the same
leading state count and device. :class:`~isaaclab.utils.warp.ProxyArray` and same-device
Torch inputs are accepted at the boundary; storage and runtime operations use Warp.
Construction copies the inputs.

.. code-block:: python

   import warp as wp
   from isaaclab.envs.reset import ResetStateBuffer

   empty = ResetStateBuffer({})
   assert empty.num_states == 0

   # A single, task-validated default row can populate the whole bank.
   defaults = {"joint_pos": robot.data.default_joint_pos.warp[:1]}
   buffer = ResetStateBuffer(defaults, num_states=1024)

An empty dictionary has no schema and contains zero states. A tree of zero-row arrays
also represents an empty buffer, retaining its schema. Nonempty reads and sampling
require prepared entries. An empty schema cannot allocate populated states without
being given their fields.

The ``states`` property returns copied ``ProxyArray`` snapshots for inspection or
persistence. Compatible saved tensors can also initialize a bank:

.. code-block:: python

   import torch

   saved_states = torch.load("reset_states.pt", map_location=device, weights_only=True)
   buffer = ResetStateBuffer(saved_states)

The task must establish asset, backend, field, unit, and controller compatibility.
Store positions relative to environment origins when sharing starts across worlds.
Storage does not perform collision checks or manage a disk cache.

Select and restore with masks
------------------------------

The native interface uses full-environment arrays and a Boolean mask. Allocate output
storage before capture, then reuse it:

.. code-block:: python

   from isaaclab.envs.reset import ResetStateSampler, ResetStateSamplerCfg

   sampler = ResetStateSampler(ResetStateSamplerCfg(seed=42), num_envs, buffer.num_states, device)
   mask = wp.ones(num_envs, dtype=wp.bool, device=device)
   restored = {"joint_pos": wp.empty((num_envs, robot.num_joints), dtype=float, device=device)}

   # Record the episodes that ended before assigning their replacement starts.
   sampler.reset(mask, success=episode_success, valid=completed_episode)
   buffer.get_state(sampler.indices, out=restored, mask=mask)
   robot.write_joint_position_to_sim_mask(position=restored["joint_pos"], env_mask=mask)

Only masked environments receive new assignments, random draws, or restored values.
Omit outcomes on the initial draw, and exclude manual or zero-step resets with
``valid``. Record the previous episode before clearing its success flag. Simultaneous
outcomes for the same entry are recorded in environment-index order; each entry
retains its latest ``history_length`` outcomes.

Pass a full-environment ``state_ids`` array to ``sampler.reset`` for explicit starts.
Optional constructor ``state_ids`` gives entries stable integer labels, such as
procedural recipe seeds; ``indices`` always contains their dense storage positions.
Catalog labels reserve ``-1`` for an unassigned environment. Assignments, rates, counts, and probabilities
are borrowed ``ProxyArray`` outputs: treat them as read-only and use explicit requests
to change assignments. Invalid requested IDs are reported in ``invalid_requests``
on the device, so inspection can happen outside capture.

For catalogs partitioned by asset combination, use ``group_ids`` and ``num_groups``.
Partitions contain equal contiguous ranges; each environment draws within its group.
The buffer itself has no group metadata. Legacy manager terms can pass ``env_ids=``
to the sampler; that shim populates a preallocated mask.

.. figure:: ../_static/how-to/reset_states/reset_lifecycle.jpg
   :width: 100%
   :alt: Initialization prepares validated states once; captured runtime records previous outcomes, scores entries, selects IDs, gathers snapshots, and restores masked worlds.

   Generation and validation run during initialization. Runtime uses fixed storage and masks.

Capture the runtime
-------------------

Warm the configured path before capturing it. Capture reset selection, gathering, and
native asset writes together; masks, outcomes, and requested IDs can then change in
place between replays:

.. code-block:: python

   def reset_selected():
       sampler.reset(mask, success=episode_success, valid=completed_episode)
       buffer.get_state(sampler.indices, out=restored, mask=mask)
       robot.write_joint_position_to_sim_mask(position=restored["joint_pos"], env_mask=mask)

   reset_selected()
   with wp.ScopedCapture(device=device) as capture:
       reset_selected()
   wp.capture_launch(capture.graph)

Prefill, rejection loops, loading, inspection, and logging remain initialization or
host work. The allocating ``get_state(ids)`` convenience accessor is intended for
inspection; use ``out=`` for the captured path. The manager-based environment around
an event does not become a captured environment merely because its reset event is
capturable. MDP2 native launch hooks can call the same runtime directly; its Torch backend uses
zero-copy input shims. Fused Warp terms can reuse ``target_success_score`` and
``sample_cdf``. No MDP2 dependency is required by the reset API.

``sampler.reseed(seed)`` updates existing random-stream storage without clearing history
or assignments. A captured graph observes the new streams. Explicitly seeded manager-based
resets notify event terms before execution, so ``env.reset(seed=...)`` also reseeds the
Lift and NIST samplers, including terms nested inside reset combinators.

Configure useful practice
--------------------------

The default combines an analytic unit-peak score around 50 percent recent success
with uniform coverage and an additive floor. This favors intermediate starts;
it does not constrain overall training success to 50 percent. Concentration changes
the score's width without shrinking its peak relative to other terms.

.. code-block:: python

   from isaaclab.envs.reset import TargetSuccessScoreCfg, UniformScoreCfg

   adaptive = ResetStateSamplerCfg(
       history_length=50,
       scores=[TargetSuccessScoreCfg(target=0.5, kappa=2.0, weight=0.25), UniformScoreCfg(weight=0.05)],
       eps=0.0001,
   )
   uniform = ResetStateSamplerCfg(scores=[UniformScoreCfg()], eps=0.0)

.. figure:: ../_static/how-to/reset_states/success_sampling.jpg
   :width: 100%
   :alt: An illustrative five-entry bank gives its largest sampling probability to the entry with 50 percent success, retaining coverage at both extremes.

   An illustrative probability snapshot, not a training curve. A uniform weight of
   ``0.05`` is an additive score, not a fixed five-percent mixture.

Custom configurations derive from :class:`~isaaclab.envs.reset.SamplingScoreCfg`.
Their scorer implements ``score(metrics, out)`` with Warp operations on preallocated
storage. Borrow additional per-entry metric arrays through ``metrics=`` at sampler
construction and update them in place. This supports visit counts, staleness, or loss
summaries without changing selection bookkeeping; ``success_rate`` is reserved.

Use the existing Lift reset event
---------------------------------

``Isaac-Lift-Franka`` already uses ``core/lift/mdp/events.py::conditional_reset``.
Its ordinary reset terms generate candidate starts, its criteria reject invalid
states, and its optional diversity descriptor spreads the retained starts. Prefill
runs once per asset-combination group before replay begins. The event now uses the
shared buffer and sampler; no additional task or environment subclass is required.

Configure its existing ``events.conditional_reset.params`` entries:

* ``buffer_size_per_group`` controls retained starts per asset combination.
* ``valid_criteria`` and ``max_prefill_iters`` control acceptance and the attempt budget.
* ``diversity_feature`` optionally spreads starts over a task-defined descriptor.
* ``reset_sampler`` configures history, scores, and independent random streams;
  ``None`` selects uniformly without recording outcomes.

After ordinary initialization, the event's ``reset_mask(mask, state_ids=ids)`` method
is its native manual-replay entry. Supply ``completed=natural_episode_mask`` to
record natural outcomes inside a captured runner. It restores root-link poses, COM velocities, joints, and
controller targets through public mask writers. It credits the existing success
reward's sticky flag before the reward manager clears it and excludes unfinished
manual resets. The existing Lift observations, rewards, registrations, and agent
configuration remain the use case.

Migrating existing reset configurations
---------------------------------------

Lift's ``success_monitor`` event parameter is replaced by ``reset_sampler``.
NIST's ``sampling`` parameter now takes ``ResetStateSamplerCfg`` directly; set
``history_length`` there instead of passing a separate ``success_monitor_cfg``.
The previous NIST sampler and strategy imports are removed: use core score
configurations and the mask-native sampler. No compatibility runtime is retained.
