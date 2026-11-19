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
procedural recipe seeds. ``sampler.indices`` gives dense storage positions for
``buffer.get_state`` and ``sampler.state_ids`` gives the corresponding labels.
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
   :alt: Initialization prepares validated states once. Runtime selects IDs, gathers snapshots, and restores masked worlds. Completed episode outcomes feed back into sampling.

   Generation and validation run during initialization. Runtime uses fixed storage and masks.

Capture the runtime
-------------------

Warm the configured path before capture. The sampler and ``get_state(..., out=...)``
can run in a CUDA graph. Lift's ``newton_mjwarp`` preset supports capturing both
operations and the asset writes together. With OvPhysX or Kit-based PhysX, run Lift's
asset writes outside capture.

Update the contents of the bound arrays in place between graph replays. The choice to
provide outcomes or explicit IDs is fixed at capture time.

.. code-block:: python

   def reset_selected():
       sampler.reset(mask, success=episode_success, valid=completed_episode)
       buffer.get_state(sampler.indices, out=restored, mask=mask)
       robot.write_joint_position_to_sim_mask(position=restored["joint_pos"], env_mask=mask)

   reset_selected()
   with wp.ScopedCapture(device=device) as capture:
       reset_selected()
   wp.capture_launch(capture.graph)

Run prefill, rejection loops, loading, inspection, and logging outside capture. Use
``get_state(ids)`` to allocate a snapshot for inspection. The example captures only
the reset operations.

Set ``ResetStateSamplerCfg.seed`` to initialize the sampler's per-environment random streams.
Use ``sampler.reseed(seed)`` between rollouts to restart those streams while retaining
history and assignments. A captured graph observes the new streams.

Configure useful practice
--------------------------

The default adds uniform weight to a Beta kernel score that peaks at 50 percent recent
success. This favors starts of intermediate difficulty while giving every entry a
chance to be drawn. Concentration narrows the score's peak while keeping its height
fixed at one.
When all success rates are zero, the default draws uniformly. Include valid starts
the policy can solve early so that the scores can begin to distinguish entries.

.. code-block:: python

   from isaaclab.envs.reset import BetaSamplingScoreCfg

   adaptive = ResetStateSamplerCfg(
       history_length=50,
       scores=[BetaSamplingScoreCfg(target=0.5, kappa=2.0, weight=0.25)],
       uniform_weight=0.0501,
   )
   uniform = ResetStateSamplerCfg(scores=[])

.. figure:: ../_static/how-to/reset_states/success_sampling.jpg
   :width: 100%
   :alt: An illustrative five-entry bank gives its largest sampling probability to the entry with 50 percent success, retaining coverage at both extremes.

   Example probabilities for five states with different success rates. The sampler
   adds ``0.0501`` to each score, then normalizes the weights within each asset group.

.. image:: ../_static/how-to/reset_states/sgs-method-light.gif
   :class: only-light
   :width: 100%
   :alt: An animation shows the policy training loop and the Beta score peaking at 50 percent success.

.. image:: ../_static/how-to/reset_states/sgs-method-dark.gif
   :class: only-dark
   :width: 100%
   :alt: An animation shows the policy training loop and the Beta score peaking at 50 percent success.

The animation illustrates how rollouts and training change which starts are most useful.
As the policy revisits starts, their estimated success rates shift along the Beta
score curve. The curve shows the score before adding uniform weight.
Animation courtesy of Octi Zhang.

Custom configurations derive from :class:`~isaaclab.envs.reset.SamplingScoreCfg`.
Their scorer implements ``score(metrics, out)`` with Warp operations on preallocated
storage. Borrow additional per-entry metric arrays through ``metrics=`` at sampler
construction and update them in place. The sampler supplies ``success_rate`` and
``episode_counts`` so a score can account for both difficulty and how much evidence
it has. Tasks can supply other signals, such as learning progress or staleness.

The sampler uses a fixed catalog and weighted draws with replacement. Tasks that
generate new entries during training or require scheduled coverage, such as visiting
each row once per cycle, own that scheduling policy.
The assignment must match the state actually played. When restoring a snapshot from
outside the catalog, exclude the resulting episode from catalog outcome tracking.

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
replays explicit starts. With adaptive sampling enabled, supply
``completed=natural_episode_mask`` to record outcomes for completed natural episodes
that took at least one step. The event credits the existing success reward's sticky
flag before the reward manager clears it and restores root-link poses, COM velocities,
joints, and controller targets through public mask writers.

Migrating existing reset configurations
---------------------------------------

Lift's ``success_monitor`` event parameter is replaced by ``reset_sampler``.
NIST's ``sampling`` parameter takes ``ResetStateSamplerCfg`` with core score
configurations. Its separate ``success_monitor_cfg`` still configures outcome
history. Set ``metric="monitor_success_rate"`` on a score to use those rates.
