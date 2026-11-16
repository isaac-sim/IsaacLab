:orphan:

Building validated reset-state buffers
======================================

Reset-state storage, preparation, selection, and physical restoration are separate
operations. :class:`~isaaclab.envs.reset.ResetStateBuffer` stores a fixed tensor
tree. Tasks can prepare those tensors through simulation, load compatible data,
or select procedural recipes without using a tensor buffer at all.

The optional helpers in :mod:`isaaclab_tasks.utils` collect candidate states and
track outcomes for sampled reset IDs. The task owns validation rules, catalog
metadata, and restoration through its normal reset lifecycle. These utilities
work with manager-based and Direct environments.

Store prepared states
---------------------

Construct a buffer from a non-empty nested dictionary of tensors. Every leaf has
one leading row per state and shares the same row count and device:

.. code-block:: python

   from isaaclab.envs.reset import ResetStateBuffer

   buffer = ResetStateBuffer(prepared_states)
   selected = buffer.get_state(row_ids)
   all_states = buffer.states

Construction detaches and copies the tensors. Both accessors return copies;
``get_state`` copies only the requested rows, while ``states`` copies the whole
tree. Buffer IDs are dense row indices from zero to ``buffer.num_states - 1``.
The buffer has no collection configuration, random generator, group metadata,
acceptance checks, or file-loading policy.

Existing task artifacts can supply the tensors directly. For example, the cube
pickup artifact retains its ``state`` payload key:

.. code-block:: python

   import torch

   payload = torch.load("reset_states.pt", map_location=device, weights_only=True)
   buffer = ResetStateBuffer(payload["state"])

The task must first establish compatibility with its assets, state fields,
backend, and reset procedure. Restoring associated metadata remains its
responsibility. This example does not add an automatic artifact loader to the
cube pickup training workflow.

Optionally collect physical starts
----------------------------------

Use :func:`~isaaclab_tasks.utils.collect_reset_states` when preparing starts needs
bounded simulation and acceptance checks:

.. code-block:: python

   from isaaclab_tasks.utils import ResetStateCollectionCfg, collect_reset_states

   generators = {"table": generate_table_starts, "grasped": generate_grasped_starts}
   states, group_ids, report = collect_reset_states(
       ResetStateCollectionCfg(size=4096, max_batches=100, seed=42),
       env_ids,
       generate=generators,
       capture=capture_reset_state,
       checks={"joint_limits": check_joint_limits, "stable": check_stability},
   )
   buffer = ResetStateBuffer(states)
   group_names = tuple(generators)

Each generator places candidates for the supplied environment indices and may
return a Boolean validity mask. Named checks return additional masks. Generation
must synchronize the scene before checks read derived poses or contacts. Capture
returns tensors for accepted environments in the supplied order.

A single generator is sufficient. Named generators receive balanced quotas;
returned group IDs follow the mapping's iteration order. The caller owns group
names, IDs, and diagnostic reports. An exhausted batch budget raises an error
with collection counts. The helper restores the caller's Torch, NumPy, and Python
random states on success or failure.

Collect during controlled initialization or offline: a generator that advances
physics also advances other worlds in the simulation. Reset normally after
collection before starting the policy. Validate finite state, limits, workspace,
allowed contacts, and grasp stability as appropriate. Also restore and briefly
simulate snapshots to check reconstructed contacts and exclude initial success.
The buffer does not certify backend compatibility; constrained mechanisms must
use their backend's constraint-consistent reset and forward-kinematics path.

Select IDs and record outcomes
------------------------------

:class:`~isaaclab_tasks.utils.ResetStateSampler` tracks each environment's assigned
ID and recent outcomes. Its default favors IDs near 50 percent success while
retaining coverage of all entries. This is a per-entry practice target, not a
fixed overall success rate.

.. code-block:: python

   from isaaclab_tasks.utils import ResetStateSampler, ResetStateSamplerCfg

   sampler = ResetStateSampler(ResetStateSamplerCfg(), num_envs, buffer.num_states, device)
   rows = sampler.reset(
       env_ids,
       success=episode_success[env_ids],
       valid=completed_episode[env_ids],
   )
   selected = buffer.get_state(rows)

Record outcomes before clearing the previous episode's success flags. Omit
outcomes on the initial draw and mask manual or zero-step resets. Pass
``state_ids=requested_ids`` to ``sampler.reset`` for an explicit choice: it records
the previous outcome before assigning the requested IDs, so the next outcome is
credited correctly. ``sampler.get_state_ids(env_ids)`` returns current assignments.

For procedural or cached providers, constructor argument ``state_ids`` can supply
unique, noncontiguous integer IDs such as recipe seeds. History and metric tensors
follow that catalog's order. The task resolves selected IDs to recipes or cached
data; these IDs do not change the tensor buffer's dense indexing contract.

Configure selection independently
---------------------------------

The generic :class:`~isaaclab_tasks.utils.sampling.Sampler` accepts
``(cfg, num_items, device)`` and computes ``probabilities(metrics)`` from named
per-item tensors of shape ``[num_items]`` on the sampler's device.
:class:`~isaaclab_tasks.utils.sampling.TargetSuccessScoreCfg`
selects a metric (``success_rate`` by default), target, concentration, and weight.
Its analytic peak is one, so changing concentration narrows the score without
changing its peak contribution. :class:`~isaaclab_tasks.utils.sampling.UniformScoreCfg`
needs no metrics. Weighted scores and ``eps`` are added before normalization;
a uniform weight of ``0.05`` does not mean a fixed five-percent mixture.

For uniform reset selection:

.. code-block:: python

   from isaaclab_tasks.utils.sampling import SamplerCfg, UniformScoreCfg

   uniform_cfg = ResetStateSamplerCfg(
       sampling=SamplerCfg(strategies=[UniformScoreCfg()], eps=0.0)
   )

Custom scorers extend :class:`~isaaclab_tasks.utils.sampling.SamplingScoreCfg`,
accept their configuration at construction, and implement ``score(metrics, out)``
without changing the metrics. Supply additional metric tensors through
``ResetStateSampler(..., metrics=metrics)`` and update those borrowed tensors in
place. ``success_rate`` is reserved for the sampler's outcome monitor. Legacy
Beta/uniform strategy names and success-rate-bound sampler calls remain supported
for existing NIST configurations.

Restore through the task
------------------------

For physical snapshots, use
:meth:`~isaaclab.scene.InteractiveScene.get_state` and
:meth:`~isaaclab.scene.InteractiveScene.reset_to` with ``is_relative=True`` on
both sides. Capture only requested rows and keep positions relative to environment
origins. Procedural recipes can instead use public asset writers.

Scene state is not a complete environment checkpoint. Restore commands, controller
targets, attachments, and task metadata through the appropriate reset hooks;
later manager resets can overwrite earlier restoration. Clear solver caches
through the backend reset lifecycle. Existing conveyor, stacking, NIST, and Franka
Pour workflows can retain their task-specific serializers and compatibility checks.

The :doc:`sparse cube-pickup walkthrough <sparse_cube_pickup>` applies these pieces
to physical approach and grasp starts, training comparisons, explicit replay, and
evaluation from ordinary table starts with held-out seeds.
