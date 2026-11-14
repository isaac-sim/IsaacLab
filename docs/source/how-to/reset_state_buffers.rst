:orphan:

Building validated reset-state buffers
======================================

A reset-state buffer collects valid starting conditions once and reuses them
during training. Expensive inverse kinematics, scripted grasps, and settling can
then run during collection rather than on every episode reset.

:mod:`isaaclab.envs.reset` separates collection from sampling and physical
restoration. The buffer stores a nested dictionary of tensors with one leading
row per start. The task decides which physical state and metadata those tensors
represent. The same utilities work with manager-based and Direct environments.

Collect starts
--------------

Use :class:`~isaaclab.envs.reset.ResetStateBufferCfg` to configure the buffer size,
collection budget, and seed. Supply ordinary callbacks for generation, capture,
and named acceptance checks:

.. code-block:: python

   from isaaclab.envs.reset import ResetStateBuffer, ResetStateBufferCfg

   buffer = ResetStateBuffer(ResetStateBufferCfg(size=4096, max_batches=100, seed=42))
   buffer.collect(
       env_ids,
       generate={"table": generate_table_starts, "grasped": generate_grasped_starts},
       capture=capture_reset_state,
       checks={"joint_limits": check_joint_limits, "stable": check_stability},
   )

Each generator receives the selected environment indices, places the candidates,
and can return a Boolean validity mask. Named checks also return one Boolean per
candidate. Generation must synchronize the scene before checks read derived body
poses or contacts. Capture returns tensors for the accepted environments in the
order supplied to the callback.

For a purely procedural tensor catalog, the candidate indices can identify recipe
rows instead of simulation environments. Generate and validate the recipe tensors
in the callbacks without writing or stepping the scene. Use this only for recipe
construction; validation of physical contacts and restored grasps still needs
simulation and replay checks.

A single generator is sufficient for ordinary randomized starts. Named generators
collect balanced groups, which prevents an easily generated reset category from
filling the entire buffer. When the collection budget is exhausted, construction
fails with the number of accepted starts and rejected checks instead of waiting
indefinitely.

Run collection during controlled initialization or in an offline script. A
generator that advances physics also advances other worlds in the simulation.
It must not run while unrelated training episodes are in progress. Reset the
environment normally after collection before starting the policy.

Validate the intended start
---------------------------

Acceptance rules belong to the task. Useful checks include finite state, joint
limits, workspace bounds, successful IK, allowed contacts, and a brief grasp or
settling check. A valid grasp may intentionally contain contact; rejecting all
contact is not an appropriate general default.

Validate restored starts as well as generated starts. Restoring poses can rebuild
contacts and controller targets differently from the original rollout. A short
replay test should check that the restored start remains usable and that it does
not already satisfy the task's success condition.

The buffer does not certify solver compatibility. In particular, constrained
Kamino mechanisms must use the backend's normal constraint-consistent reset and
forward-kinematics path. Task callbacks can check the resulting state without
putting solver internals into the buffer implementation.

Sample and restore
------------------

:class:`~isaaclab_tasks.utils.ResetStateSampler` tracks the starting row assigned
to each parallel environment and learns a rolling success rate for each row.
Its default favors starts near 50 percent success and retains uniform exploration.
This targets useful practice rather than enforcing a fixed overall success rate.

.. code-block:: python

   from isaaclab_tasks.utils import ResetStateSampler, ResetStateSamplerCfg

   sampler = ResetStateSampler(
       ResetStateSamplerCfg(), num_envs, buffer.num_states, device
   )
   rows = sampler.reset(
       env_ids,
       success=episode_success[env_ids],
       valid=completed_episode[env_ids],
   )
   state = buffer.get_state(rows)
   # Restore the task's physical state and associated metadata here.

Record outcomes before clearing the previous episode's success flags. Omit
outcomes for the initial draw and mask manual or zero-step resets. Only resetting
environments receive new row assignments.

The sampler reuses the shared success monitor and sampling strategies from
:mod:`isaaclab_tasks.utils.sampling`. Configure row selection through
``ResetStateSamplerCfg.sampling``; the monitor config controls outcome history.
Non-default monitor sampling controls are rejected. For a uniform baseline,
configure the existing uniform strategy:

.. code-block:: python

   from isaaclab_tasks.utils.sampling import SamplerCfg, UniformSamplingStrategyCfg

   uniform_cfg = ResetStateSamplerCfg(
       sampling=SamplerCfg(strategies=[UniformSamplingStrategyCfg()], eps=0.0)
   )

For physical scene states, reuse
:meth:`~isaaclab.scene.InteractiveScene.get_state` and
:meth:`~isaaclab.scene.InteractiveScene.reset_to`, with ``is_relative=True`` on
both sides. Capture callbacks must select the requested rows from the scene
state. Task-specific tensor recipes can instead use the public asset writers.
Keep positions relative to environment origins so starts transfer between worlds.

Scene state is not a complete environment checkpoint. Restore commands, action
targets, attachment state, and other task metadata through the appropriate task
reset hooks. Account for the existing manager reset order: values restored by an
event can be overwritten by subsequent command or action manager resets. Solver
caches should be cleared through the backend reset lifecycle.

Using existing task reset data
----------------------------------------

The buffer accepts task-specific tensor trees rather than prescribing a new asset
serializer. Existing reset workflows can therefore adopt collection independently
of their scene geometry and state representation:

* Conveyor recipes can store robot poses, parcel layouts, and transfer metadata.
* Stacking can generate starts along a scripted approach, grasp, and placement.
* NIST can retain its collision checks and explicitly capture assembly variant IDs.
* Franka Pour can retain dataset compatibility checks and particle-layout IDs;
  large particle layouts can remain separate from the per-row recipe tensors.

These integrations do not change existing public task registrations or stored
dataset formats. Selecting rows and validating an artifact's compatibility are
different responsibilities from constructing physically valid starts.

Evaluate complete task starts
----------------------------------------

Use the cube pickup reference to compare sparse-reward training with ordinary and
buffered starts. Generate grasp starts by physically approaching and closing the
fingers. Starts captured after grasping must remain outside the goal region.
The example succeeds as soon as a grasped cube reaches the commanded
point ``(0.5, 0.0, 0.15)`` [m] in the robot root frame, within a radius of
``0.04`` [m]. The reward is a terminal ``+1`` on that transition.

The registered ``IsaacContrib-Cube-Pickup-Franka`` task collects table, approach,
pregrasp, and physical grasp starts. It uses relative Cartesian actions and a
binary pickup-to-point reward. Generate and inspect a small buffer first:

.. code-block:: bash

   uv run python scripts/environments/cube_pickup.py --mode generate \
       --num_envs 64 --bank_size 256 --output outputs/pickup/reset_states.json

Train with the existing RSL-RL runner. ``--num_envs`` is the number of environments
per GPU in distributed training. Training runs without a viewer when no visualizer
is requested:

.. code-block:: bash

   uv run isaaclab train_multigpu --num_gpus 4 --rl_library rsl_rl \
       --task IsaacContrib-Cube-Pickup-Franka --num_envs 1024

The task saves its generated reset states alongside the training logs. Evaluate a
saved checkpoint from ordinary table starts using a separate seed:

.. code-block:: bash

   uv run python scripts/environments/cube_pickup.py --mode evaluate \
       --checkpoint /path/to/model.pt --num_envs 64 --episodes 512 \
       --seed 10000 --output outputs/pickup/evaluation.json

Evaluate the learned policy separately from ordinary table starts using held-out
positions and seeds. Report that result separately from success under the adaptive
training mixture. Success on already-grasped starts alone does not establish that
the policy learned to pick up the cube from the table.
