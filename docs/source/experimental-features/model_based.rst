Model-based motion tracking
===========================

``isaaclab_rl.model_based`` trains a dynamics model from physical transitions and
updates an actor through differentiable imagined rollouts. The implementation is
ported from `HybridRobotics/SuperMimic
<https://github.com/HybridRobotics/supermimic/tree/5efbf114e11f725c31d83e217be987c13c0e4ce2>`_
at revision ``5efbf114e11f725c31d83e217be987c13c0e4ce2``. This revision, rather than a
moving branch or an inferred paper configuration, defines the algorithm baseline.

The ``isaaclab_rl.model_based`` package owns neural networks, replay, supervised
world-model learning, BPTT, critic learning, state integration, differentiable
kinematics, imagined rollouts, and reset sampling. The
``isaaclab_tasks.core.motion_tracking`` task supplies reference goals, rewards,
physical observations, and randomization through native managers. Its G1
configuration reuses the controller from the existing motion-tracking task.
Train and play use the standard Isaac Lab entrypoints.

Source attribution and licenses
-------------------------------

This backend is an adaptation of existing open-source work, not an independently
invented learning algorithm. Its source boundaries are:

* **Algorithm and reference task mathematics:**
  ``HybridRobotics/SuperMimic`` at the revision linked above, under Apache-2.0.
  Ported networks, replay, optimizers, learners, runner, and task mathematics retain
  their upstream attribution and modification notices. The license is retained in
  ``docs/licenses/dependencies/supermimic-official-license.txt``.
* **Initial native integration:**
  `Isaac Lab PR #8826 by ooctipus
  <https://github.com/isaac-sim/IsaacLab/pull/8826>`_, at commit
  ``b2de4e7c4702a1f5417c22e00ec71b9e4ba887cb``. Inherited SuperMimic portions retain
  the ETH Zurich / Chong Zhang attribution and MIT license in
  ``docs/licenses/dependencies/supermimic-license.txt``. Isaac Lab integration
  code uses the repository's BSD-3-Clause license.
* **Quaternion arithmetic:** WXYZ adapters reuse Isaac Lab math. Matrix conversion
  and inversion retain the norm reduction order of the BSD-3-Clause helpers in
  `mjlab at revision 8ee51fbc
  <https://github.com/mujocolab/mjlab/blob/8ee51fbcf806a7419189f706d9e394cbeb7790fa/src/mjlab/utils/lab_api/math.py>`_.
* **Normalization utility:** networks reuse ``rsl_rl.modules.EmpiricalNormalization``;
  see ``docs/licenses/dependencies/rsl_rl-license.txt``. The model-based learners
  and runner are separate from the RSL-RL training backend.

Native task interfaces, CLI wiring, distributed coordination, and tests adapt
these components to Isaac Lab. Source-file notices identify the applicable
license; the repository's top-level license does not replace third-party notices.

Datasets and robot assets are external inputs with their own terms. Users must
download them from their providers. See
:doc:`/source/how-to/train_model_based` for download sources, local preparation,
and runnable training commands.

Reference behavior
------------------

The port retains these reference computations:

* GenCoord predicts root and joint velocity increments and integrates poses
  analytically. MaxCoord predicts body velocity increments and recovers joints
  with the reference analytic IK.
* Networks use Linear/LayerNorm/GELU blocks. Each goal encoder produces a
  tanh-bounded latent, consumed with observation history by the shared decoder.
* Goal windows contain ten frames at offsets 0, 5, ..., 45, clamped within each
  clip. The actor excludes the goal's relative root-position channels.
* Replay stores pre-step snapshots. An eight-snapshot model window scores seven
  transitions, using the reference nominal-error weights and channel averaging.
* Muon optimizes intermediate Linear weights; fused Adam optimizes the remaining
  parameters. The port retains learning rates, gradient clipping, CUDA graph
  updates, and normalizer update ordering.
* BPTT differentiates through dynamics, analytical rewards, soft termination,
  survival probabilities, and the terminal critic. Critic training retains GAE,
  the upper value/target bound, and five passes over four minibatches.
* Return-based reset sampling retains its 20-step window, bounded-memory value
  estimates, and explore/focus/keep mixture.

Defaults match the pinned runner: 1,000 replay frames per environment, model
layers ``(2048, 2048, 2048)``, actor layers ``(2048, 1024, 512, 256)``, critic layers
``(256, 128)``, latent width 32, model window 8, model batch size 2048, ten model
batches, actor rollout 32, and actor batch size 4096. The default actor observes
the represented state with history length one. Proprioception history and joint
motion/velocity encoder training remain configurable.

An iteration is **one vectorized physical environment step**. With 4096
environments and the default replay ratio of 16, there are approximately 0.4
learning cycles per iteration. A 40,000-iteration run collects 163,840,000 physical
transitions and performs 15,997 learning cycles after warmup. Iterations from the
earlier experimental backend are not interchangeable with this counter.

Simulator and data boundary
---------------------------

The reference learning equations are distinct from reproducing an entire paper
experiment. The official repository depends on ``mjlab_research``; that dependency
and its exact physical configuration are not included in the public source used
for this port. Isaac Lab uses its native Newton/MuJoCo Warp articulation, manager
lifecycle, and USD asset. Physical asset conversion, contact/actuator settings,
randomization, reset behavior, and dataset preprocessing therefore require a
separate comparison before claiming paper-level reproduction.

The G1 task uses a 20 ms control period with four 5 ms physics steps. Physical
and imagined rewards call the same tracking formula. Native XYZW root/body
quaternions are converted to the reference WXYZ layout at the interface. WXYZ
adapters reuse Isaac Lab quaternion utilities; matrix conversion and inversion
retain the reference norm reduction order for optimizer parity. G1 maximal
coordinates use the reference IK's two/three-hinge-chain assumptions; this is not
a general inverse-kinematics solver for arbitrary robots.

For prepared references, each NPZ clip supplies float arrays ``joint_pos``,
``joint_vel``, ``root_pos_w``, ``root_quat_w`` (WXYZ), ``root_lin_vel_w``, and
``root_ang_vel_w`` in the articulation's joint order at 50 Hz. An optional ``fps``
field must equal 50. Clip boundaries are retained when files are concatenated;
provided velocities are used unchanged. CSV/NPY pose files are also accepted in
the legacy XYZ/XYZW/joint-radian format with ``env.commands.motion.source_fps``.
That path resamples and estimates velocities locally; it does not establish
equivalence to the official processed dataset.

The reference loader is separate from the BeyondMimic named-body NPZ loader:
its files store root and joint state and supplied velocities, while body state
is reconstructed by differentiable kinematics. Treating these formats as
interchangeable would change the learning targets.

Training and playback
---------------------

Use a matching 29-joint G1 USD/MJCF pair and reference directory:

.. code-block:: bash

   uv run --frozen isaaclab train \
     --rl_library model_based \
     --task Isaac-Motion-Tracking-GenCoord-G1 \
     --num_envs 4096 --max_iterations 40000 \
     --motion_dir /path/to/motions \
     --asset_path /path/to/g1_29dof.usda \
     --kinematics_path /path/to/g1_29dof.xml

Choose ``Isaac-Motion-Tracking-MaxCoord-G1`` for maximal coordinates.
For a proprioception-history policy, add
``agent.actor_obs=proprio agent.history_length=10``. The reference default uses
motion goals exclusively. Set
``agent.slot_mix.motion=0.5 agent.slot_mix.velocity=0.5`` to train the two input
modalities together; both slots still optimize motion-tracking reward.

``metrics.jsonl`` records physical iterations, collected transitions, learning
cycles, losses, and rewards. Optional W&B logging is enabled with
``agent.wandb_project=PROJECT``. Checkpoints include the pinned reference revision,
task/agent configurations, model weights, and optimizer states. They do not
include the physical simulator state or replay; playback loading does not imply
an exact training continuation.

.. code-block:: bash

   uv run --frozen isaaclab play --rl_library model_based \
     --checkpoint /path/to/model_040000.pt \
     --num_envs 1 --steps 3000 --visualizer newton_gl --real-time

Playback restores the saved configuration. Use ``--motion_dir``, ``--asset_path``,
and ``--kinematics_path`` to relocate inputs; use ``--video`` for the standard
native video recorder.

Validation
----------

Differential tests load the pinned upstream tensor modules independently and
compare state integration, losses, action gradients, actor routing, GAE, replay,
reset sampling, reward gradients, and a complete GPU actor/critic optimizer step.
The local mjlab checkout supplies its original quaternion primitives for this
comparison, without loading its simulator. Native physical tests independently
check kinematics against simulated body buffers, physical/imagined reward parity,
reference-frame advancement, and reset history.

.. code-block:: bash

   export ISAACLAB_SUPERMIMIC_REFERENCE=/path/to/supermimic
   export ISAACLAB_MJLAB_REFERENCE=/path/to/mjlab
   export ISAACLAB_G1_TRACKING_DATA=/path/to/g1_tracking
   uv run --extra test python -m pytest source/isaaclab_rl/test/test_model_based_reference.py
   uv run --extra test python -m pytest source/isaaclab_tasks/test/test_model_based_tracking.py

Checkpoint compatibility
------------------------

Playback accepts checkpoints from the earlier standalone ``model_based`` task
layout. It relocates serialized MDP functions, preserves the saved actuator
gains and action scales, and translates the main branch's renamed actuator-limit
fields. The previous ``IsaacContrib-Motion-Tracking-GenCoord-G1`` and MaxCoord IDs
remain aliases. This does not import the separate experimental ``supermimic``
backend or add simulator/replay-state resume to weight playback.

Published-paper preset
----------------------

The public source defaults above are not the same as Table A.5 of
arXiv:2610.28480v2. Select ``--agent paper_model_based_cfg_entry_point`` on the
GenCoord task for Adam throughout, ten proprioception frames, TF32, and cosine
learning-rate decay to 0.03 of the initial rate over 250,000 physical vector
steps. The run lasts 320,000 such steps and saves every 4,000. Both physical
steps and learning updates are logged; the paper's iteration terminology does
not independently establish that these counters match its original runs.

This preset uses nine replay snapshots to score the eight predicted transitions
in Eq. (1); ``replay_ratio=18`` retains 0.4 learning cycles per physical step.
The source-default preset continues to use eight snapshots and seven targets.
Use ``--agent paper_no_critic_cfg_entry_point`` for the same settings without
terminal-value bootstrapping, and ``agent.policy.horizon=8`` (or 16, 32, 64, 128)
for the published horizon sweep. To apply the published physical settings on the
native task, also select ``env.events.context.mode=reset`` and
``env.scene.robot.spawn.articulation_props.self_collision_enabled=true``.
These overrides do not certify equivalence of the underlying collision assets.

``Train/normalized_episode_return`` divides each completed episode's accumulated
reward by the maximum reward over its intended remaining reference/time-limit
budget. Early failures retain their unvisited tail in that denominator. This is
a training diagnostic under adaptive resets, not the first-frame, full-clip
checkpoint evaluation in Appendix F. Playback disables proprioceptive noise and
sets foot friction to one.

Multi-GPU training
------------------

``train_multigpu --rl_library model_based`` averages gradients for the world
model, actor, and critic before clipping and stepping. Empirical normalization
uses statistics from all ranks. CUDA graph updates include the NCCL collectives.
Rank zero saves the shared model; each rank records its configuration and metrics.

Environment and learner batch sizes are **per GPU**, as for other Isaac Lab
distributed backends. For two GPUs with a combined 4096 environments, 2048 world
model samples, and 4096 policy samples, use:

.. code-block:: bash

   uv run isaaclab train_multigpu --rl_library model_based --num_gpus 2 -- \
     --task Isaac-Motion-Tracking-GenCoord-G1 \
     --agent paper_model_based_cfg_entry_point --num_envs 2048 \
     --motion_dir /path/to/npz_motions \
     agent.wm.batch_size=1024 agent.policy.batch_size=2048

Distributed tracking requires NPZ reference clips and assigns sorted files to
ranks in round-robin order. Each rank retains complete clips and its own replay
buffer and adaptive reset sampler. Together the shards cover the input corpus;
the samplers do not maintain one global reset distribution. Playback restores
the complete corpus. This implementation preserves the learning equations but
does not certify the unpublished distributed setup used by the paper.

The learning rate is not automatically scaled. Keeping global batch sizes fixed
also retains the single-GPU learning rate and learning-update ratio. Model
weights are replicated; reference data, simulated environments, and learning
batches can be divided to reduce memory on each GPU.
