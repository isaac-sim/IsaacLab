.. _supermimic:

SuperMimic motion tracking
==========================

The experimental ``IsaacContrib-Motion-Tracking-G1`` task integrates the
`SuperMimic workflow <https://github.com/zita-ch/supermimic_vibed>`__ into Isaac Lab.
Use the standard reinforcement-learning entrypoints with ``--rl_library supermimic``.
The physical teacher runs Newton/MuJoCo Warp with CUDA graphs enabled. A learned world model
fits its recorded transitions, and the policy improves through differentiable imagined rollouts.

This integration supports the MLP world model and the optional one-frame NeRD architecture.
It includes stationary padding of reference goals beyond recorded clip endpoints: padded poses
hold their position and orientation, with zero velocity. Recorded endpoint velocities remain unchanged.
The integration does not establish the paper's reported zero-failure result or an exact reproduction
of its motion selection. Evaluation reports failures over the actual prepared corpus.

Ownership
---------

The task is an ordinary :class:`~isaaclab.envs.ManagerBasedRLEnv`. Its configuration lives in
``isaaclab_tasks.contrib.g1_tracking.g1_tracking_env_cfg`` and owns the scene, actuators, observations,
commands, rewards, terminations and recorder terms. The matching 29-DoF G1 asset variant and its
actuator settings are task-specific; substituting another G1 asset changes the dynamics.

The remaining task modules have separate responsibilities:

* ``mdp`` implements the native manager terms, including transition capture and adaptive reference sampling.
* ``tracking`` defines the state layout and the differentiable observation, reward and termination
  expressions shared by real and imagined transitions.
* ``motion`` loads and resamples the reference corpus and constructs future goals.
* ``kinematics`` supplies the matching differentiable link geometry and joint order.

``isaaclab_rl.supermimic`` owns the learner configuration, networks, normalizers, replay ring and
optimization. The native ``train_supermimic`` and ``play_supermimic`` entrypoint backends construct
the environment once and own stepping, checkpoint I/O, logging and evaluation. No external
SuperMimic package or launcher is required. The task does not import the learner, W&B or PhysicsNeMo.

Transition capture and storage
------------------------------

``env.recorders.transition`` is disabled in the task's default configuration. SuperMimic training
enables ``TransitionRecorderCfg`` and reads its ``latest`` tensor dictionary after each control step.
This term captures the state, action, prior action, randomized context, reference index, reward,
episode step and termination flags. It captures the true successor before an automatic reset.
Its preallocated tensors are overwritten on the next step; a consumer that retains transitions
must copy them into its own storage. The learner does this in its time/world replay ring.

The term uses the normal recorder lifecycle. It does not replace environment stepping or own replay.
By default, ``record_to_episodes=False`` avoids retaining per-world episodes. Offline consumers can
enable that setting and configure ``env.recorders.dataset_export_mode`` and the export directory
to use the same capture with the manager's batched episode storage and HDF5 exporter.
Other tasks can define their own :class:`~isaaclab.managers.RecorderTerm` schema at task-config level;
the core manager has no dependency on G1 or a learning algorithm.

Prepare the robot and motions
-----------------------------

Use a CUDA-capable Isaac Lab installation with the Newton backend and its standalone MJCF importer.
Start with the 40-motion retargeted LAFAN1 corpus. The preparation command downloads the robot
description and motions at a pinned dataset revision and records file hashes. The conversion
command is Isaac Lab's existing MJCF-to-USD tool:

.. code-block:: bash

   uv run python scripts/tools/prepare_g1_tracking.py --output /data/g1_tracking
   uv run python scripts/tools/convert_mjcf.py \
       /data/g1_tracking/g1_description/g1_29dof.xml \
       /data/g1_tracking/g1_usd/g1_29dof.usd

The converter writes the composed asset to
``/data/g1_tracking/g1_usd/g1_29dof/g1_29dof.usda``. To reuse existing local inputs, supply
``--source_dir /path/to/prepared_inputs`` to the first command and choose an empty output directory.
That directory must contain ``g1_description/g1_29dof_rev_1_0.urdf``, its meshes and ``lafan1_g1``.
Local reuse records the local source and hashes, rather than claiming a downloaded revision.
The task defaults to the same layout under ``<IsaacLab>/data/g1_tracking``; the examples below
override those paths for an external data volume.

For BONES-SEED, obtain the ``g1.tar.gz`` archive from the ``isaac/bones-seed-g1:1`` dataset and run:

.. code-block:: bash

   uv run python scripts/tools/prepare_bones.py \
       --archive /data/bones-seed-g1/g1.tar.gz \
       --mjcf /data/g1_tracking/g1_description/g1_29dof.xml \
       --output /data/bones_prepared --workers 16

The converter reads named CSV columns, converts centimetres/degrees to metres/radians and resamples
120 Hz poses to 50 Hz. It writes retained poses under ``motions/``, per-clip decisions in
``motions.jsonl`` and filter settings and hashes in ``manifest.json``. The geometric and name filters
are explicit independent choices; the output is not claimed to match the authors' 129,514-clip subset.
Robot and motion data remain external inputs subject to their source licenses. BONES motion data
is attributed to Bones Studio.

Train and resume
----------------

Train the MLP baseline through the unified entrypoint:

.. code-block:: bash

   uv run python scripts/reinforcement_learning/train.py \
       --rl_library supermimic --task IsaacContrib-Motion-Tracking-G1 \
       --num_envs 4096 --log_dir logs/supermimic/g1_mlp \
       env.scene.robot.spawn.usd_path=/data/g1_tracking/g1_usd/g1_29dof/g1_29dof.usda \
       env.commands.motion.kinematics_path=/data/g1_tracking/g1_description/g1_29dof.xml \
       env.commands.motion.motion_dir=/data/g1_tracking/lafan1_g1

The default agent uses BF16 neural projections, FP32 geometry and compiled tensor rollouts.
Task settings use ``env.*`` overrides; optimizer, replay and runner settings use ``agent.*``.
The analytical imagined objective requires the task's observation and termination functions.
Observation history, stateless policy noise and task reward parameters remain configurable.
The runner rejects incompatible observation functions or layouts instead of silently optimizing
a different imagined task.

For a short integration smoke test, use ``--num_envs 32 --max_iterations 2`` and append:

.. code-block:: bash

   agent.buffer_steps=16 agent.warmup_steps=4 agent.world_model_window=2 \
   agent.world_model_batch_size=32 agent.world_model_batches=1 agent.policy_batch_size=32 \
   agent.horizon=4 agent.critic_epochs=1 agent.critic_minibatches=1 \
   agent.world_model_only_iterations=0 agent.log_interval=1 \
   env.observations.policy.history_length=3 env.commands.motion.goal_frames=2

These reduced settings test the pipeline; they are not a learning benchmark. For the prepared
BONES corpus, replace the motion directory with ``/data/bones_prepared/motions`` and set
``env.commands.motion.source_fps=50``. For a nine-hour budget, add
``--max_iterations 250000 agent.max_hours=9``. The budget is cumulative when resuming and is
checked after each update. Initial environment setup and replay warmup are outside that budget.

Optional NeRD world model
~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``agent.world_model=nerd-one-frame`` to select PhysicsNeMo Robotics' ``ParallelNeRD`` with
four attention spans of one. The policy, critic, task, replay and optimization objective are shared
with the MLP configuration. NeRD is used inside the learner; the physical teacher still uses
MuJoCo Warp. This option does not deploy a learned collision or constraint solver into Newton.

This experimental option requires a compatible PhysicsNeMo Robotics installation exposing
``physicsnemo_robotics.models.parallel_nerd.ParallelNeRD``. It is imported only when selected;
the default MLP workflow has no PhysicsNeMo dependency. The integration was validated with
PhysicsNeMo Robotics ``0.1.0.dev0`` at revision ``a81160ef`` and PhysicsNeMo ``2.1.1`` in the
development environment. A clean, released package combination is not established here;
the experimental dependency must be supplied separately.

Logging and preemption
~~~~~~~~~~~~~~~~~~~~~~

Every run writes resolved configurations, a run manifest and ``train_log.csv``. To enable W&B,
authenticate its SDK and append ``agent.wandb_project=<project> agent.wandb_entity=<entity>``.
An optional stable ``agent.wandb_run_id=<id>`` lets resumed runs continue the same history.
The runner records the actual run identity in ``wandb-run.json`` and the checkpoint.

``last.pt`` contains the world model, policy, critic, their optimizers, normalizers, adaptive sampler
statistics and semantic configurations. It is replaced atomically after the first update, at the
configured interval, after 120 seconds by default, and on completion. When W&B is enabled, each
save is uploaded as a version of ``run-<id>-checkpoint`` with the ``latest`` alias; the runner waits
for upload before reporting ``durable_checkpoint_iteration``. Milestones also save
``policy_<iteration>.pt`` for playback. SIGINT/SIGTERM request a save after the current update.
An immediate kill can lose progress since the last completed checkpoint.

To resume, repeat the original task and learning overrides, add ``--checkpoint /path/to/last.pt``
and increase ``--max_iterations``. Logging, save intervals and budget settings can change.
Physical state and replay are rebuilt; model and optimizer state and sampler statistics resume.
This is a training continuation, not a bitwise continuation of the simulator trajectory.
The runner accepts its native checkpoint metadata; legacy external-run checkpoints require
explicit conversion before use.

Evaluate and record video
-------------------------

Playback restores the native checkpoint's task settings and evaluates every clip exactly once:

.. code-block:: bash

   uv run python scripts/reinforcement_learning/play.py \
       --rl_library supermimic --checkpoint logs/supermimic/g1_mlp/policy_014000.pt \
       --num_envs 64 --output logs/supermimic/g1_mlp/evaluation.json

The output includes the failure count, failed names, per-clip returns and lengths, and both
clip-weighted and frame-weighted normalized returns. Failed tails remain in the full-clip
denominator. Use these physical evaluation metrics to assess policy quality; dynamics loss
measures prediction fit and can improve without improving tracking success.

Use ``--motion_dir`` to select another corpus in the same pose schema and source frame rate.
``--asset_path`` and ``--kinematics_path`` relocate the matching assets. Add the standard
``--video --video_length 600`` options to record playback with Isaac Lab's native renderer.
For an interpretable video, select a small motion directory and a small ``--num_envs`` count.

Validation scope
----------------

The native tests cover terminal capture, observation/kinematics agreement, episode export,
stationary goal padding, replay boundaries, complete evaluation batches and gradients through
compiled MLP/NeRD dynamics. Asset-dependent tests use ``ISAACLAB_G1_TRACKING_DATA`` pointing to
the prepared directory above. The dependency-direction tests prevent reintroducing an external
training forwarder or moving learner ownership into the task.

The source adaptation preserves the upstream SuperMimic MIT notice in
``docs/licenses/dependencies/supermimic-license.txt``.
