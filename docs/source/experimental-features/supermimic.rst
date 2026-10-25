.. _supermimic:

Learned dynamics with NeRD and SuperMimic
=========================================

Isaac Lab supports two experimental learned-dynamics workflows in its native task and learning
packages. Cartpole and ANYmal C collect offline NeRD datasets and train policies in a learned
Newton solver. G1 motion tracking uses the SuperMimic online model-based learning workflow.
Both use :class:`~isaaclab.managers.RecorderManager` to capture transitions before autoreset,
with storage selected independently by the consumer. PhysicsNeMo Robotics owns the NeRD network;
Isaac Lab owns task configuration, capture, optimization and the Newton integration.

SuperMimic motion tracking
--------------------------

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

The offline NeRD workflow below uses that same network implementation with temporal attention.
Its generalized-coordinate state and fixed offline normalization differ from SuperMimic's
task-specific state and online statistics. Those representations are intentionally distinct.

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

Offline NeRD: Cartpole and ANYmal C
-----------------------------------

The registered tasks live in ``isaaclab_tasks.core.cartpole.cartpole_nerd_env_cfg`` and
``isaaclab_tasks.core.velocity.config.anymal_c.nerd_env_cfg``. Each has a numerical teacher,
a task-independent ``-Explore`` configuration, and a ``-Learned`` configuration for policy
training/playback. They use ordinary action, observation, reward and termination managers.
The exploration tasks configure ``NeRDRecorderCfg``; teacher and learned playback leave capture
disabled by default. Algorithm settings are registered under ``nerd_cfg_entry_point`` and are
selected from the recording's task metadata, without launching a simulation during offline fitting.

The shared Newton representation and solver live in ``isaaclab_newton.physics``. The recorder
term lives in ``isaaclab_newton.envs.mdp.recorders``. ``isaaclab_rl.nerd`` owns offline window
sampling and optimization. The two scripts under ``scripts/learning/nerd`` are composition roots
for collection and fitting; policy learning uses the existing reinforcement-learning scripts.
There is no additional integration package or external task launcher.

Cartpole records applied efforts at 120 Hz. ANYmal C records executed joint-position targets
over a 20 ms control period, including the teacher's four 5 ms ANYdrive updates. Its state
features use the root heading frame, body-frame gravity, and four fixed foot-sphere probes.
The learned solver predicts that entire period, then uses Newton forward kinematics to update
body state. The numerical collision and constraint solve is replaced; fixed probe feature
queries and forward kinematics remain. Both numerical and learned physics enable CUDA graphs.
The learned manager retains a private Torch allocation pool for its captured temporaries.

``Isaac-Velocity-Flat-Anymal-C-NeRD-Effort-Explore`` records every 5 ms using the applied effort
after ANYdrive and its effort limits. The matching ``-Effort-Learned`` task keeps ANYdrive
outside NeRD and advances learned physics four times per 20 ms policy action. Use these task
names with the commands below for a physics-only model. The existing target-conditioned
tasks retain their 20 ms actuator-plus-physics contract. Their checkpoints are not interchangeable.

Effort capture requires ``decimation=1`` and ``num_substeps=1``. Explicit position controllers
are supported because their final applied effort is available in ``joint_f``. Implicit position
drives are rejected: their solver-generated force and integration derivative are not represented
by that input. Fixed implicit damping with a zero velocity target remains part of the physical
model. Capture and deployment both reject external body wrenches and unrepresented velocity
targets. CUDA graphs stay enabled; the existing ANYdrive implementation steps outside the
graph, with learned physics replayed at each 5 ms tick.

Collect and fit
~~~~~~~~~~~~~~~

Install the optional PhysicsNeMo Robotics dependency described above. These commands use
``--no-sync`` to preserve that separately installed experimental dependency:

.. code-block:: bash

   uv run --no-sync python scripts/learning/nerd/record.py \
       --task Isaac-Cartpole-NeRD-Explore --output logs/nerd/cartpole/data.hdf5 \
       --num_envs 1024 --steps 1500
   uv run --no-sync python scripts/learning/nerd/train.py \
       --dataset logs/nerd/cartpole/data.hdf5 --output logs/nerd/cartpole/model \
       --wandb_project <project> --wandb_entity <entity>

   uv run --no-sync python scripts/learning/nerd/record.py \
       --task Isaac-Velocity-Flat-Anymal-C-NeRD-Explore --output logs/nerd/anymal/data.hdf5 \
       --num_envs 4096 --steps 600 --action_scale 1.5
   uv run --no-sync python scripts/learning/nerd/train.py \
       --dataset logs/nerd/anymal/data.hdf5 --output logs/nerd/anymal/model \
       --wandb_project <project> --wandb_entity <entity>

The default Cartpole recipe uses 30,000 updates and ten input frames. ANYmal C uses 60,000
updates and thirteen frames, with its network geometry configured alongside the task. The
training loop samples history from flattened device-resident transitions, so memory does not
multiply by history length. Data must still fit on the selected device. ``--config`` accepts
JSON overrides of ``NeRDTrainCfg``; a supplied ``network`` dictionary replaces the task's
network geometry. For a smoke test, collect 32 steps in four worlds and fit
with ``--max_updates 3 --batch_size 16 --log_interval 1 --save_interval 1``.

Collection writes native HDF5 episodes, a JSON representation descriptor, and the resolved
task YAML beside the dataset. Keep those files together. The recorder publishes preallocated
``state``, ``next_state``, ``control``, ``contacts``, ``terminated`` and ``truncated`` tensors.
Contact features come from the pre-state; executed control is read after the step; the terminal
successor is read before reset. An online consumer can read ``latest`` with
``record_to_episodes=False`` and feed replay or a prediction-error objective without a writer.

Fitting splits complete episodes, repeats the first frame for reset prefixes, includes terminal
transitions, and rejects discontinuous episodes. It reports training/held-out normalized MSE
and held-out delta MSE. These measure dynamics prediction, not policy task performance.

``last.pt`` contains network weights, fixed normalization, physical model identity, frame and
control conventions, optimizer state, sampler state and progress. Saves are atomic, happen at
least every 120 seconds between updates, and are uploaded synchronously when W&B is enabled.
SIGINT/SIGTERM finishes the current update and saves. Resume with:

.. code-block:: bash

   uv run --no-sync python scripts/learning/nerd/train.py \
       --dataset logs/nerd/anymal/data.hdf5 --output logs/nerd/anymal/model \
       --checkpoint logs/nerd/anymal/model/last.pt --max_updates 80000 \
       --wandb_project <project> --wandb_entity <entity>

The checkpoint owns the normalization statistics. Data, split, network and optimization settings
must match; the update budget can increase without changing the learning-rate decay schedule.

Model and precision options
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``network_type`` selects the PhysicsNeMo network class. ParallelNeRD is the default. To use
VectorNeRD, supply a JSON configuration such as:

.. code-block:: json

   {
     "network_type": "physicsnemo_robotics.models.nerd:NewtonNeRDVectorTransformer",
     "network": {
       "num_heads": 8, "hidden_size": 192, "head_hidden": 64,
       "temporal_depth": 4, "position_encoding": "relative_lag",
       "temporal_attention_spans": [4, 3, 3, 3]
     }
   }

``precision`` accepts ``float32`` (default), ``bfloat16`` or ``float16``. Parameters, normalized
targets and losses remain FP32. FP16 uses a saved gradient scaler and retries a skipped update
with the same samples; a nonfinite forward prediction stops training. ParallelNeRD FP16 requires
the dependency's attention matrix multiplications to disable autocast explicitly. The original
integration's runtime monkeypatch is not used. Development validation used PhysicsNeMo Robotics
revision ``a81160ef`` plus the attention-accumulation fix ``ee6d2e0``. That fix is an unpublished
dependency; the earlier revision alone is insufficient for safe FP16 fitting.

Launch distributed fitting through Torch's standard launcher:

.. code-block:: bash

   uv run --no-sync python -m torch.distributed.run --standalone --nproc_per_node=2 \
       scripts/learning/nerd/train.py --dataset logs/nerd/anymal/data.hdf5 \
       --output logs/nerd/anymal/distributed --batch_size 4096

The batch size is global and must be divisible by the rank count. Each rank holds the dataset,
draws an independent sample stream and participates in gradient averaging. Validation is
partitioned across ranks. Rank zero owns logs, checkpoint publication and W&B uploads.
Checkpoints save every rank's sampler state; resume requires the same rank count.

Train and evaluate a policy
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Select the learned task and pass its dynamics checkpoint through normal task overrides:

.. code-block:: bash

   uv run --no-sync python scripts/reinforcement_learning/train.py \
       --rl_library rsl_rl --task Isaac-Cartpole-NeRD-Learned --num_envs 4096 \
       env.sim.physics.solver_cfg.checkpoint=logs/nerd/cartpole/model/last.pt
   uv run --no-sync python scripts/reinforcement_learning/train.py \
       --rl_library rsl_rl --task Isaac-Velocity-Flat-Anymal-C-NeRD-Learned --num_envs 4096 \
       env.sim.physics.solver_cfg.checkpoint=logs/nerd/anymal/model/last.pt

   uv run --no-sync python scripts/reinforcement_learning/play.py \
       --rl_library rsl_rl --task Isaac-Velocity-Flat-Anymal-C-NeRD \
       --checkpoint /path/to/policy/model_299.pt --num_envs 16 --video --video_length 600

The last command evaluates the learned-physics policy in MuJoCo Warp. To evaluate the same
policy in learned physics, use ``-Learned`` and the same dynamics-checkpoint override. The
policy observation/action layouts and objectives agree between the two configurations.
ANYmal's acceleration penalty uses the control period in both solvers. Contact-force rewards,
torque penalties and physics randomizations absent from the model's inputs are disabled in both.

Scope and migration
~~~~~~~~~~~~~~~~~~~

These are joint-backed rigid scenes with fixed physical parameters. The solver rejects a changed
physical model, unsupported timing, external body wrenches and unmodeled velocity commands.
Physical identity includes fixed-root orientation/height, body flags, collision filtering and
per-world topology. Horizontal reset-origin translations are ignored only for translation-invariant
flat scenes. Version-2 checkpoints include architecture, gradient-scaler and per-rank sampler
state. Recollect/refit datasets whose older physical fingerprint omitted these properties.
ANYmal's fixed-plane probes do not represent rough terrain. A rough-terrain model needs a matching
terrain feature contract for collection and learned rollout; changing the task's terrain alone
does not provide that contract.

Native HDF5 and checkpoints replace the experimental ``isaaclab_physicsnemo`` Zarr/artifact
launchers. Existing artifacts are not silently converted or treated as compatible. Historical
training results from that branch are not new validation of this workflow. The native integration
tests cover both tasks, terminal capture and retained values, offline update/resume, and repeated
learned-solver graph replay with asynchronous resets and intervening tensor allocations.
They also cover the 5 ms effort path, VectorNeRD, FP16 update/resume and two-rank CPU gradient
averaging against an independent combined-batch update. These are integration checks, not
long-horizon policy benchmarks or a multi-GPU scaling measurement.

Joint wrenches and contact forces are not learned outputs. Reconstruction requires the physical
model, contact law and acceleration in addition to position/velocity. Endpoint state alone does
not generally determine a unique contact-force distribution. Analytic reconstruction is being
validated separately and is not exposed as a learned-solver sensor capability.
