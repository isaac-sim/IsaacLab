.. _nerd:

Offline NeRD: Cartpole and ANYmal C
===================================

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

NeRD physics always consumes applied joint efforts from ``Control.joint_f``. The actuator owns
position/velocity commands, delays, internal state and motor-effort clipping. It runs before each
physics step, and capture copies its final effort at that same solver boundary. There is no
target-conditioned physics mode or recorder control selector. Solver-side motor drives, including
damping drives, and MuJoCo-specific actuators are rejected because they bypass this boundary.
External body wrenches remain unsupported; passive physical forces are part of the fixed model.

Cartpole records at 120 Hz, with its damping and effort limits implemented by a Newton actuator.
ANYmal C records every 5 ms after ANYdrive evaluates and clips its effort. Its state features use
the root heading frame, body-frame gravity and four fixed foot-sphere probes. The learned task
keeps ANYdrive and advances learned physics four times per 20 ms policy action. Each NeRD step
predicts one physics transition, then Newton forward kinematics updates body state.

Capture requires ``decimation=1`` and ``num_substeps=1``; policy training can use decimation to
repeat actuator evaluation and learned physics. Both numerical and learned physics enable CUDA
graphs. The learned manager retains a private Torch allocation pool for its captured temporaries.
The existing ANYdrive implementation steps outside the graph, with learned physics replayed at
each 5 ms tick. Position and velocity controllers are supported when their efforts are computed
upstream; NeRD does not predict motor effort as an additional output.

Rough terrain and live contacts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Isaac-Velocity-Rough-Anymal-C-NeRD`` and its ``-Explore`` / ``-Learned`` variants use the
same 5 ms applied-effort contract with Newton collision detection enabled. NeRD replaces the
numerical dynamics solve; collision detection, ANYdrive and forward kinematics remain.
The exploration task uses terrain-aware reset patches to avoid spawning inside raised terrain.
The policy observations are proprioceptive; these tasks do not add a terrain-height observation.

Set ``NeRDRecorderCfg.max_contacts`` to a positive capacity to select PhysicsNeMo Robotics'
``RigidContactSetEncoder``. It captures the collision pipeline's pre-solver contacts after
actuator evaluation, without rerunning collision detection. Each packed contact contains validity,
body/other-body identity, dynamic-body status, point, normal, lever arm, signed gap and relative
velocity. The learner converts these to typed, permutation-invariant contact sets: only valid
continuous geometry is normalized; identities and masks retain their semantics. Contact and state
histories use the same episode boundaries and repeated reset prefixes. Capacity overflow stops
capture or deployment instead of silently dropping constraints.

Terrain geometry can change between collection and deployment because it is represented through
live contacts. The checkpoint still fixes robot geometry, inertias, gains, limits, collision filtering,
gravity and the uniform external contact material. This contract does not support independently
moving external objects or per-surface material variation. Wrench accuracy is a separate validation
requirement; enabling contact conditioning does not enable learned force sensors.

.. code-block:: bash

   uv run --no-sync python scripts/learning/nerd/record.py \
       --task Isaac-Velocity-Rough-Anymal-C-NeRD-Explore \
       --output logs/nerd/rough/exploration.hdf5 --num_envs 128 --steps 1600 \
       --action_scale 0.75 --action_hold 4
   uv run --no-sync python scripts/learning/nerd/train.py \
       --dataset logs/nerd/rough/exploration.hdf5 --output logs/nerd/rough/model \
       --wandb_project <project> --wandb_entity <entity>

``--action_hold`` sets the interval between action updates while retaining every physics transition.
For locomotion data, ``--policy /path/to/exported/policy.pt`` accepts a feedforward TorchScript
policy consuming the task's ``policy`` observations. Match its observation layout, joint ordering
and action scale through task configuration. A policy trained in PhysX may need
``env.scene.robot.joint_ordering=physx`` when collecting with Newton. Multiple compatible
datasets can be passed together to ``train.py --dataset``. Use separate terrain seeds for final
evaluation; the learner's episode split alone does not establish terrain generalization.

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
       --num_envs 4096 --steps 2400 --action_scale 1.5 --action_hold 4
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
Contact features come from the pre-state; applied effort is copied immediately before the solver; the terminal
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
torque penalties and physics randomizations absent from the model's inputs are disabled by default in both.

Reconstructed forces for rewards
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An existing applied-effort checkpoint can optionally report estimated contact forces and incoming
joint wrenches. Enable reconstruction in the learned task configuration and declare ordinary sensors;
no additional network or force labels are required:

.. code-block:: python

   from isaaclab.envs import mdp
   from isaaclab.managers import RewardTermCfg, SceneEntityCfg
   from isaaclab.sensors import ContactSensorCfg, JointWrenchSensorCfg

   cfg.sim.physics.solver_cfg.reconstruct_forces = True
   cfg.scene.robot.enable_joint_wrench = True
   cfg.scene.contact_forces = ContactSensorCfg(
       prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_friction_forces=True
   )
   cfg.scene.joint_wrenches = JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot")
   cfg.rewards.contact_load = RewardTermCfg(
       func=mdp.contact_forces,
       weight=-1e-4,
       params={"threshold": 100.0, "sensor_cfg": SceneEntityCfg("contact_forces")},
   )

Custom reward functions can read ``env.scene["contact_forces"].data.net_forces_w.torch`` for
net contact forces on each sensed body in world coordinates (N). They can read
``env.scene["robot"].data.body_joint_wrench.torch`` for incoming joint wrenches, ordered as
``[force_x, force_y, force_z, torque_x, torque_y, torque_z]`` in N and N m, expressed in the
child-side joint frame with torque taken about its anchor. ``JointWrenchSensor`` exposes the
same kind of measurement through its existing ``data.force`` and ``data.torque`` interfaces.
These are the latest physics-step measurements, not control-period averages. Contact force
and incoming joint force are different quantities; gravity, inertia and motor effort also
contribute to a joint reaction. The external body-force input buffer remains an input and is
not used to publish estimates. External body-wrench inputs and IMU acceleration remain unsupported.

Reconstruction runs Newton collision detection even for a flat checkpoint without contact inputs.
It uses the physical mass matrix and contact Jacobian to project the predicted acceleration while
holding applied motor effort fixed, enforcing unilateral contact and elliptic friction cones.
Newton-Euler evaluation then supplies incoming joint wrenches when requested. Only the diagnostic
acceleration is projected: the learned positions and velocities are unchanged. CUDA graph capture
remains enabled, and persistent projection buffers belong to the solver.

This is an experimental estimate, not equivalence to MuJoCo's compliant contact solver. Endpoint
motion does not uniquely identify every contact-force distribution, and small velocity errors can
produce large force errors after division by the physics timestep. Validate the force-dependent
reward against held-out teacher trajectories before relying on it for policy training; successful
state prediction alone does not establish wrench accuracy. The bounded projection uses FP64 setup
and a fused FP32 solve with at most 1,000 iterations and a stopping tolerance of ``1e-4`` on the
largest primal residual or force-iterate change in scaled cone coordinates. This is not a bound on
sensor error or a guarantee that every solve reaches that tolerance.

The initial implementation supports CUDA, 1--60 velocity DOFs per world, at most 60 constraint
rows per world, joint limits, and frictionless or three-dimensional elliptic contacts. Equalities,
tendons, joint friction loss, adhesive contacts and higher-dimensional contact cones are rejected.
Capacity overflow and nonfinite reconstructed values fail explicitly. Each frictional contact
uses three constraint rows, so 60 rows do not imply support for 60 simultaneous frictional contacts.

Scope and migration
~~~~~~~~~~~~~~~~~~~

These are joint-backed rigid scenes with fixed physical parameters. The solver rejects a changed
physical model, unsupported timing, external body wrenches and solver-side motor drives.
Physical identity includes fixed-root orientation/height, body flags, collision filtering and
per-world topology. Horizontal reset-origin translations are ignored only for translation-invariant
flat scenes. Version-2 checkpoints include architecture, gradient-scaler and per-rank sampler
state. Recollect/refit datasets whose older physical fingerprint omitted these properties.
Target-conditioned datasets and checkpoints are rejected. The former flat ANYmal ``-Effort-Explore``
and ``-Effort-Learned`` tasks are now the standard ``-Explore`` and ``-Learned`` tasks, with no
compatibility aliases. Their applied-effort checkpoints retain the same input and physical contract.
Cartpole datasets and checkpoints collected with the former solver-side damping drive must be
recollected and refit for its actuator-owned damping.
ANYmal's flat-model fixed-plane probes do not represent rough terrain. Use the rough task family
to collect and fit the live-contact representation; flat and rough checkpoints are not interchangeable.

Native HDF5 and checkpoints replace the experimental ``isaaclab_physicsnemo`` Zarr/artifact
launchers. Existing artifacts are not silently converted or treated as compatible. Historical
training results from that branch are not new validation of this workflow. The native integration
tests cover both tasks, terminal capture and retained values, offline update/resume, and repeated
learned-solver graph replay with asynchronous resets and intervening tensor allocations.
They also cover clipped actuator effort retained before buffer reuse, rejection of legacy target
models and solver-side drives, VectorNeRD, FP16 update/resume and two-rank CPU gradient
averaging against an independent combined-batch update. These are integration checks, not
long-horizon policy benchmarks or a multi-GPU scaling measurement.
The rough case also exercises live contact capture, contact-conditioned fitting and resume,
deployment with a different terrain and world count, and rejection of changed contact friction.

Force-reconstruction tests additionally cover fixed-effort support and flight, friction saturation,
contact capacity boundaries, masked resets, and existing sensor/reward access during graph replay.
Enabling reconstruction preserves the learned trajectory in the flat ANYmal integration test.
