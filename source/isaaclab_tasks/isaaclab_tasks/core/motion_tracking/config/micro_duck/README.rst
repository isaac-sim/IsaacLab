MicroDuck motion tracking
=========================

This task extends the shared motion-tracking framework introduced with G1.
It reuses its observations, actions, reference-state initialization and PPO
configuration, with the existing MicroDuck asset and native BAM actuators.
The supported physics backend is Newton with MuJoCo Warp, at 200 Hz physics
and 50 Hz control. Playback supports the Newton GL viewer on Linux. Other
physics backends and renderers have not been validated for this task.

Task
----

``Isaac-Motion-Tracking-MicroDuck-Feet-Head`` uses one configuration in
``tracking_env_cfg.py``. It adds foot-position and head-position (``jaw_soft``)
tracking to the shared whole-body rewards, both with weight 4 and a
0.02-meter scale. The action-rate L2 penalty has weight -0.05.

The task terminates when either foot's absolute height error exceeds
0.09 meters, the head's exceeds 0.18 meters, or the anchor's exceeds 0.15 meters.
The inherited orientation failure condition also applies, with threshold 0.8.
These are tracking-error limits, not minimum body heights or fall-only detection.
Foot and head rewards measure
XYZ error after horizontal position and yaw alignment, retaining world height.
The foot term averages the two per-foot exponential rewards; the head term
is independent of the feet. Source contact labels are not used.

The policy has 79 actor observations, 220 critic observations and
14 joint-position actions. The networks are unchanged from the shared G1
tracking configuration.

Reference data
--------------

Provide a compatible reference NPZ separately through
``env.commands.motion.motion_file``. No motion dataset, checkpoint or video
is bundled with this task. Use only data you are authorized to access and
redistribute.

The shared loader requires ``fps``, ``joint_names``, ``body_names``,
``joint_pos``, ``joint_vel``, ``body_pos_w``, ``body_quat_w``,
``body_lin_vel_w`` and ``body_ang_vel_w``. References must contain the
configured tracked links and articulation joints, use meters and radians,
store quaternions in WXYZ order, and run at 50 Hz. Link poses are expressed
in world coordinates. Arrays are aligned by their stored names.

The MicroDuck command in ``tracking_env_cfg.py`` maps source body ``neck_pitch`` to asset body
``neck_pitch_1`` without renaming the joint. By default, input linear
velocities refer to link origins. It uses the asset's nominal local COM
offsets to convert them once, in memory, to COM velocities. Link poses and
angular velocities are unchanged. Neither the original NPZ nor the shared
``MotionLoader`` is modified.

For inputs already containing COM velocities for the same inertial model,
set ``env.commands.motion.input_velocity_point=com``. Do not use this option
for the original link-origin velocity references. Extra source contact and
timing arrays are not part of the policy command.

Train and play
--------------

From an Isaac Lab checkout containing the shared G1 motion-tracking framework
and this package, install the training dependencies and provide your paths:

.. code-block:: bash

   uv sync --frozen --extra rsl-rl
   export MOTION_FILE=/absolute/path/to/motion.npz
   export CHECKPOINT=/absolute/path/to/model.pt

   uv run --no-sync isaaclab train \
       --task Isaac-Motion-Tracking-MicroDuck-Feet-Head \
       --rl_library rsl_rl --num_envs 1024 --max_iterations 30000 --seed 42 \
       env.commands.motion.motion_file="$MOTION_FILE"

   uv run --no-sync isaaclab play \
       --task Isaac-Motion-Tracking-MicroDuck-Feet-Head \
       --rl_library rsl_rl --num_envs 1 --seed 42 \
       --checkpoint "$CHECKPOINT" --viz newton_gl --real-time \
       env.commands.motion.motion_file="$MOTION_FILE"

The single configuration includes both foot and head tracking; no reward-weight
or termination-threshold overrides are needed for the common training setup.
Training starts from scratch unless a checkpoint is explicitly supplied.

Playback uses the shared adaptive start-frame sampling, without reset-state
perturbations or observation corruption. Native actuator dynamics and their randomization
remain enabled. Clip completion is a timeout, not a tracking failure.
The default episode limit is 14 seconds; set ``env.episode_length_s`` longer
than the reference duration for longer clips. Full-clip completion and
trajectory errors should be evaluated together, since short training
episodes alone do not establish motion-tracking quality.
