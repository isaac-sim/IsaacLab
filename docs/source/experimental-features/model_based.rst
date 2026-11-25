SuperMimic-style Motion Tracking
========================================

RSL-RL SuperMimic learns a world model from simulation and trains a
policy through predicted motion rollouts. It is based on
`SuperMimic <https://github.com/HybridRobotics/supermimic>`_ and uses Isaac Lab's
motion-tracking tasks and standard train/play commands.

How it works
------------

Training repeats three steps:

1. Run the current policy in simulation and store state transitions in a replay buffer.
2. Fit the world model to short sequences sampled from that buffer.
3. Predict short motion rollouts with the world model and improve the policy by
   backpropagating the tracking return through those predictions. A critic estimates
   the return beyond each rollout.

The world model stays fixed during each policy update and continues learning in
later iterations. Gradients pass through the learned model, not the simulator.
Tracking rewards and termination are computed from predicted states and the reference.

.. list-table:: Model inputs and outputs
   :header-rows: 1
   :widths: 20 50 30

   * - Model
     - Inputs
     - Output
   * - World model
     - Root height, projected gravity, root-frame velocities, joint state,
       actions, and body masses.
     - Root and joint velocity increments and self-collision events;
       poses follow by integration.
   * - Policy (actor)
     - Robot observations or observation history, and the reference joint command.
     - 29 normalized joint-position commands, scaled to joint targets.
   * - Critic
     - Full robot state, reference-motion goal, and simulation context.
     - Expected discounted return as a scalar.

The selected task and preset determine the state representation and observation
history. Playback uses only the policy; the world model and critic are needed for training.

Tasks and configuration
-----------------------

SuperMimic uses ``Isaac-Motion-Tracking-G1``, the same environment as PPO.
Select ``--agent rsl_rl_supermimic_cfg_entry_point presets=state_obs`` to enable
model-based learning. The default agent remains PPO.

The environment supplies commands, reset sampling, events, observations, rewards
and termination thresholds. The differentiable task reads the resolved manager
configurations and evaluates their supported terms on predicted states.
For 29 joints, the world model consumes 68 state features and predicts 35 velocity
increments. Root and joint poses use semi-implicit integration at the task's
50 Hz control rate, with four physics steps per control step.

The agent uses Muon/Adam optimizers, one observation frame and 32-step policy
rollouts. Override individual options on the command line, for example:

* ``agent.algorithm.horizon=16`` for 16-step predicted policy rollouts.
* ``--logger wandb --log_project_name PROJECT`` for W&B logging.

.. raw:: html

   <video controls playsinline preload="metadata" style="width: 100%; height: auto;" aria-label="SuperMimic reference and real G1">
     <source src="https://github.com/Double7sBurger/IsaacLab/releases/download/g1-visual-locomotion-media/G1_Reference_vs_Real_3motions.mp4" type="video/mp4">
   </video>

Inputs
------

* A floating-base, 29-joint G1 USD asset matching your reference motions.
* A motion file or directory of clips in CSV, NPY, or NPZ format.
* The source frame rate and, for CSV files, the column mapping described below.

Download the dataset from its provider and follow its access and license terms:

* `LAFAN1 Retargeting Dataset
  <https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/tree/ce1572906efe6157840e8474d5a0d7aa87481e74>`_:
  download ``g1/`` and ``robot_description/g1/``, including meshes.
* `BONES-SEED <https://huggingface.co/datasets/bones-studio/seed>`_:
  request access with your own account, then download and extract ``g1.tar.gz``.

Training reads your local data without modifying it. If your robot description
is in URDF format, follow :doc:`/source/how-to/import_new_asset` to import a
floating-base USD asset.

Motion formats
--------------

Each file contains one clip. Joint values follow the USD articulation's joint
order. The loader resamples CSV and NPY clips to the task's 50 Hz control rate.

* **CSV or NPY poses:** the default layout is root XYZ position in meters,
  root XYZW quaternion, then joint angles in radians. Set
  ``env.commands.motion.source_fps`` to the input frame rate.
* **Named CSV columns:** set ``env.commands.motion.csv.header=true`` and map
  the root and joint columns. The example below shows BONES-SEED's format.
* **NPZ references:** use 50 Hz float arrays ``joint_pos``, ``joint_vel``,
  ``root_pos_w``, ``root_quat_w`` (WXYZ), ``root_lin_vel_w``, and
  ``root_ang_vel_w``. Joint arrays have shape ``(frames, joints)``; root arrays
  have shape ``(frames, 3)`` or ``(frames, 4)`` for quaternions. Units are
  meters, radians, and per-second velocities; root velocities use world axes.
  An optional ``fps`` field must be 50. Directories containing NPZ files select
  those files; otherwise CSV and NPY files are loaded.

Full-body NPZ files are also accepted: provide ``fps``, ``joint_names``,
``body_names``, joint positions/velocities and ``body_pos_w``, ``body_quat_w``
(WXYZ), ``body_lin_vel_w``, ``body_ang_vel_w``. Named channels are reordered to
the robot and tracked-body order.

For BONES-SEED CSV clips:

.. code-block:: bash

   uv run --frozen isaaclab train --rl_library rsl_rl \
       --task Isaac-Motion-Tracking-G1 \
       --agent rsl_rl_supermimic_cfg_entry_point presets=state_obs \
       --num_envs 4096 --max_iterations 40000 \
       env.commands.motion.motion_file=/data/bones-seed/g1 \
       env.scene.robot.spawn.usd_path=/data/robots/g1/g1.usda \
       env.commands.motion.source_fps=120 \
       env.commands.motion.csv.header=true \
       'env.commands.motion.csv.root_position=["root_translateX","root_translateY","root_translateZ"]' \
       'env.commands.motion.csv.root_rotation=["root_rotateX","root_rotateY","root_rotateZ"]' \
       env.commands.motion.csv.rotation_format=euler_xyz \
       'env.commands.motion.csv.joint_column_template="{name}_dof"' \
       env.commands.motion.csv.position_scale=0.01 \
       env.commands.motion.csv.angles_in_degrees=true

For other CSV layouts, configure:

* ``root_position`` and ``root_rotation``: column names or zero-based indices.
* ``rotation_format``: ``xyzw``, ``wxyz``, or ``euler_xyz`` (extrinsic XYZ).
* ``joint_column_template`` or ``joint_columns``: joint-name mappings.
* ``position_scale``: scale source lengths to meters.
* ``angles_in_degrees``: convert Euler and joint angles from degrees.
* ``world_rotation``: XYZW rotation from source world axes to training axes.

Optional ``root_linear_velocity``, ``root_angular_velocity``, and
``joint_velocity_columns`` mappings provide per-second velocities. Root
velocities use source world axes. Missing velocities are estimated within each
clip. See :class:`isaaclab.utils.datasets.motion_loader.CsvMotionCfg` for all
CSV settings.

Train
-----

For LAFAN1 motions sampled at 30 Hz:

.. code-block:: bash

   uv run --frozen isaaclab train --rl_library rsl_rl \
       --task Isaac-Motion-Tracking-G1 \
       --agent rsl_rl_supermimic_cfg_entry_point presets=state_obs \
       --num_envs 4096 --max_iterations 40000 \
       env.commands.motion.motion_file=/data/lafan1_download/g1 \
       env.scene.robot.spawn.usd_path=/data/robots/g1/g1.usda \
       env.commands.motion.source_fps=30

Training writes checkpoints, TensorBoard metrics and the resolved configurations to
``logs/rsl_rl/g1_motion_tracking/<timestamp>``. The final checkpoint is ``model_39999.pt``.

Training outputs
----------------

Outputs are saved under ``logs/rsl_rl/g1_motion_tracking/<timestamp>``:

* ``model_*.pt``: actor, critic, world-model weights and optimizer states.
* TensorBoard events with training losses, rewards and throughput.
* Saved environment and agent configurations under ``params/``.

One iteration advances every environment by one control step. For example,
40,000 iterations with 4096 environments collect 163,840,000 transitions.
``iteration`` in the learner checkpoint reports the separate count of learning cycles.

Multi-GPU training
------------------

Use ``train_multigpu`` with environment and learner batch sizes specified
**per GPU**. For two GPUs and 4096 environments in total:

.. code-block:: bash

   uv run --frozen isaaclab train_multigpu --rl_library rsl_rl --num_gpus 2 -- \
       --task Isaac-Motion-Tracking-G1 \
       --agent rsl_rl_supermimic_cfg_entry_point presets=state_obs --num_envs 2048 \
       env.commands.motion.motion_file=/path/to/motions \
       env.scene.robot.spawn.usd_path=/path/to/g1.usd \
       agent.algorithm.world_model_cfg.batch_size=1024 agent.algorithm.policy_batch_size=2048

Each GPU samples the reference set in its own environments and stores local replay data.
Model updates are synchronized across GPUs; rank zero saves checkpoints and training metrics. The learning
rate is unchanged unless you override it. Playback uses the full reference set.

Play and record
---------------

Play a checkpoint using the same task, agent preset and motion inputs as training:

.. code-block:: bash

   uv run --frozen isaaclab play --rl_library rsl_rl \
       --task Isaac-Motion-Tracking-G1 \
       --agent rsl_rl_supermimic_cfg_entry_point presets=state_obs \
       --checkpoint /path/to/run/model_39999.pt \
       --num_envs 1 --visualizer newton_gl --real-time \
       env.commands.motion.motion_file=/data/lafan1_download/g1 \
       env.scene.robot.spawn.usd_path=/data/robots/g1/g1.usda \
       env.commands.motion.source_fps=30

To record 30 seconds, add ``--video newton_gl --video_length 1500`` and use
``uv run --frozen --extra video``. If training used a non-default preset, pass the
same ``--agent`` and model overrides when playing its checkpoint.

Pretrained dance sample
-----------------------

The pretrained dance sample uses the BONES-SEED
``dance_hiphop_box_step_R_fast_001__A318_M`` motion. Given the sample checkpoint,
reference file, and matching USD asset, run:

.. code-block:: bash

   uv run --frozen isaaclab play --rl_library rsl_rl \
       --checkpoint /path/to/dance/checkpoint.pt \
       --motion_dir /path/to/dance/motion.npz \
       --asset_path /path/to/dance/g1.usd \
       --num_envs 1 --visualizer newton_gl --real-time

The reference lasts 7.5 seconds and restarts at clip boundaries. The sample
checkpoint uses the motion encoder and its ten-frame reference window.

G1 LEAPP deployment adapter
------------------------------

Export a trained policy using its task, agent and motion settings:

.. code-block:: bash

   uv run --extra leapp isaaclab leapp export --rl_library rsl_rl \
       --task Isaac-Motion-Tracking-G1 \
       --agent rsl_rl_supermimic_cfg_entry_point presets=state_obs \
       --checkpoint /path/to/run/model_39999.pt \
       env.commands.motion.motion_file=/path/to/motion.npz \
       --export_save_path exported

Export the pretrained dance checkpoint and one reference clip to a LEAPP bundle:

.. code-block:: bash

   uv run --extra leapp isaaclab leapp export \
       --task Isaac-Motion-Tracking-GenCoord-G1 --rl_library rsl_rl \
       --checkpoint /path/to/dance/checkpoint.pt \
       --motion_dir /path/to/dance/hiphop.npz --export_save_path exported

Pass a local checkpoint path or URL and select one reference clip. For a newly
trained policy, set ``env.commands.motion.motion_file``; for the dance sample,
use ``--motion_dir``.
The output directory contains an ONNX model, a pipeline YAML, initial feedback
state, and ``motion.npz``. Keep these files together. The graph includes sensor
preprocessing, observation history, normalization, motion inference and joint
position scaling; it outputs joint targets and the checkpoint's PD gains.

Install Isaac ROS and the G1 driver as described in the
:doc:`visual locomotion deployment guide </source/policy_deployment/07_g1_sim2real/training_and_deploying_a_g1_visual_locomotion_policy>`.
The reference publisher also requires SciPy (``sudo apt-get install python3-scipy``).
After sourcing the ROS environment, run:

.. code-block:: bash

   python3 scripts/deploy/g1_deploy.py \
       --pipeline exported/<task>/<task>.yaml \
       --interface <iface>

The launcher starts the reference publisher when ``motion.npz`` is beside the
pipeline. Use ``--motion`` to select another exported clip with the same joint
order. Follow the existing guide's IMU calibration and operator enable/stop
procedure before engaging the controller.

The actor first tracks the clip's initial pose with zero reference velocity.
Start the motion after the robot is standing:

.. code-block:: bash

   ros2 service call /g1_policy/start_motion std_srvs/srv/Trigger '{}'

The controller blends into motion over one second, plays the clip once, then
blends back to tracking the final pose with zero reference velocity. The actor
continues receiving joint and IMU feedback throughout; the history contains the
blended action actually sent to the joint controller. Choose clips whose initial
and final poses support standing. Each deployment session plays the clip once;
reposition the robot before restarting deployment for another playback.
To return early to a stationary reference:

.. code-block:: bash

   ros2 service call /g1_policy/stand std_srvs/srv/Trigger '{}'

The reference publisher sends a timestamped ``isaac_ros_tensor_msgs/msg/TensorList``
on ``/g1_policy/motion`` at 50 Hz. Each float32 window has shape ``[10, 63]``:
29 joint positions, 29 joint velocities, a WXYZ root quaternion and a blend weight
per frame. Frames are spaced by five control steps, clamped at the clip boundary.
The graph receives pelvis orientation in XYZW order, pelvis-frame angular
velocity, and joint positions/velocities in the order declared in its YAML.
At motion start the publisher aligns the reference heading to the pelvis IMU.

To stop policy control, activate the driver's damping controller:

.. code-block:: bash

   ros2 control switch_controllers \
       --activate freeze_controller \
       --deactivate forward_joint_command_controller safety_controller
