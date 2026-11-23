:orphan:

SuperMimic-style Motion Tracking
========================================

Train a G1 motion-tracking policy with the ``model_based`` backend, then play or
record its checkpoint. Install Isaac Lab and its training dependencies first.
See :doc:`/source/experimental-features/model_based` for training options.

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

Train
-----

For LAFAN1 motions sampled at 30 Hz:

.. code-block:: bash

   uv run --frozen isaaclab train --rl_library model_based \
       --task Isaac-Motion-Tracking-GenCoord-G1 \
       --num_envs 4096 --max_iterations 40000 \
       --motion_dir /data/lafan1_download/g1 \
       --asset_path /data/robots/g1/g1.usda \
       --log_dir logs/model_based/g1 \
       env.commands.motion.source_fps=30

The output directory contains ``model_*.pt`` checkpoints, saved configurations,
and ``metrics.jsonl`` with iterations, losses, and rewards. The command above
saves its final checkpoint as ``logs/model_based/g1/model_040000.pt``.

Play and record
---------------

Play a saved checkpoint:

.. code-block:: bash

   uv run --frozen isaaclab play --rl_library model_based \
       --checkpoint logs/model_based/g1/model_040000.pt \
       --num_envs 1 --steps 3000 --visualizer newton_gl --real-time

Playback restores the checkpoint's configuration. Keep its motion and asset
files accessible, or supply their new locations with ``--motion_dir`` and
``--asset_path``.

To record a 30-second video:

.. code-block:: bash

   uv run --frozen --extra video isaaclab play --rl_library model_based \
       --checkpoint logs/model_based/g1/model_040000.pt \
       --num_envs 1 --video newton_gl --video_length 1500

Pretrained dance sample
-----------------------

The GenCoord dance sample uses the BONES-SEED
``dance_hiphop_box_step_R_fast_001__A318_M`` motion. Given the sample checkpoint,
reference file, and matching USD asset, run:

.. code-block:: bash

   uv run --frozen isaaclab play --rl_library model_based \
       --checkpoint /path/to/dance/checkpoint.pt \
       --motion_dir /path/to/dance/motion.npz \
       --asset_path /path/to/dance/g1.usd \
       --num_envs 1 --visualizer newton_gl --real-time

The reference lasts 7.5 seconds and restarts at clip boundaries. The sample
checkpoint is for GenCoord playback. The ``--checkpoint pretrained`` shortcut
requires the sample assets
to be available on the Isaac Lab asset server.

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

For BONES-SEED CSV clips:

.. code-block:: bash

   uv run --frozen isaaclab train --rl_library model_based \
       --task Isaac-Motion-Tracking-GenCoord-G1 \
       --num_envs 4096 --max_iterations 40000 \
       --motion_dir /data/bones-seed/g1 \
       --asset_path /data/robots/g1/g1.usda \
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

Sim-to-real
-----------

For G1 hardware deployment, see the
`G1 LEAPP deployment guide <https://isaac-sim.github.io/IsaacLab/main/source/how-to/train_model_based.html#g1-leapp-deployment-adapter>`_.
