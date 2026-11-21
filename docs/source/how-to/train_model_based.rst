:orphan:

Training model-based motion tracking
====================================

This guide prepares local G1 motion references and runs the standalone
``model_based`` backend. Install Isaac Lab and its training dependencies first.
See :doc:`/source/experimental-features/model_based` for the algorithm, source
attribution, configuration presets, and distributed-training constraints.

Download the data yourself
--------------------------

**Users must obtain the datasets from their original providers themselves.**
This repository does not distribute motion datasets, prepared dataset caches,
or trained checkpoints. Training reads local files; it does not download a
dataset or grant access to one. Review the provider's terms and complete any
access request before downloading. Dataset and robot-asset licenses are separate
from the licenses of the training code.

* For LAFAN1 G1 motions and the matching robot description, use the
  `LAFAN1 Retargeting Dataset
  <https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/tree/ce1572906efe6157840e8474d5a0d7aa87481e74>`_.
  Download the ``g1/`` directory, ``robot_description/g1/`` directory (including
  meshes), and the provider's license and README. The dataset card identifies
  LAFAN1's data license as CC BY-NC-ND 4.0; the repository's code license does not
  replace those data terms.
* For BONES-SEED, use the official
  `Bones Studio dataset page <https://huggingface.co/datasets/bones-studio/seed>`_.
  Accept its license and obtain access using your own account, then download
  ``g1.tar.gz`` and retain the provider's license and dataset revision. Do not
  substitute an internal storage location or an unofficial mirror.

Keep authentication credentials local and out of configuration files, commits,
and training source archives. Keep downloaded data outside Git. Motion loading
is read-only: column mapping, unit conversion, and resampling happen in memory,
without a converted dataset, source-file updates, or automatic motion filtering.

Prepare the G1 asset and LAFAN1 references
------------------------------------------

The following example assumes the downloaded dataset retains its upstream
directory layout under ``/data/lafan1_download``. Arrange the local inputs for
the preparation script:

.. code-block:: bash

   mkdir -p /data/g1_inputs
   cp -a /data/lafan1_download/g1 /data/g1_inputs/lafan1_g1
   cp -a /data/lafan1_download/robot_description/g1 /data/g1_inputs/g1_description

Run these commands from the Isaac Lab repository root. Choose an empty output
directory; ``--source_dir`` reuses your download without contacting Hugging Face.

.. code-block:: bash

   uv run python scripts/tools/prepare_g1_tracking.py \
       --source_dir /data/g1_inputs --output /data/g1_tracking
   uv run python scripts/tools/convert_mjcf.py \
       /data/g1_tracking/g1_description/g1_29dof.xml \
       /data/g1_tracking/g1_usd/g1_29dof.usd

The preparation writes a free-floating 29-joint MJCF and ``source.json`` with
the local source path and file hashes. Preserve the downloaded revision alongside
that manifest; local input hashes alone do not establish an upstream revision.
The USD converter writes the composed asset to
``/data/g1_tracking/g1_usd/g1_29dof/g1_29dof.usda``.

Train and play
--------------

Use the prepared local paths and the LAFAN1 source rate of 30 Hz:

.. code-block:: bash

   uv run --frozen isaaclab train --rl_library model_based \
       --task Isaac-Motion-Tracking-GenCoord-G1 \
       --num_envs 4096 --max_iterations 40000 \
       --motion_dir /data/g1_tracking/lafan1_g1 \
       --asset_path /data/g1_tracking/g1_usd/g1_29dof/g1_29dof.usda \
       --kinematics_path /data/g1_tracking/g1_description/g1_29dof.xml \
       env.commands.motion.source_fps=30

After training, replace the checkpoint path with the file produced by your run:

.. code-block:: bash

   uv run --frozen isaaclab play --rl_library model_based \
       --checkpoint /path/to/run/model_040000.pt \
       --num_envs 1 --steps 3000 --visualizer newton_gl --real-time

Keep the local asset and motion paths accessible for playback. Reference
trajectories are kinematic targets; reproducing them does not establish that
the retargeted motion is dynamically feasible on the robot.

Read other CSV datasets
-----------------------

The shared ``isaaclab.utils.datasets.motion_loader`` adapter reads robot poses
using ``CsvMotionCfg``. Each CSV is one complete, uniformly sampled clip. The
default is headerless XYZ/XYZW/joint-radian data in the task's joint order, as
used by the LAFAN1 example. Set ``source_fps`` to the actual input rate.

For named columns, set ``csv.header=true`` and map the root position and rotation
columns. Joint columns default to the robot's joint names; use
``joint_column_template`` for a common naming pattern or ``joint_columns`` to
map individual target joint names to source columns. Headerless files can use
zero-based column indices instead. Extra numeric columns, such as frame IDs,
are ignored unless mapped. Missing required columns or nonfinite motion values
raise an error; the loader does not silently remove samples.

After downloading and extracting the official BONES-SEED G1 archive yourself,
point training at its CSV directory. Reuse the matching G1 USD/MJCF asset paths
from the previous example:

.. code-block:: bash

   uv run --frozen isaaclab train --rl_library model_based \
       --task Isaac-Motion-Tracking-GenCoord-G1 \
       --num_envs 4096 --max_iterations 40000 \
       --motion_dir /data/bones-seed/g1 \
       --asset_path /data/g1_tracking/g1_usd/g1_29dof/g1_29dof.usda \
       --kinematics_path /data/g1_tracking/g1_description/g1_29dof.xml \
       env.commands.motion.source_fps=120 \
       env.commands.motion.csv.header=true \
       'env.commands.motion.csv.root_position=["root_translateX","root_translateY","root_translateZ"]' \
       'env.commands.motion.csv.root_rotation=["root_rotateX","root_rotateY","root_rotateZ"]' \
       env.commands.motion.csv.rotation_format=euler_xyz \
       'env.commands.motion.csv.joint_column_template="{name}_dof"' \
       env.commands.motion.csv.position_scale=0.01 \
       env.commands.motion.csv.angles_in_degrees=true

The adapter accepts XYZW or WXYZ quaternions and extrinsic XYZ Euler angles.
``position_scale`` converts source lengths to meters; ``angles_in_degrees``
applies to Euler angles and joint angles. ``world_rotation`` optionally rotates
the source world frame into the training world frame using an XYZW quaternion.
It does not change robot-local joint axes or retarget a human skeleton to G1.

Optional ``root_linear_velocity``, ``root_angular_velocity``, and
``joint_velocity_columns`` mappings read supplied per-second velocities. Root
velocities must be expressed in the source world frame. At the task's 50 Hz
rate, supplied velocities are retained; otherwise they are linearly resampled.
Only missing velocities are estimated from poses, separately within each clip.

CSV clips can also be sharded across GPUs. Both training and playback use the
same adapter, and checkpoints retain its configuration. The loader writes no
motion files or disk caches. Dataset-specific feasibility filtering is not part
of loading: choose clips appropriate for the scene, including required terrain
or objects. Loading the full BONES archive does not reproduce the paper's
filtered motion subset.
