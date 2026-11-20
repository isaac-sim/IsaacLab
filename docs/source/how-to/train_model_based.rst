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
and training source archives. Keep downloaded and converted data outside Git;
an existing local preparation can be reused for subsequent experiments.

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

Prepare BONES-SEED instead
--------------------------

After downloading the official G1 archive yourself, reuse the matching G1 MJCF:

.. code-block:: bash

   uv run python scripts/tools/prepare_bones.py \
       --archive /data/bones-seed/g1.tar.gz \
       --mjcf /data/g1_tracking/g1_description/g1_29dof.xml \
       --output /data/bones_prepared --workers 16 --source_fps 120

This converter reads the archive directly and writes filtered, resampled 50 Hz
NPY poses under ``motions/``, per-clip decisions in ``motions.jsonl``, and
settings and hashes in ``manifest.json``. Its filters are local preprocessing
choices, not a claim to reproduce the paper's processed subset.

For single-GPU training, use the same training command with
``--motion_dir /data/bones_prepared/motions`` and
``env.commands.motion.source_fps=50``. Do not use the original archive's 120 Hz
rate for the converted poses. Multi-GPU reference sharding requires the complete
50 Hz NPZ format described in
:doc:`/source/experimental-features/model_based`; these NPY files are not that
distributed input format.
