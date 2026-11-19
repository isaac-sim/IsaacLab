.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

:orphan:

.. _g1-motion-tracking:

Training a G1 Motion-Tracking Policy
====================================

``Isaac-Motion-Tracking-G1`` trains the official 29-DoF Unitree G1, without dexterous hands,
to follow a reference motion using RSL-RL PPO and Newton's MuJoCo Warp solver.
Each run trains on one clip. The policy runs at 50 Hz, with physics and implicit PD at 200 Hz.

Prerequisites
-------------

Follow the :doc:`Isaac Lab installation </source/isaaclab/installation>` for Newton, then install
the RSL-RL and video-recording dependencies from the repository root:

.. code-block:: bash

   uv sync --extra rsl-rl --extra video

The first run downloads and caches the official G1 USD and the selected reference motion.
Internet access is required until those assets are cached. This task fixes the physics backend
in its configuration; no ``physics=`` or ``presets=`` selection is needed.

The :ref:`Command Builder <environment-browser>` also lists ``Isaac-Motion-Tracking-G1``.
Select **Train** or **Play** and **rsl_rl** to generate the base command.
Selecting **Pretrained checkpoint** also selects the CR7 reference clip for that checkpoint.

Reference motions
-----------------

Six 50 Hz clips are available under the asset-server prefix
``https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2/Isaac/IsaacLab/Motions/G1/v1/``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Motion
     - File
   * - Spin kick (default)
     - ``g1_spinkick.npz``
   * - Walk
     - ``g1_walk.npz``
   * - Run
     - ``g1_run.npz``
   * - Cartwheel
     - ``g1_cartwheel.npz``
   * - Speed vault
     - ``g1_speed_vault.npz``
   * - CR7 celebration
     - ``g1_cr7_grounded.npz``

Each NPZ includes joint and body names; no separate metadata file is needed for these clips.
See the `data format and provenance <https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2/Isaac/IsaacLab/Motions/G1/v1/README.txt>`__
and `SHA-256 checksums <https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2/Isaac/IsaacLab/Motions/G1/v1/SHA256SUMS>`__.
Custom motion files must match the policy rate and provide the robot's joint and tracked-body names;
the :class:`~isaaclab.utils.datasets.MotionLoader` documents the required arrays and quaternion convention.

Train
-----

Train the default spin-kick clip:

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl --task Isaac-Motion-Tracking-G1 --num_envs 4096

The default budget is 30,000 PPO iterations, with 24 control steps per environment per iteration
and a checkpoint every 500 iterations. Checkpoints and the saved environment/agent configurations
are written under ``logs/rsl_rl/g1_motion_tracking/<run>/``.
For a short setup check, replace ``--num_envs 4096`` with ``--num_envs 16 --max_iterations 2``.
This checks the training pipeline and does not produce a trained policy.

To train another clip, set ``env.commands.motion.motion_file`` to its full URL or a local NPZ path.
Use a separate experiment name to keep its checkpoints together:

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl --task Isaac-Motion-Tracking-G1 \
       --experiment_name g1_motion_tracking_walk \
       env.commands.motion.motion_file=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2/Isaac/IsaacLab/Motions/G1/v1/g1_walk.npz

The actor uses reference joint positions/velocities, relative torso orientation, angular velocity,
joint feedback, and previous actions. The critic additionally receives privileged position and
linear-velocity observations. Adaptive sampling increases the probability of starting near failed
parts of the reference. Clip completion truncates the episode through the normal reset path.

For reward weights, termination thresholds, and domain randomization, see
``source/isaaclab_tasks/isaaclab_tasks/core/motion_tracking/motion_tracking_env_cfg.py``.
The G1 asset, motor gains, and tracked links are configured in ``config/g1/env_cfg.py`` in that directory.

Play and record
---------------

The pretrained example tracks CR7. Once the checkpoint is published, play it with the Newton GL
viewer and explicitly select its reference clip:

.. code-block:: bash

   uv run isaaclab play --rl_library rsl_rl --task Isaac-Motion-Tracking-G1 \
       --num_envs 1 --checkpoint pretrained --viz newton_gl \
       env.commands.motion.motion_file=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2/Isaac/IsaacLab/Motions/G1/v1/g1_cr7_grounded.npz

For a local checkpoint, replace ``pretrained`` with its path and set
``env.commands.motion.motion_file`` to the same clip used during training.
The checkpoint path alone does not select its reference motion.
Playback disables observation noise, but retains reference-phase sampling, reset perturbations,
and interval pushes. Episodes restart after a tracking failure or clip completion.

Record 500 control steps (10 seconds) without opening a viewer:

.. code-block:: bash

   uv run isaaclab play --rl_library rsl_rl --task Isaac-Motion-Tracking-G1 \
       --num_envs 1 --checkpoint pretrained --video --video_length 500 \
       env.commands.motion.motion_file=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2/Isaac/IsaacLab/Motions/G1/v1/g1_cr7_grounded.npz

The recording is saved under the checkpoint's run directory in ``videos/play/`` and may contain
multiple episodes. See :doc:`Recording Videos </source/how-to/record_video>` for camera and output options.

Deployment
----------

This guide covers the Isaac Lab simulation task. Robot-side deployment is maintained separately
in `g1_deploy <https://github.com/Double7sBurger/g1_deploy#motion-tracking>`__.
Its deployment contract and hardware validation are separate from the simulation workflow described here.
