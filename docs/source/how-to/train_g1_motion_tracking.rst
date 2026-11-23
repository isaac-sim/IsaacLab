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

.. raw:: html

   <video autoplay loop muted playsinline controls preload="metadata" style="width:100%; display:block;">
     <source src="https://huggingface.co/datasets/nvidia/isaaclab-docs-assets/resolve/main/environments/previews/locomotion/g1_jump_and_turn.mp4" type="video/mp4">
   </video>

Prerequisites
-------------

Follow the :doc:`Isaac Lab installation </source/isaaclab/installation>` for Newton, then install
the RSL-RL and video-recording dependencies from the repository root:

.. code-block:: bash

   uv sync --extra rsl-rl --extra video

The first run downloads and caches the official G1 USD and the selected reference motion.
Internet access is required until those assets are cached.

The :ref:`Command Builder <environment-browser>` also lists ``Isaac-Motion-Tracking-G1``.
Select **Train** or **Play** and **rsl_rl** to generate the base command.

Reference motions
-----------------

Six 50 Hz clips are available in the same asset-server directory:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Motion
     - File
   * - Spin kick
     - ``g1_spinkick.npz``
   * - Walk
     - ``g1_walk.npz``
   * - Run
     - ``g1_run.npz``
   * - Cartwheel
     - ``g1_cartwheel.npz``
   * - Speed vault
     - ``g1_speed_vault.npz``
   * - CR7 celebration (default)
     - ``g1_cr7_grounded.npz``

Train
-----

Train the default CR7 clip:

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl --task Isaac-Motion-Tracking-G1 --num_envs 4096

The default budget is 30,000 PPO iterations, with 24 control steps per environment per iteration
and a checkpoint every 500 iterations. Checkpoints and the saved environment/agent configurations
are written under ``logs/rsl_rl/g1_motion_tracking/<run>/``.

To train another clip, set ``env.commands.motion.motion_file`` to its full URL or a local NPZ path.
Use a separate experiment name to keep its checkpoints together:

.. code-block:: bash

   MOTION_FILE="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2/Isaac/IsaacLab/Motions/G1/v1/g1_walk.npz"
   uv run isaaclab train --rl_library rsl_rl --task Isaac-Motion-Tracking-G1 \
       --experiment_name g1_motion_tracking_walk \
       env.commands.motion.motion_file="${MOTION_FILE}"

The actor uses reference **joint positions/velocities, relative torso orientation, angular velocity,
joint feedback, and previous actions**. The critic additionally receives privileged position and
linear-velocity observations. Adaptive sampling increases the probability of starting near failed
parts of the reference. Clip completion truncates the episode through the normal reset path.

For reward weights, termination thresholds, and domain randomization, see
``source/isaaclab_tasks/isaaclab_tasks/core/motion_tracking/motion_tracking_env_cfg.py``.
The G1 asset, motor gains, and tracked links are configured in ``config/g1/env_cfg.py`` in that directory.

Play and record
---------------

Play the pretrained CR7 policy with the Newton GL viewer:

.. code-block:: bash

   uv run isaaclab play --rl_library rsl_rl --task Isaac-Motion-Tracking-G1 \
       --num_envs 1 --checkpoint pretrained --viz newton_gl

For a local checkpoint, replace ``pretrained`` with its path. For a clip other than CR7, set
``env.commands.motion.motion_file`` to the same clip used during training.
The checkpoint path alone does not select its reference motion.
Playback disables observation noise, but retains reference-phase sampling, reset perturbations,
and interval pushes. Episodes restart after a tracking failure or clip completion.

Record 500 control steps (10 seconds) without opening a viewer:

.. code-block:: bash

   uv run isaaclab play --rl_library rsl_rl --task Isaac-Motion-Tracking-G1 \
       --num_envs 1 --checkpoint pretrained --video --video_length 500

The recording is saved under the checkpoint's run directory in ``videos/play/`` and may contain
multiple episodes. See :doc:`Recording Videos </source/how-to/record_video>` for camera and output options.

Deployment
----------

This guide covers the Isaac Lab simulation task. Robot-side deployment is maintained separately
in `g1_deploy <https://github.com/Double7sBurger/g1_deploy#motion-tracking>`__.
Its deployment contract and hardware validation are separate from the simulation workflow described here.
