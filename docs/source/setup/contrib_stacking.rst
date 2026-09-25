.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

Contributed cube-stacking tasks
===============================

These Newton MJWarp tasks stack three cubes from randomized table starts and
reset-authored intermediate states. The Franka and KUKA-Allegro state policies
use joint and object observations; the Franka camera actor uses a fixed RGB
camera and robot proprioception. The camera task keeps simulator object state
in its critic or teacher, never in the deployed actor.

.. list-table:: Registered tasks
   :header-rows: 1

   * - Task ID
     - Policy
   * - ``IsaacContrib-Stack-Cube-Franka-RL``
     - Franka state PPO
   * - ``IsaacContrib-Stack-Cube-KukaAllegro-RL``
     - KUKA-Allegro state PPO
   * - ``IsaacContrib-Stack-Cube-Franka-RL-Camera``
     - Franka RGB PPO
   * - ``IsaacContrib-Stack-Cube-Franka-RL-Camera-Distillation``
     - Franka RGB student and privileged state teacher

Pretrained checkpoints for the Franka state, KUKA-Allegro state, and Franka
camera-student tasks are available in the Isaac-dev Nucleus pretrained-checkpoints
collection.

Train the state task, then play a local checkpoint:

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl --task IsaacContrib-Stack-Cube-Franka-RL
   uv run isaaclab play --rl_library rsl_rl --task IsaacContrib-Stack-Cube-Franka-RL \
       --checkpoint /path/to/model.pt --num_envs 1 --viz newton

For RGB distillation, pass a compatible Franka state teacher checkpoint. The
student camera needs an explicit renderer; ``isaacsim_rtx`` is the deployment
rendering choice used by this task:

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl \
       --task IsaacContrib-Stack-Cube-Franka-RL-Camera-Distillation \
       --checkpoint /path/to/teacher.pt renderer=isaacsim_rtx

The camera teacher has 109 inputs (including commanded joint targets), while
the published Franka state PPO actor has 100. Train or adapt a teacher for the
109-input contract before starting distillation; do not pass the state PPO
checkpoint directly to this command.

Play uses randomized table starts with training curricula disabled. Use
``--video --video_length 300`` with ``--viz newton`` to record six seconds
at the 50 Hz policy rate, and check the success termination before sharing a
clip. Checkpoint observation order, camera resolution,
distribution type, and renderer must match the task configuration; a checkpoint
that cannot be loaded is not interchangeable with another stacking variant.

The fixed-camera student is a simulation deployment example. Real-robot use
still requires camera calibration, a matching controller interface, safety
limits, and independent real-world validation.

Previews
--------

These Franka and KUKA-Allegro clips were recorded with Newton physics and the
Newton visualizer from randomized table starts. The task success termination
fired at steps 170 and 213, respectively; each clip ends on the resulting stack.
The tabletop is dark, and the shared ground uses the standard checker visual;
the invisible contact surface is not rendered. The camera student is not shown
because a successful local playback has not yet been reproduced.

.. raw:: html

   <video controls muted loop playsinline width="48%" preload="metadata" aria-label="Franka state-policy stacking preview">
     <source src="../../_static/tasks/previews/stack-franka-newton-mjwarp-rsl-rl.mp4" type="video/mp4">
   </video>
   <video controls muted loop playsinline width="48%" preload="metadata" aria-label="KUKA-Allegro state-policy stacking preview">
     <source src="../../_static/tasks/previews/stack-kuka-allegro-newton-mjwarp-rsl-rl.mp4" type="video/mp4">
   </video>
