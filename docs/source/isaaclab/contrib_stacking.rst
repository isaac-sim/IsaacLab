.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

Franka cube stacking
====================

``IsaacContrib-Stack-Cube-Franka-RL`` trains an RSL-RL PPO policy to stack three
4 cm cubes with Newton MJWarp. It uses the standard Franka asset's MuJoCo
physics variant and passive mimic finger. The actor observes robot state and
cube state in reset-assigned role order. Success requires a stable tower and
a released gripper.

Reset curriculum
----------------

A reset event builds a bank of validated full-scene states. The bank includes
table starts and intermediate pick, transport, placement, and release states.
Restoring a row writes the robot state, cube poses, zero velocities, and
matching actuator targets. Held cubes follow the reset pose of the hand.
Cube colors are permuted across the physical roles.

The curriculum records each completed outcome against its previous row before
the reset event selects a replacement. Training draws 35% of starts from the
table and uses the remaining starts for intermediate states, weighted toward
50% recent success. Progress used for sampling and final stack success are
tracked separately. Play mode samples randomized table starts.

To test a specific phase, configure the reset event before creating the
environment:

.. code-block:: python

   from isaaclab_tasks.contrib.stack import mdp
   from isaaclab_tasks.utils import parse_env_cfg

   cfg = parse_env_cfg("IsaacContrib-Stack-Cube-Franka-RL", num_envs=4)
   cfg.curriculum = None
   cfg.events.reset_from_state_buffer.params["fixed_recipe"] = int(mdp.StackResetRecipe.FIRST_PICK)

Training and playback
---------------------

.. code-block:: bash

   uv run --extra rsl-rl isaaclab train --rl_library rsl_rl --task IsaacContrib-Stack-Cube-Franka-RL physics=newton_mjwarp
   uv run --extra rsl-rl isaaclab play --rl_library rsl_rl --task IsaacContrib-Stack-Cube-Franka-RL \
       --checkpoint /path/to/model.pt --num_envs 1 --visualizer newton_gl physics=newton_mjwarp

The state actor uses a fixed 100-input observation order. The flat asset's
MuJoCo physics variant and passive finger change the dynamics relative to
earlier stack runs. Revalidate earlier checkpoints against the current task.

Camera PPO and distillation
---------------------------

``IsaacContrib-Stack-Cube-Franka-RL-Camera`` trains PPO with one fixed
128 × 128 RGB camera and robot proprioception. Its actor uses the camera
and robot signals. The critic also observes the simulator state. Training
varies camera calibration at startup and image appearance between episodes.
The variant uses a fixed role-to-color assignment so the visual student can
identify the intended stacking order.

``IsaacContrib-Stack-Cube-Franka-RL-Camera-Distillation`` trains the same camera
actor using a compatible state PPO checkpoint as its frozen teacher.
Teacher labels use the same action clipping as the student. The teacher
uses the state task's 100-input observation order, so older 109-input
checkpoints require retraining. Add ``--checkpoint`` with the path to the
teacher when starting distillation from the environment browser's base
training command.

.. code-block:: bash

   uv run --extra rsl-rl isaaclab train --rl_library rsl_rl --task IsaacContrib-Stack-Cube-Franka-RL-Camera \
       physics=newton_mjwarp renderer=newton_renderer
   uv run --extra rsl-rl isaaclab train --rl_library rsl_rl --task IsaacContrib-Stack-Cube-Franka-RL-Camera-Distillation \
       --checkpoint /path/to/teacher.pt physics=newton_mjwarp renderer=newton_renderer

The camera image is channel-first RGB scaled by ``1 / 255``. Checkpoint
observation order, image size, and policy distribution must match the
selected task and its agent configuration.
