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
