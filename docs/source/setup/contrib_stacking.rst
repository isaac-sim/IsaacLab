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

Learning from resets
--------------------

The tasks use Isaac Lab's normal manager lifecycle, not a separate replay
environment. The reset event builds and validates a bank once, then restores
robot joints, cube poses, zero velocities, and matching position targets on
each episode reset. Intermediate states span finger closure, lift, transport,
placement, and release; held cubes follow the hand's forward kinematics rather
than an independently interpolated cube trajectory. Cube-color permutations
augment these physical states without changing the task.

The curriculum records each completed episode before the reset event selects
the next state. It mixes 35% randomized table starts with 65% intermediate
states weighted toward the success monitor's target rate. Learning-progress
success updates that sampler; full-stack success separately requires a stable,
released tower. Play mode disables the curriculum and uses only table starts.

To inspect a phase without adaptive sampling, configure the existing reset event
before creating the environment; no custom reset loop is needed:

.. code-block:: python

   from isaaclab_tasks.contrib.stack import mdp
   from isaaclab_tasks.utils import parse_env_cfg

   cfg = parse_env_cfg("IsaacContrib-Stack-Cube-Franka-RL", num_envs=4)
   cfg.curriculum = None
   cfg.events.reset_from_state_buffer.params["fixed_recipe"] = int(mdp.StackResetRecipe.FIRST_PICK)

Other recipes cover transport, placement, release, and table starts. Leave
``fixed_recipe=None`` and retain the curriculum for normal training. These are
task-specific reset banks; an offline grasp generator does not run during resets.

The Franka task uses ``FRANKA_PANDA_CFG`` with the ``mujoco`` physics payload
and the asset's authored finger mimic: the policy commands the leading finger,
while the follower remains passive. Task-specific arm impedance gains and
gravity compensation are retained.

.. note::

   The flat asset's MuJoCo payload and passive-finger control change the
   dynamics relative to older stacking runs. Matching observation dimensions
   alone does not make a Franka checkpoint compatible: retrain or revalidate
   older state and camera policies before using them with this configuration.

Training and playback
---------------------

Train the state task, then play a local checkpoint:

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl --task IsaacContrib-Stack-Cube-Franka-RL
   uv run isaaclab play --rl_library rsl_rl --task IsaacContrib-Stack-Cube-Franka-RL \
       --checkpoint /path/to/model.pt --num_envs 1 --visualizer newton_gl

For RGB distillation, pass a compatible Franka state teacher checkpoint. The
student camera needs an explicit renderer; ``isaacsim_rtx`` is the deployment
rendering choice used by this task:

.. code-block:: bash

   OMNI_KIT_ACCEPT_EULA=Y ACCEPT_EULA=Y uv run --extra isaacsim isaaclab train --rl_library rsl_rl \
       --task IsaacContrib-Stack-Cube-Franka-RL-Camera-Distillation \
       --checkpoint /path/to/teacher.pt renderer=isaacsim_rtx

The teacher reuses the state PPO actor's 100-input observation order and model
configuration, so a state-task checkpoint trained on the current dynamics can
be loaded directly. Older 109-input camera teachers are incompatible with this
shared interface; retrain them before resuming distillation.

Play uses randomized table starts with training curricula disabled. Use
``--video --video_length 300`` with ``--visualizer newton_gl`` to record six seconds
at the 50 Hz policy rate, and check the success termination before sharing a
clip. Checkpoint observation order, camera resolution,
distribution type, and renderer must match the task configuration; a checkpoint
that cannot be loaded is not interchangeable with another stacking variant.

The fixed-camera student is a simulation deployment example. Real-robot use
still requires camera calibration, a matching controller interface, safety
limits, and independent real-world validation.

Previews
--------

These reference Franka and KUKA-Allegro clips were recorded with Newton physics and the
Newton visualizer from randomized table starts. The task success termination
fired at steps 170 and 213, respectively; each clip ends on the resulting stack.
The tabletop is dark, and the shared ground uses the standard checker visual;
the invisible contact surface is not rendered. The camera student is not shown
because a successful local playback has not yet been reproduced.
The Franka clip predates the switch to the standard flat-asset configuration
and is not validation of an older checkpoint on the current dynamics.

.. raw:: html

   <video controls muted loop playsinline width="48%" preload="metadata" aria-label="Franka state-policy stacking preview">
     <source src="https://media.githubusercontent.com/media/maxkra15/IsaacLab/226ead1683a31094e263116d6cb82d8225c4ddf5/docs/source/_static/tasks/previews/stack-franka-newton-mjwarp-rsl-rl.mp4" type="video/mp4">
   </video>
   <video controls muted loop playsinline width="48%" preload="metadata" aria-label="KUKA-Allegro state-policy stacking preview">
     <source src="https://media.githubusercontent.com/media/maxkra15/IsaacLab/226ead1683a31094e263116d6cb82d8225c4ddf5/docs/source/_static/tasks/previews/stack-kuka-allegro-newton-mjwarp-rsl-rl.mp4" type="video/mp4">
   </video>
