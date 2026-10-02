.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

:orphan:

MicroDuck Walking
=================

``IsaacContrib-Velocity-Flat-MicroDuck`` trains the standard walking MicroDuck on a plane with
Newton MJWarp and :ref:`BAM servos <actuators-bam>`. It ports the walking recipe from
`microduck_rl <https://github.com/pollen-robotics/microduck_rl>`_ through the original Isaac Lab
MicroDuck branch. This task uses the regular walking asset; roller and backlash tasks are not registered.

The walking USD is currently loaded from
``source/isaaclab_assets/data/Robots/PollenRobotics/MicroDuck/microduck_walk.usd``.
Place the authored asset there before running. It is kept outside Git while asset-server publication is pending.

Train with RSL-RL:

.. code-block:: bash

   uv run --extra rsl-rl isaaclab train --rl_library rsl_rl \
     --task IsaacContrib-Velocity-Flat-MicroDuck --num_envs 4096

Play an RSL-RL checkpoint:

.. code-block:: bash

   uv run --extra rsl-rl isaaclab play --rl_library rsl_rl \
     --task IsaacContrib-Velocity-Flat-MicroDuck --num_envs 16 \
     --checkpoint /path/to/model.pt --visualizer newton_gl

Policy interface
----------------

The control period is 0.02 s: four physics steps of 0.005 s. The 14 actions are joint-position
offsets [rad] from the standing pose, ordered left leg, neck/head, then right leg. The explicit
joint order in ``flat_env_cfg.py`` is used even when USD joint ordering differs.

The actor reads 61 values in this order: base angular velocity (3), projected gravity (3),
joint-position offsets (14), joint velocities (14), previous actions (14), velocity commands (3),
head-pose commands (4), and body-pose commands (6). The body command remains in the interface
although its reward is disabled. The critic receives 76 values including privileged foot state.

Training retains the previous Lab task's encoder bias, shared IMU misalignment, observation delays,
velocity/head command sampling, rewards, and staged curricula. BAM friction is resampled on every
episode reset through the native drive; battery voltage and sag gain are sampled at construction.
The line-search budget is 50 iterations because the original budget of 20 exhausted the current solver.
Playback disables additive observation noise and interval pushes; encoder bias, IMU misalignment,
latency, and reset randomization remain active.

This is not an exact reproduction of mjlab physics. Foot heights use ankle frames with a sole-height
offset, and the walking USD's self-contact signal covers sole against sole. Its disabled shin and
battery-holder colliders cannot reproduce mjlab's self-collision-only geometry. A compatible policy
layout therefore still requires a rollout check on the current assets and actuators.

Rough terrain
-------------

``IsaacContrib-Velocity-Rough-MicroDuck`` adds the gentle terrain recipe from
`microduck_rl's velocity task <https://github.com/pollen-robotics/microduck_rl/blob/8d0db74916a4f833d1d9b95d6a1d7f4d13b9d5ec/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py>`_.
It uses the same walking USD, BAM settings, rewards, randomization, and PPO configuration as the flat task.
No additional assets or teacher policies are required.

.. code-block:: bash

   uv run --extra rsl-rl isaaclab train --rl_library rsl_rl \
     --task IsaacContrib-Velocity-Rough-MicroDuck --num_envs 4096

The terrain contains 8 m square tiles in ten difficulty levels: 25% flat, 25% stairs with steps
up to 1.5 cm high, 30% random grids with height offsets up to 1 cm, and 20% slopes with a rise/run
of 0.03--0.10. The distance walked controls progression through terrain levels. Initial resets sample
levels 0--5. Playback uses a smaller, randomly generated map without terrain progression.

The actor remains blind to terrain, with the same 61 observations and 14 actions. Two downward rays
per foot, 4 cm ahead and behind its frame, measure the closest ground for the clearance and swing-height
rewards and the critic's foot-height observations. The critic still receives 76 values. Rays query only
the shared terrain. If both rays miss, clearance falls back to the tile origin and is bounded by the
sensor's 1 m range.

Lab generates a triangle mesh instead of mjlab's box geometry, so contact behavior is not identical.
The terrain importer authors MuJoCo contact parameters ``solref=(0.04, 1.0)`` and
``solimp=(0.85, 0.95, 0.001, 0.5, 2.0)`` before Newton parses the scene. The task allows 200 contacts
and 1024 constraint rows per environment and uses 100 solver iterations with 50 line-search iterations.
This keeps the current flat training run's solver budget rather than mjlab's 30-iteration rough setting.
