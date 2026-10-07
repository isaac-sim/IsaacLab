.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

:orphan:

MicroDuck Walking
=================

``IsaacContrib-Velocity-Flat-MicroDuck`` trains the walking MicroDuck on a plane with Newton MJWarp and
:ref:`BAM servos <actuators-bam>`, porting the recipe from
`microduck_rl <https://github.com/pollen-robotics/microduck_rl>`_.

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

The policy runs at 50 Hz (four 0.005 s physics steps). Its 14 actions are joint-position offsets [rad]
from the standing pose, ordered left leg, neck and head, then right leg, independent of the USD joint order.

The actor reads 61 values: base angular velocity (3), projected gravity (3), joint-position offsets (14),
joint velocities (14), previous actions (14), velocity commands (3), head-pose commands (4), and body-pose
commands (6). The critic reads 76 values, including privileged foot state.

Training randomizes encoder bias, IMU misalignment, observation delay, BAM friction, mass, center of mass,
and armature. Playback disables observation noise and pushes.

Rough terrain
-------------

``IsaacContrib-Velocity-Rough-MicroDuck`` uses
`microduck_rl's gentle terrain mix <https://github.com/pollen-robotics/microduck_rl/blob/8d0db74916a4f833d1d9b95d6a1d7f4d13b9d5ec/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py>`_:
flat ground, stairs up to 1.5 cm, random grids up to 1 cm, and gentle slopes in ten difficulty levels, advanced by
distance walked. The actor stays blind to terrain; two downward rays per foot give the critic and the foot-clearance
rewards the ground height.

Backlash
--------

Add ``presets=backlash`` to either task to train the robot with ±1° of gearbox play in series with each servo:

.. code-block:: bash

   uv run --extra rsl-rl isaaclab train --rl_library rsl_rl \
     --task IsaacContrib-Velocity-Flat-MicroDuck --num_envs 4096 presets=backlash

Encoders and head-pose rewards then measure servo plus play angle, only servo joints incur the soft-limit penalty,
and the policy interface is unchanged, so existing walking policies load.
