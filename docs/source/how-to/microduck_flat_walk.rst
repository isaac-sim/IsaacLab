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

Walking with fall recovery
--------------------------

``IsaacContrib-Recovery-Velocity-Flat-Backlash-MicroDuck`` trains one PPO policy from scratch
to walk and get up. It uses the local ``microduck_allcollisions_backlash.usd`` asset, including
body and head contacts. No teacher or walking checkpoint is required.

.. code-block:: bash

   uv run --extra rsl-rl isaaclab train --rl_library rsl_rl \
     --task IsaacContrib-Recovery-Velocity-Flat-Backlash-MicroDuck \
     --num_envs 16384 --seed 42

Level zero starts upright and terminates on a fall. The curriculum collects one completed
episode per environment from the current level, retaining early failures until the later
survivors finish. After a full cohort and at least 4096 episodes, it checks fall-free survival
(at least 80%) and mean velocity tracking score (at least 0.65) on upright starts. Both must
pass to introduce recovery starts. Tracking uses the lower of the linear and angular scores
and requires at least half the commanded motion for nontrivial translation/turn commands.
Later levels additionally require at least 60% success on randomized starts. Success requires
stable standing, two continuous seconds of command tracking, and completing the episode
without an unresolved fall. Failed gates retain the current difficulty.

Levels 1--5 increase the randomized-start fraction from 10% to 50%, roll/pitch range from
±36° to ±180°, and servo offsets from ±0.2 to ±1.0 rad, clamped to soft joint limits.
Yaw spans ±180° at all levels. Randomized starts are released from a root height of 0.4 m;
these are controlled drops, not a bank of settled ground poses. Passive play hinges start
centered. At least half the starts stay upright throughout the curriculum.

After level zero, **all** falls get the same six-second recovery window, including spontaneous
falls during walking. A fall is root height below 5.5 cm or tilt beyond 70°. Clearing the
window requires 0.5 s continuously above 9.5 cm, within 30° of upright, and below 2 rad/s
angular speed. Brief threshold crossings do not restart the window. Eight seconds of total
recovery time also ends an episode, preventing repeated short stand-ups from extending it
indefinitely. These failures are terminations; the usual 20 s horizon is a time-limit truncation.
Each episode keeps the termination budget it had at reset, even if the curriculum advances.

Walking rewards fade smoothly with reduced height and uprightness, freeing the legs and head
for recovery. Signed orientation and height provide recovery shaping; a spontaneous fall has a
one-off cost. Intentional randomized starts incur no initial fall cost. The head-bias integrator
is disabled because head support on the ground is useful for getting up. The actor keeps 61
observations and 14 actions; the critic has 80 observations, including recovery state, remaining
attempt/episode budgets, and root height. PPO uses discount 0.995 and 16 minibatches.

Monitor ``Curriculum/recovery/level``, ``survival``, ``tracking``, and ``recovery_success`` in
TensorBoard. These thresholds are initial training settings, not established performance claims.
Evaluate upright walking and randomized recovery separately before raising difficulty.
Playback fixes the curriculum at level five. For a specific level, or to resume the curriculum
level recorded in a training log, override ``env.recovery.initial_level``; the adaptive counters
are environment state and are not included in PPO checkpoints. Fixed-level evaluation uses
``env.recovery.curriculum_enabled=False``.
