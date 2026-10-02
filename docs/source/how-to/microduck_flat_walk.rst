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
MicroDuck branch. The flat and rough tasks each support the regular and backlash walking assets.

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

Backlash variants
-----------------

Use ``IsaacContrib-Velocity-Flat-Backlash-MicroDuck`` or
``IsaacContrib-Velocity-Rough-Backlash-MicroDuck`` to train the corresponding task with gearbox play:

.. code-block:: bash

   uv run --extra rsl-rl isaaclab train --rl_library rsl_rl \
     --task IsaacContrib-Velocity-Flat-Backlash-MicroDuck --num_envs 4096

   uv run --extra rsl-rl isaaclab train --rl_library rsl_rl \
     --task IsaacContrib-Velocity-Rough-Backlash-MicroDuck --num_envs 4096

Both load ``source/isaaclab_assets/data/Robots/PollenRobotics/MicroDuck/microduck_walk_backlash.usd``.
This local asset adds a passive hinge with ±1° of play to each servo, for 28 joints and 14 actions.
The actor and critic retain their 61- and 76-value layouts. Joint observations and head-tracking
rewards measure the output-side encoder: servo angle or velocity plus the corresponding play hinge.
Encoder bias is applied once per servo, and the actor's existing velocity observation delay is retained.
BAM firmware uses output-side position feedback while motor back-EMF remains motor-side.

Only servo joints incur the soft-limit penalty, since passive hinges normally touch their stops.
Reset centers the play hinges; the existing armature randomization includes them, as in the reference.
Both backlash tasks use 100 solver iterations; the flat task's 10-iteration budget was exhausted
at 4096 environments with play hinges. The variants inherit their respective terrain, reward, and PPO settings, with separate
``microduck_velocity_flat_backlash`` and ``microduck_velocity_rough_backlash`` experiment directories.
The observation layout permits loading existing walking policies, but the changed dynamics still
require rollout validation.

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
