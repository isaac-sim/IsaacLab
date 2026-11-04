:orphan:

.. _microduck-walking-training:

Training MicroDuck
==================

``IsaacContrib-Velocity-Flat-MicroDuck`` and ``IsaacContrib-Velocity-Rough-MicroDuck`` train
MicroDuck to walk with Newton MJWarp and BAM servos. Add ``presets=backlash`` to select the robot
with a passive play hinge in series with each servo.

Walking recipe
--------------

Both tasks use the AGILE-inspired combined recipe by default. It tracks horizontal root-link
velocity in the heading frame with a tolerance of 0.15 m/s and world-frame yaw rate with a tolerance
of 0.5 rad/s on rough terrain and 0.35 rad/s on flat ground, each with reward weight 4.0. The nominal
joint-posture reward applies only to standing commands. The actor and critic have 61 and 76 inputs,
respectively, and use normalized MLPs with
hidden layers ``[512, 256, 128]`` and ELU activation.

Train the default rough recipe with symmetry augmentation and backlash:

.. code-block:: bash

    uv run isaaclab train --task IsaacContrib-Velocity-Rough-MicroDuck \
        --rl_library rsl_rl presets=backlash

Use ``IsaacContrib-Velocity-Flat-MicroDuck`` for flat ground, or omit ``backlash`` for the regular
robot.

Rough terrain advances after an episode survives and tracks its commands, with at least 1 m of
commanded and actual walking and 5 s of translation commands. It accumulates path length, so turns
and changes of direction retain progress. Mostly stationary episodes hold their difficulty;
failures and poor tracking lower it.

Rough training defaults
-----------------------

``IsaacContrib-Velocity-Rough-MicroDuck`` uses the locomotion-first recipe with 16,384 environments
and left/right data augmentation enabled by default. Mirror loss is
disabled. It introduces the action-rate penalty stages at iterations 750, 1,000, 1,250, 1,500
and 1,750, and delays the standing-fraction, head-pose and center-of-mass curricula by 500
iterations. It uses a 0.5 rad/s yaw-tracking reward width. Stage values, final command ranges,
terrain promotion and PPO settings are unchanged.
The runner's default training budget is 50,000 iterations on one GPU.

The flat task retains 4,096 environments, its original curricula,
and PPO without symmetry augmentation by default. Its action-rate penalty increases from -0.1
to -1.0 between training iterations 3,000 and 7,000.

This batch size is part of the intended recipe; smaller batches are not established quality-equivalent
replacements.

Distributed training
--------------------

MicroDuck scales iteration and curriculum counts with global ``WORLD_SIZE`` (default: 1).
Keeping the task's default environment count per GPU (16,384 for rough terrain, 4,096 for flat
ground) preserves the single-GPU sample budget, rounded up to full rollouts. This also handles
nodes with unequal worker counts.

.. code-block:: bash

    uv run isaaclab train_multigpu --num_gpus 8 \
        --task IsaacContrib-Velocity-Rough-MicroDuck --rl_library rsl_rl \
        --num_envs 16384 --seed 42 presets=backlash

For N ranks, training uses ``ceil(50000 / N)`` iterations and a ``ceil(250 / N)`` checkpoint interval.
All six fixed-step curriculum boundaries are divided by N, rounding up. Explicit iteration, checkpoint
and curriculum overrides take precedence. To override N, supply both ``env.ngpus`` (GPUs per node) and
``env.nnodes``; their product affects recipe scaling only. Both default to ``None``; incomplete pairs raise an error.

Select actual workers with ``--num_gpus`` (per node) and ``--nnodes``; see :ref:`multi-node-training`.
Single-GPU defaults, the 24-step rollout and PPO settings are unchanged. Terrain progression remains
episode-based. Larger optimizer batches can change convergence despite the matching sample budget.

Previous walking recipe
-----------------------

To retain the previous recipe in a custom task config, use ``mdp.track_linear_velocity`` with weight
2.0 and ``std=math.sqrt(0.1)``, and ``mdp.track_angular_velocity`` with weight 2.0 and
``std=math.sqrt(0.5)``. Set ``rewards.pose.params["standing_only"] = False`` and restore the action-rate
weight stages to ``(500, -0.2)``, ``(750, -0.4)``, ``(1000, -0.6)``, ``(1250, -0.8)``, and ``(1500, -1.0)``.

Symmetry augmentation
---------------------

The rough task enables left/right data augmentation by default. Select
``--agent rsl_rl_with_symmetry_cfg_entry_point`` to enable it for the flat task as well.
This doubles each PPO minibatch with reflected observations and
actions. It uses MicroDuck's servo order and joint-axis signs, and reflects the critic's foot
features as well. Mirror loss is disabled. RSL-RL may still log a detached symmetry diagnostic.

.. code-block:: bash

    uv run isaaclab train --task IsaacContrib-Velocity-Flat-MicroDuck \
        --rl_library rsl_rl --agent rsl_rl_with_symmetry_cfg_entry_point
