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
of 0.35 rad/s, each with reward weight 4.0. The nominal joint-posture reward applies only to standing
commands, and the action-rate penalty increases from -0.1 to -1.0 between training iterations 3,000
and 7,000. The actor and critic have 61 and 76 inputs, respectively, and use normalized MLPs with
hidden layers ``[512, 256, 128]`` and ELU activation.

For example, train on rough terrain with backlash:

.. code-block:: bash

    uv run isaaclab train --task IsaacContrib-Velocity-Rough-MicroDuck \
        --rl_library rsl_rl presets=backlash

Use ``IsaacContrib-Velocity-Flat-MicroDuck`` for flat ground, or omit ``backlash`` for the regular
robot.

Rough terrain advances after an episode survives and tracks its commands, with at least 1 m of
commanded and actual walking and 5 s of translation commands. It accumulates path length, so turns
and changes of direction retain progress. Mostly stationary episodes hold their difficulty;
failures and poor tracking lower it.

Distributed training
--------------------

With 4,096 environments per GPU, MicroDuck preserves the single-GPU sample budget, rounded up to full
rollouts, by automatically scaling with global ``WORLD_SIZE`` (default: 1). This also handles nodes
with unequal worker counts.

.. code-block:: bash

    uv run isaaclab train_multigpu --num_gpus 8 \
        --task IsaacContrib-Velocity-Rough-MicroDuck --rl_library rsl_rl \
        --num_envs 4096 --seed 42 presets=backlash

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
