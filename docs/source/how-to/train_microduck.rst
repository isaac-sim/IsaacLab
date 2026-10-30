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

Symmetry augmentation
---------------------

Select ``--agent rsl_rl_with_symmetry_cfg_entry_point`` to train either walking task with
left/right data augmentation. This doubles each PPO minibatch with reflected observations and
actions. It uses MicroDuck's servo order and joint-axis signs, and reflects the critic's foot
features as well. Mirror loss is disabled. RSL-RL may still log a detached symmetry diagnostic.

.. code-block:: bash

    uv run isaaclab train --task IsaacContrib-Velocity-Flat-MicroDuck \
        --rl_library rsl_rl --agent rsl_rl_with_symmetry_cfg_entry_point

Single-GPU fast-start experiment
--------------------------------

The optional recipe below combines augmentation with a wider yaw-tracking tolerance of 0.5 rad/s,
following `AGILE's G1 reward kernel <https://github.com/nvidia-isaac/WBC-AGILE/tree/main/agile/rl_env/tasks/locomotion/g1>`_. It keeps the original action scale, networks, exploration, and
other rewards. Mirror loss is disabled. Use early checkpoint evaluations to select the training
budget; the recipe does not guarantee convergence across seeds or terrain presets.

.. code-block:: bash

    PYTHONPATH=scripts/benchmarks/microduck uv run isaaclab train \
        --external_callback microduck_fast_recipe.register \
        --task IsaacContrib-Velocity-Flat-MicroDuck-FastStudy --rl_library rsl_rl \
        --num_envs 4096 --seed 42 --max_iterations 501 agent.save_interval=100

Evaluating early checkpoints
----------------------------

Compare saved checkpoints with the same fixed-command benchmark:

.. code-block:: bash

    uv run python scripts/benchmarks/microduck/evaluate_microduck.py \
        --checkpoint /path/to/model_500.pt --output /path/to/evaluation.json

The evaluator uses the default flat robot and reads the action scale from the saved environment
configuration. It does not restore rough terrain, backlash, or other dynamics overrides. Its 12 command
cases cover standing, forward/backward walking, mirrored lateral motion, turns, and combined
translation and turning. It reports velocity MAE and RMSE, falls, surviving sample coverage,
per-robot stance asymmetry, swing durations, single-support fraction, and policy mirror error.
Startup and reset randomization remain enabled. It excludes a robot's samples after its first fall
and reports the resulting coverage explicitly.
Compare physical metrics and wall time as well as rewards and iteration counts.
