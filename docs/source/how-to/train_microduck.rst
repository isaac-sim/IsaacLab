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

With 4,096 environments per GPU, add ``presets=multi_gpu`` alongside any robot preset. The preset
defaults ``env.num_gpus`` to ``WORLD_SIZE``, the total rank count set by the distributed launcher.

.. code-block:: bash

    uv run isaaclab train_multigpu --num_gpus 8 \
        --task IsaacContrib-Velocity-Rough-MicroDuck --rl_library rsl_rl \
        --num_envs 4096 --seed 42 presets=backlash,multi_gpu

``--num_gpus`` selects the number of workers per node. For example, two nodes with
``--nnodes 2 --num_gpus 8`` use 16 ranks in total. See :ref:`multi-node-training` for the rendezvous
options. To scale the recipe using a different count, pass a Hydra override such as ``env.num_gpus=4``;
this changes the training budget and schedules, independently of the number of launched workers.

For a scaling count N, the preset selects 50,000 / N iterations and saves every 250 / N iterations,
rounding both up. Eight GPUs use 6,250 iterations and save every 32 iterations; 16 GPUs use 3,125
iterations and save every 16 iterations. All six fixed-step curricula divide their boundaries by N,
rounding up to a whole step. These values are computed after Hydra applies overrides, so an explicit
``agent.max_iterations``, ``agent.save_interval``, or curriculum-stage override still takes precedence.

With automatic rank detection and 4,096 environments per GPU, the total sample budget matches the
single-GPU run, rounded up to a full distributed rollout. Larger optimizer batches can still change
convergence. Terrain progression remains episode-based. Without ``multi_gpu``, the original iteration
budget and fixed-step curricula are retained.

Previous walking recipe
-----------------------

To retain the previous recipe in a custom task config, use ``mdp.track_linear_velocity`` with weight
2.0 and ``std=math.sqrt(0.1)``, and ``mdp.track_angular_velocity`` with weight 2.0 and
``std=math.sqrt(0.5)``. Set ``rewards.pose.params["standing_only"] = False`` and restore the action-rate
weight stages to ``(500, -0.2)``, ``(750, -0.4)``, ``(1000, -0.6)``, ``(1250, -0.8)``, and ``(1500, -1.0)``.
