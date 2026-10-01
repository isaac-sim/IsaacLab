isaaclab\_assets.robots
=======================

.. automodule:: isaaclab_assets.robots

Franka collision scope
----------------------

The standard Franka core tasks use full arm and gripper collisions. The
``Isaac-Reach-Franka-Minimal`` example uses ``FRANKA_MINIMAL_CFG``: the same USD, authored
inertials, controls, and Reach MDP with only hand and fingertip colliders. Use this configuration
when arm contacts are not required. Removing unused colliders can reduce simulation work;
measure the gain for your backend and environment count, and requalify policies when changing scope.

Compare the two tasks with the same physics backend, environment count, seed and explicit checkpoint.
The minimal task does not have a separate published checkpoint. For example, pass the public Reach
checkpoint explicitly:

.. code-block:: bash

   uv run isaaclab play --rl_library rsl_rl --task Isaac-Reach-Franka-Minimal \
      --checkpoint https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Reach-Franka_newtonmjwarp_none_rsl_rl.pt \
      --num_envs 8192 --seed 42 --viz none physics=newton_mjwarp

Repeat with ``--task Isaac-Reach-Franka`` for the full-collision comparison. Keep backend selectors
independent of collision scope; neither task exposes an ``arm_collisions`` preset.
