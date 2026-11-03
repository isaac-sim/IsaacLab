isaaclab\_assets.robots
=======================

.. automodule:: isaaclab_assets.robots

Unitree H2
----------

``H2_CFG`` uses the official 31-DOF USD model from
`Unitree's asset repository <https://github.com/unitreerobotics/unitree_model/tree/b6a8942b0803b6c137e58cef12beb4b03e4a2fa7/H2>`__.
The asset revision is pinned, and its USD layers are downloaded on first use. Torque and velocity
limits come from the asset; the PD gains and armature are starting values for locomotion training.

The flat and rough velocity tasks train all 31 joints from scratch with RSL-RL PPO. The rough task
includes terrain height observations and a terrain curriculum; the flat task uses a ground plane and
proprioceptive observations. Start training with:

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Flat-H2 --num_envs 4096 physics=newton_mjwarp
   uv run isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Rough-H2 --num_envs 4096 physics=newton_mjwarp

The default training lengths are 6000 iterations for flat terrain and 8000 for rough terrain.
Neither task enables policy symmetry augmentation or mirror loss.
To use a local copy of the same asset, append ``env.scene.robot.spawn.usd_path=/path/to/H2/H2_dae.usd``;
keep the accompanying ``configuration`` directory beside the root USD file.
