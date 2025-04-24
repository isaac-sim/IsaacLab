Debugging and Training Guide
============================

In this tutorial, we'll guide developers working with Isaac Lab to understand the
impact of various parameters on training time, GPU utilization, and memory usage.
This is especially helpful for addressing Out of Memory (OOM) errors that commonly
occur during reinforcement learning (RL) training. We will touch on common errors seen
during RL training in Isaac Lab and provide some guidance on troubleshooting steps.


Training with Parallel Environments
-----------------------------------

The key RL paradigm of Isaac Lab is to train with many environments in parallel.
Here, we define an environment as an instance of a robot or multiple robots interacting with other robots or objects in simulation.
By creating multiple environments in parallel, we generate multiple copies of the environment such that the robots in each environment can explore the world independently of other environments.
The number of environments thus becomes an important hyperparameter for training.
In general, the more environments we have running in parallel,
the more data we can collect during rollout, which in turn, provides more data
for RL training and allows for faster training since the RL agent can learn from parallel experiences.

However, the number of environments can also be bounded by other factors.
Memory can often be a hard constraint on the number of environments we can run in parallel.
When more environments are added to the world, the simulation also requires more memory to represent and simulate each object in the scene.
The number of environments we can simulate in parallel thus often depend on the amount of memory resources available on the machine.
In addition, different forms of simulation can also consume various amounts of memory.
For example, objects with high fidelity visual and collision meshes will consume more memory than simple primitive shapes.
Deformable simulation will also likely require more memory to simulate than rigid bodies.

Training with rendering often consumes much higher memory than running with only physics simulation. This is especially true when rendering at relatively large resolutions. Additionally, when training RL policies with image observations, we often also require more memory to hold the rollout trajectories of image buffers and larger networks for the policies. Both of these components will also impact the amount of memory available for the simulation.

To reduce memory consumption, one method is to simplify collision meshes of the assets where possible to keep only bare minimum collision shapes required for correct simulation of contacts.
Additionally, we recommend only running with the viewport when debugging with a small number of environments.
When training with larger number of environments in parallel, it is recommended to run in headless mode to avoid any rendering overhead.
If the RL pipeline requires rendering in the loop, make sure to reduce the number of environments, taking into consideration for the dimensions of the image buffers and the size of the policy networks. When hitting out of memory errors, the simplest solution may be to reduce the number of environments.


Hyperparameter Tuning
---------------------

Although in many cases, simulating more environments in parallel can yield faster training and better results, there are also cases where diminishing returns are observed when the number of environments reaches certain thresholds.
This threshold will vary depending on the complexity of the environment, task, policy setup, and RL algorithm.
When more environments are simulated in parallel, each simulation step requires more time to simulate, which will impact the overall training time.
When the number of environments is small, this increase in per-step simulation time is often insignificant compared to the increase in training performance from more experiences collected.
However, when the number of environments reaches a point, the benefits from having even more experiences for the RL algorithm may start to saturate, and the amount of increased simulation time can outweigh the benefits in training performance.

In contrast to diminishing returns on number of environments that are too large, training with low number of environments can also be challenging.
This is often due to the RL policies not getting enough experiences to learn from.
To address this issue, it may be helpful to increase the batch size or the horizon length to accommodate for the smaller amount of data collected from lower number of parallel environments.
When the number of environments is constrained by available resources, running with parallel GPUs or training across multiple nodes can also help alleviate issues due to limited rollouts.


Debugging NaNs during Training
------------------------------

One common error seen during RL training is the appearance of NaNs in the observation buffers, which often get propagated into the policy networks and cause crashes in the downstream training pipeline.
In most cases, the appearance of NaNs occur when the simulation becomes unstable.
This could be due to drastic actions being applied to the robots that exceed the limits of the simulation, or resets of the assets into invalid states.
Some helpful tips to reduce the occurrence of NaNs include proper tuning of the physics parameters for the assets to ensure that joint, velocity, and force limits are within reasonable ranges and the gains are correctly tuned for the robot.
It is also a good idea to check that actions applied to the robots are reasonable and will not impose large forces or impulses on the objects.
Reducing the timestep of the physics simulation can also help improve accuracy and stability of the simulation, in addition to increasing the solver iterations.


Understanding Training Outputs
------------------------------

Each RL library produces its own output data during training.
Some libraries are more verbose and generate logs that contain more detailed information on the training process, while others are more compact.
In this section, we will explain the common outputs from the RL libraries.


RL-Games
^^^^^^^^

For each iteration, RL-Games prints statistics of the data collection, inference, and training performance.

.. code:: bash

  fps step: 112918 fps step and policy inference: 104337 fps total: 78179 epoch: 1/150 frames: 0

``fps step`` refers to the environment step FPS, which includes the applying actions, computing observations, rewards, dones, and resets, as well as stepping simulation.

``step and policy inference`` measure everything in ``fps step`` along with the time it takes for the policy inference to compute the actions.

``fps total`` measure the above and the time it takes for the training iteration.

At specified intervals, it will also log the current best reward and the path of the intermmediate checkpoints saved to file.

.. code:: bash

  => saving checkpoint 'IsaacLab/logs/rl_games/cartpole_direct/2024-12-28_20-23-06/nn/last_cartpole_direct_ep_150_rew_294.18793.pth'
  saving next best rewards:  [294.18793]


RSL RL
^^^^^^

For each iteration, RSL RL provides the following output:

.. code:: bash

                          Learning iteration 0/150

                       Computation: 50355 steps/s (collection: 1.106s, learning 0.195s)
               Value function loss: 22.0539
                    Surrogate loss: -0.0086
             Mean action noise std: 1.00
                       Mean reward: -5.49
               Mean episode length: 15.79
  --------------------------------------------------------------------------------
                   Total timesteps: 65536
                    Iteration time: 1.30s
                        Total time: 1.30s
                               ETA: 195.2s


This output encapsulates the total FPS for data collection, inference, and learning, along with the per-step breakdown for collection and learning time per step.
In addition, statistics for the training losses are provided, along with the current average reward and episode length.

In the bottom section, it logs the total number of steps completed so far, the total ieration time for the current ieration, the total overall training time, and the estimated training time to complete the full number of iterations.


SKRL
^^^^

SKRL provides a very simplistic output showing the training progress as a percentage of the total number of timesteps (divided by the number of environments). It also includes the total elapsed time so far and the estimated time to complete training.

.. code:: bash

    0%|                                          | 2/4800 [00:00<10:02,  7.96it/s]


Stable-Baselines3
^^^^^^^^^^^^^^^^^

Stable-Baselines3 provides a detailed output, outlining the rollout statistics, timing, and policy data.

.. code:: bash

  ------------------------------------------
  | rollout/                |              |
  |    ep_len_mean          | 30.8         |
  |    ep_rew_mean          | 2.87         |
  | time/                   |              |
  |    fps                  | 8824         |
  |    iterations           | 2            |
  |    time_elapsed         | 14           |
  |    total_timesteps      | 131072       |
  | train/                  |              |
  |    approx_kl            | 0.0079056695 |
  |    clip_fraction        | 0.0842       |
  |    clip_range           | 0.2          |
  |    entropy_loss         | -1.42        |
  |    explained_variance   | 0.0344       |
  |    learning_rate        | 0.0003       |
  |    loss                 | 10.4         |
  |    n_updates            | 20           |
  |    policy_gradient_loss | -0.0119      |
  |    std                  | 1            |
  |    value_loss           | 17           |
  ------------------------------------------

Under the ``rollout/`` section, average episode length and reward are logged for the iteration. Under ``time/``, data for the total FPS, number of iterations, total time elapsed, and the total number of timesteps are provided. Finally, under ``train/``, statistics of the training parameters are logged, such as KL, losses, learning rates, and more.
