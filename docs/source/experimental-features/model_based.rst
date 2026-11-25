SuperMimic-style Motion Tracking
========================================

RSL-RL SuperMimic learns a world model from simulation and trains a
policy through predicted motion rollouts. It is based on
`SuperMimic <https://github.com/HybridRobotics/supermimic>`_ and uses Isaac Lab's
motion-tracking tasks and standard train/play commands.

Start with :doc:`/source/how-to/train_model_based` for data downloads, input
formats, training, and playback.

How it works
------------

Training repeats three steps:

1. Run the current policy in simulation and store state transitions in a replay buffer.
2. Fit the world model to short sequences sampled from that buffer.
3. Predict short motion rollouts with the world model and improve the policy by
   backpropagating the tracking return through those predictions. A critic estimates
   the return beyond each rollout.

The world model stays fixed during each policy update and continues learning in
later iterations. Gradients pass through the learned model, not the simulator.
Tracking rewards and termination are computed from predicted states and the reference.

.. list-table:: Model inputs and outputs
   :header-rows: 1
   :widths: 20 50 30

   * - Model
     - Inputs
     - Output
   * - World model
     - Root height, projected gravity, root-frame velocities, joint state,
       actions, and body masses.
     - Root and joint velocity increments and self-collision events;
       poses follow by integration.
   * - Policy (actor)
     - Robot observations or observation history, and the reference joint command.
     - 29 normalized joint-position commands, scaled to joint targets.
   * - Critic
     - Full robot state, reference-motion goal, and simulation context.
     - Expected discounted return as a scalar.

The selected task and preset determine the state representation and observation
history. Playback uses only the policy; the world model and critic are needed for training.

Tasks and configuration
-----------------------

SuperMimic uses ``Isaac-Motion-Tracking-G1``, the same environment as PPO.
Select ``--agent rsl_rl_supermimic_cfg_entry_point presets=state_obs`` to enable
model-based learning. The default agent remains PPO.

The environment supplies commands, reset sampling, events, observations, rewards
and termination thresholds. The differentiable task reads the resolved manager
configurations and evaluates their supported terms on predicted states.
For 29 joints, the world model consumes 68 state features and predicts 35 velocity
increments. Root and joint poses use semi-implicit integration at the task's
50 Hz control rate, with four physics steps per control step.

The agent uses Muon/Adam optimizers, one observation frame and 32-step policy
rollouts. Override individual options on the command line, for example:

* ``agent.algorithm.horizon=16`` for 16-step predicted policy rollouts.
* ``--logger wandb --log_project_name PROJECT`` for W&B logging.

Training outputs
----------------

Outputs are saved under ``logs/rsl_rl/g1_motion_tracking/<timestamp>``:

* ``model_*.pt``: actor, critic, world-model weights and optimizer states.
* TensorBoard events with training losses, rewards and throughput.
* Saved environment and agent configurations under ``params/``.

One iteration advances every environment by one control step. For example,
40,000 iterations with 4096 environments collect 163,840,000 transitions.
``iteration`` in the learner checkpoint reports the separate count of learning cycles.

Multi-GPU training
------------------

Use ``train_multigpu`` with environment and learner batch sizes specified
**per GPU**. For two GPUs and 4096 environments in total:

.. code-block:: bash

   uv run --frozen isaaclab train_multigpu --rl_library rsl_rl --num_gpus 2 -- \
       --task Isaac-Motion-Tracking-G1 \
       --agent rsl_rl_supermimic_cfg_entry_point presets=state_obs --num_envs 2048 \
       env.commands.motion.motion_file=/path/to/motions \
       env.scene.robot.spawn.usd_path=/path/to/g1.usd \
       agent.algorithm.world_model_cfg.batch_size=1024 agent.algorithm.policy_batch_size=2048

Each GPU samples the reference set in its own environments and stores local replay data.
Model updates are synchronized across GPUs; rank zero saves checkpoints and training metrics. The learning
rate is unchanged unless you override it. Playback uses the full reference set.

Sim-to-real
-----------

For G1 hardware deployment, see the
`G1 LEAPP deployment guide <https://isaac-sim.github.io/IsaacLab/main/source/how-to/train_model_based.html#g1-leapp-deployment-adapter>`_.

Sources and licenses
--------------------

* **SuperMimic algorithm and task mathematics:**
  `HybridRobotics/SuperMimic
  <https://github.com/HybridRobotics/supermimic/tree/5efbf114e11f725c31d83e217be987c13c0e4ce2>`_
  (Apache-2.0). License: ``docs/licenses/dependencies/supermimic-official-license.txt``.
* **Isaac Lab integration:**
  `PR #8826 by ooctipus <https://github.com/isaac-sim/IsaacLab/pull/8826>`_,
  commit ``b2de4e7c4702a1f5417c22e00ec71b9e4ba887cb``. Inherited SuperMimic code
  retains the ETH Zurich / Chong Zhang MIT license in
  ``docs/licenses/dependencies/supermimic-license.txt``.
* **Shared task and agent integration:**
  `Lionel Gulich's motion-tracking integration
  <https://github.com/isaac-sim/IsaacLab/pull/9151>`_,
  commit ``a0d94e7b7605a912cc6b9dc7c81f2e0ff252ebda`` (BSD-3-Clause).
* **Coordinate transforms:** Isaac Lab math utilities.
* **Learning, replay and neural networks:** ``rsl-rl-lib-extras`` (imported as ``rsl_rl``);
  license: ``docs/licenses/dependencies/rsl_rl-license.txt``.

Isaac Lab integration code uses BSD-3-Clause. Dataset and robot-asset terms
are provided by their respective distributors.
