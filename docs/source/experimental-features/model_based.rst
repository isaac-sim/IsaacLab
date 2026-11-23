Model-based motion tracking
===========================

The ``model_based`` backend learns a world model from simulation and trains a
policy through predicted motion rollouts. It is based on
`SuperMimic <https://github.com/HybridRobotics/supermimic>`_ and uses Isaac Lab's
motion-tracking tasks and standard train/play commands.

Start with :doc:`/source/how-to/train_model_based` for data downloads, input
formats, training, and playback.

Tasks and configuration
-----------------------

Two G1 tasks are available:

* ``Isaac-Motion-Tracking-GenCoord-G1``: predicts root and joint motion.
* ``Isaac-Motion-Tracking-MaxCoord-G1``: predicts body motion and recovers joint
  motion with inverse kinematics.

Both take a matching 29-joint G1 USD asset and reference clips. The control rate
is 50 Hz, with four physics steps per control step. Use a checkpoint trained for
the selected task.

Select the training preset with ``--agent``:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Preset
     - Configuration
   * - ``model_based_cfg_entry_point`` (default)
     - Muon/Adam optimizers, one observation frame, 32-step policy rollouts.
   * - ``paper_model_based_cfg_entry_point`` (GenCoord)
     - Adam, ten proprioception frames, cosine learning-rate decay, 320k steps.
   * - ``paper_no_critic_cfg_entry_point`` (GenCoord)
     - The paper preset without terminal critic bootstrapping.

Override individual options on the command line, for example:

* ``agent.actor_obs=proprio agent.history_length=10`` for proprioception history.
* ``agent.policy.horizon=16`` for 16-step predicted policy rollouts.
* ``agent.wandb_project=PROJECT`` for W&B logging.

Training outputs
----------------

``--log_dir`` selects the output directory; the default is
``logs/model_based/<timestamp>``. Training writes:

* ``model_*.pt``: model weights, optimizer states, and task/agent configurations.
* ``metrics.jsonl``: physical iterations, transitions, learning updates, losses,
  rewards, and elapsed time.
* Configuration files and a run manifest.

One iteration advances every environment by one control step. For example,
40,000 iterations with 4096 environments collect 163,840,000 transitions.
``learning_updates`` reports the separate count of learning cycles.

Multi-GPU training
------------------

Use ``train_multigpu`` with environment and learner batch sizes specified
**per GPU**. For two GPUs and 4096 environments in total:

.. code-block:: bash

   uv run --frozen isaaclab train_multigpu --rl_library model_based --num_gpus 2 -- \
       --task Isaac-Motion-Tracking-GenCoord-G1 \
       --agent paper_model_based_cfg_entry_point --num_envs 2048 \
       --motion_dir /path/to/motions \
       --asset_path /path/to/g1.usd \
       agent.wm.batch_size=1024 agent.policy.batch_size=2048

Reference clips are split across GPUs; provide at least one clip per GPU.
Rank zero saves checkpoints, and each rank writes its metrics. The learning
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
* **Quaternion helpers:**
  `mjlab, revision 8ee51fbc
  <https://github.com/mujocolab/mjlab/blob/8ee51fbcf806a7419189f706d9e394cbeb7790fa/src/mjlab/utils/lab_api/math.py>`_
  (BSD-3-Clause), together with Isaac Lab math utilities.
* **Observation normalization:** ``rsl_rl.modules.EmpiricalNormalization``;
  license: ``docs/licenses/dependencies/rsl_rl-license.txt``.

Isaac Lab integration code uses BSD-3-Clause. Dataset and robot-asset terms
are provided by their respective distributors.
